# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR pipeline (KB-Whisper + NeMo diarization)
#
# CUDA version strategy:
#   RunPod workers have driver 550 = CUDA 12.4.
#   nemo_toolkit 2.7+ resolves torch 2.11/cu13 which needs driver ≥560 → crash.
#   Fix: base on cuda:12.4.1-cudnn-runtime (system cuDNN avoids all PyPI
#   cu12/cu13 package conflicts), install torch cu124, then force-reinstall
#   after nemo overwrites it.
#
# Why cudnn-runtime base (not just runtime):
#   - System cuDNN at /usr/lib/x86_64-linux-gnu/libcudnn.so.9 (always in ld path)
#   - No PyPI nvidia-cudnn-cu12/cu13 namespace conflicts
#   - torch cu124 finds system cudnn automatically
#
# Disk budget on GHA free runners: ~14GB after cleanup step (~20GB).
#   cuda:12.4.1-cudnn-runtime = ~3.5GB compressed (~8GB uncompressed)
#   Fits fine with the "Free disk space" step in .github/workflows/build.yml.
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip

# ── 1. System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

# ── 2. PyTorch cu124 ─────────────────────────────────────────────────────────
# Match RunPod worker driver (550.x = CUDA 12.4).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.2.0,<2.6.0" "torchaudio>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. Heavy ML libraries ─────────────────────────────────────────────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 3b. Re-pin torch + torchaudio to cu124 ────────────────────────────────────
# nemo_toolkit resolves torch 2.11+/cu13 (driver ≥560 required — workers have 550).
# Force-reinstall cu124 wheels after nemo overwrites them. --no-deps avoids
# cascading dep changes but leaves the original cu12 nvidia-* packages from
# the first torch install intact.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.2.0,<2.6.0" "torchaudio>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall --no-deps

# ── 3c. Purge cu13 nvidia packages (keep cu12 clean) ──────────────────────────
# NeMo installed nvidia-*-cu13 packages alongside the cu12 ones we need.
# They share the site-packages/nvidia/*/lib namespace and shadow each other.
# Remove all cu13 nvidia packages so only cu12 remains. System cuDNN from
# the base image is at /usr/lib/x86_64-linux-gnu/ (default ld path, not affected).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip uninstall -y $(pip list --format=freeze 2>/dev/null | grep -oE '^nvidia-[a-z0-9-]+-cu13' || true) || true

# ── 4. Light runtime dependencies ────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ── 5. Download and bake model weights into image ─────────────────────────────
WORKDIR /app
COPY download_models.py .
RUN python3 download_models.py

# ── 6. Application code ───────────────────────────────────────────────────────
COPY handler.py transcribe.py diarize.py merge.py preprocess.py ./

CMD ["python3", "-u", "handler.py"]
