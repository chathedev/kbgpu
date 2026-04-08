# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR pipeline (KB-Whisper + NeMo diarization)
#
# CUDA version strategy:
#   RunPod workers have driver 550 = CUDA 12.4.
#   nemo_toolkit 2.7+ resolves torch 2.11/cu13 which needs driver ≥560 → crash.
#   Fix: base on cuda:12.4.1 (no cudnn — saves ~600MB vs cudnn variant),
#   install torch cu124, then force-reinstall after nemo overwrites it.
#
# Disk budget on GHA free runners: ~14GB.
#   cuda:12.4.1-runtime-ubuntu22.04 = ~5.5GB compressed
#   vs cuda:12.4.1-cudnn-runtime = ~7.5GB → runs out of space
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
# torchaudio NOT from pytorch.org — CUDA wheel dlopen-fails on CPU build runners.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. Heavy ML libraries ─────────────────────────────────────────────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 3b. Re-pin torch to cu124 ─────────────────────────────────────────────────
# nemo_toolkit resolves torch 2.11+/cu13 (driver ≥560 required — workers have 550).
# Force-reinstall cu124 wheel after nemo to override it. --no-deps avoids cascade.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall --no-deps

# ── 3c. Register nvidia CUDA libs from Python packages ────────────────────────
# NeMo installs nvidia-cuda-runtime-cu13 etc. as Python packages. Their .so files
# live in site-packages/nvidia/*/lib/ — NOT in the system ld search path.
# At build time (no GPU on GHA runner), importing NeMo would fail with:
#   libcudart.so.13: cannot open shared object file: No such file or directory
# This step adds those paths to ldconfig so dynamic linking works at build time.
RUN python3 -c "import glob, os, site; paths=[p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp,'nvidia/*/lib'))]; open('/etc/ld.so.conf.d/nvidia-pypi.conf','w').write('\n'.join(paths)+'\n') if paths else None; print('Registered',len(paths),'CUDA lib paths')" \
    && ldconfig

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
