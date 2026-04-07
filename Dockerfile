# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR pipeline (KB-Whisper + NeMo diarization)
#
# Layer order is intentional for maximum GHA cache reuse:
#   1. System deps          — changes: never
#   2. PyTorch cu121        — changes: rarely
#   3. Heavy ML libs        — changes: rarely
#   3b. Pin torch back      — NeMo upgrades torch to cu13; re-pin to cu121
#   4. Light deps           — changes: occasionally
#   5. Model weights        — changes: when download_models.py changes
#   6. App code             — changes: every commit  ← fast, no pip at all
#
# KEY ISSUE: nemo_toolkit 2.x resolves to torch 2.11 + nvidia-cuda-runtime-13
# which requires CUDA driver ≥560. RunPod workers have driver 550 (CUDA 12.4).
# Fix: force-reinstall torch cu121 AFTER nemo so the driver-compatible wheel wins.
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

# ── 2. PyTorch (CUDA 12.4) ───────────────────────────────────────────────────
# Install torch cu124 to match RunPod worker driver (550.x = CUDA 12.4).
# torchaudio NOT from pytorch.org (CUDA wheel dlopen fails on CPU build runners).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "torch>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. Heavy ML libraries ─────────────────────────────────────────────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 3b. Re-pin torch to cu124 ─────────────────────────────────────────────────
# nemo_toolkit resolves latest torch (2.11+, cu13) which needs driver ≥560.
# Force-reinstall cu124 wheel so we always use the driver-compatible version.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "torch>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall --no-deps

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
