# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR pipeline (KB-Whisper + NeMo diarization)
#
# Layer order is intentional for maximum GHA cache reuse:
#   1. System deps          — changes: never
#   2. PyTorch (CUDA)       — changes: rarely
#   3. Heavy ML libs        — changes: rarely
#   4. Light deps           — changes: occasionally
#   5. Model weights        — changes: when download_models.py changes
#   6. App code             — changes: every commit  ← fast, no pip at all
#
# BuildKit pip cache mounts (--mount=type=cache) make even a cache-miss
# layer fast: packages are read from the runner's local pip cache instead
# of re-downloaded from PyPI.
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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

# ── 2. PyTorch (CUDA) ────────────────────────────────────────────────────────
# Install ONLY torch CUDA here. torchaudio is intentionally NOT installed
# from pytorch.org — the CUDA torchaudio wheel links against libcuda.so which
# doesn't exist on GHA's CPU-only build runners, causing dlopen to fail during
# `download_models.py`. We let nemo_toolkit pull its own compatible torchaudio
# from PyPI (a CPU wheel that dlopen-succeeds everywhere). The CUDA torch is
# all that's needed for GPU tensor ops; torchaudio CPU handles audio I/O fine.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "torch>=2.2.0" \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3. Heavy ML libraries ─────────────────────────────────────────────────────
# nemo_toolkit pulls in ~200 transitive packages (transformers, scipy, etc.)
# faster-whisper is relatively light but listed here since it changes rarely.
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 4. Light runtime dependencies ────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ── 5. Download and bake model weights into image ─────────────────────────────
# This layer is cached by GHA as long as download_models.py doesn't change.
# When it does change, the ~15 min download runs once and is cached again.
WORKDIR /app
COPY download_models.py .
RUN python3 download_models.py

# ── 6. Application code ───────────────────────────────────────────────────────
# Copied last so code changes don't bust the expensive layers above.
COPY handler.py transcribe.py diarize.py merge.py preprocess.py ./

CMD ["python3", "-u", "handler.py"]
