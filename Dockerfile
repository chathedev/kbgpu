# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR pipeline
#   KB-Whisper-large (faster-whisper / CTranslate2)
#   pyannote/speaker-diarization-community-1
#
# Target: RunPod serverless, GPU driver 550 → CUDA 12.4 max.
# Base cuda:12.4.1-cudnn-runtime ships system libcudnn.so.9 (default ld path),
# so torch cu124 finds cuDNN without PyPI package gymnastics.
#
# No more NeMo — pyannote is simpler, faster, cleaner deps.
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    HF_HOME=/models/hf_cache \
    HF_HUB_CACHE=/models/hf_cache/hub

# HF_TOKEN must be provided at runtime via the RunPod endpoint template env
# vars. It is used to download pyannote community-1 the first time the
# container starts — the HF cache is then captured by FlashBoot snapshots,
# so subsequent wake-ups load from local disk without any network calls.

# ── 1. System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

# ── 2. PyTorch cu124 ─────────────────────────────────────────────────────────
# Match RunPod worker driver (550.x = CUDA 12.4). Install torch AND its full
# dependency tree (no --no-deps) — without NeMo in the mix, there are no
# conflicts to work around.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.2.0,<2.6.0" "torchaudio>=2.2.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. ML libraries (pyannote + faster-whisper) ──────────────────────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 3b. Re-pin torch + torchaudio to cu124 ────────────────────────────────────
# pyannote.audio 4.0 resolves torch >=2.11/cu13 which needs driver >=560.
# RunPod workers have driver 550 (CUDA 12.4 max) → "driver too old" RuntimeError.
# Force-reinstall cu124 wheels of torch+torchaudio with --no-deps after pyannote.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.4.0,<2.6.0" "torchaudio>=2.4.0,<2.6.0" \
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
