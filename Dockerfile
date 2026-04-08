# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR worker (KB-Whisper transcription ONLY)
#
# Pipeline on RunPod serverless:
#   download audio → ffmpeg 16kHz mono WAV → KB-Whisper → words
#
# Speaker diarization is done separately in the TIVLY backend via the
# pyannoteAI Precision-2 API — keeping this worker laser-focused on Whisper
# gives much faster startup, smaller image, and fewer moving parts.
#
# Target: RunPod workers with GPU driver 550.x → CUDA 12.4 max.
# Using cuda:12.4.1-cudnn-runtime keeps system cuDNN on the default ld path,
# so faster-whisper's CTranslate2 finds libcudnn.so.9 out of the box.
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
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

# ── 2. PyTorch cu124 ─────────────────────────────────────────────────────────
# faster-whisper doesn't need torch for inference (it uses CTranslate2) but
# we install it anyway in case future features need it. Keeps the image
# versatile without measurable cold-start cost — torch is lazy-imported.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.4.0,<2.6.0" "torchaudio>=2.4.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. ML libraries (faster-whisper only — no pyannote, no NeMo) ─────────────
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-ml.txt

# ── 4. Light runtime dependencies ────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ── 5. Download and bake KB-Whisper-large into the image ─────────────────────
WORKDIR /app
COPY download_models.py .
RUN python3 download_models.py

# ── 6. Application code ───────────────────────────────────────────────────────
COPY handler.py transcribe.py preprocess.py ./

CMD ["python3", "-u", "handler.py"]
