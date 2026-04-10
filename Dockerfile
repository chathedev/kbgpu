# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# kbgpu — Swedish ASR + speaker diarization worker
#
# Pipelines on RunPod serverless:
#   mode=transcribe: download audio → ffmpeg 16kHz mono WAV → KB-Whisper → words
#   mode=diarize   : download audio → DiariZen → speaker segments
#   mode=separate  : denoise / isolate a speaker sample clip
#
# Target: RunPod workers with GPU driver 550.x → CUDA 12.4 max.
# Using cuda:12.4.1-cudnn-runtime keeps system cuDNN on the default ld path,
# so faster-whisper's CTranslate2 finds libcudnn.so.9 out of the box.
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    HF_HOME=/models/diar \
    HUGGINGFACE_HUB_CACHE=/models/diar

# ── 1. System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

# ── 2. PyTorch cu124 ─────────────────────────────────────────────────────────
# faster-whisper doesn't strictly need torch (it uses CTranslate2) but
# DiariZen / pyannote.audio do. Install once, let both consume it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.4.0,<2.6.0" "torchaudio>=2.4.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. ML libraries (faster-whisper + DiariZen deps) ─────────────────────────
# Use the legacy resolver so pip doesn't spend minutes backtracking across
# 30+ lightning/fsspec/torchmetrics version combinations. All versions
# are pinned in requirements-ml.txt.
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --use-deprecated=legacy-resolver -r /tmp/requirements-ml.txt

# ── 4. DiariZen from source ──────────────────────────────────────────────────
# DiariZen ships the pipeline code + a pyannote-audio fork as a submodule.
# We `pip install --no-deps -e .` so it doesn't try to downgrade torch or
# pyannote (those are already installed from requirements-ml.txt).
RUN git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git /opt/diarizen \
    && cd /opt/diarizen \
    && git submodule update --init --recursive || true
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /opt/diarizen && pip install --no-deps -e .
# Install the bundled pyannote-audio fork if present (overrides the pip one).
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -d /opt/diarizen/pyannote-audio ]; then \
        cd /opt/diarizen/pyannote-audio && pip install --no-deps -e .; \
    fi

# ── 5. Light runtime dependencies ────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ── 6. Pre-download models into image layer ─────────────────────────────────
WORKDIR /app
COPY download_models.py .
RUN python3 download_models.py

# ── 7. Application code ───────────────────────────────────────────────────────
COPY handler.py transcribe.py preprocess.py diarize.py ./

CMD ["python3", "-u", "handler.py"]
