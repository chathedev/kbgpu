FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch first (specific CUDA version)
RUN pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download KB-Whisper-large at build time for fast cold starts
RUN python3 -c "\
from faster_whisper import WhisperModel; \
print('Downloading KB-Whisper-large...'); \
WhisperModel('KBLab/kb-whisper-large', device='cpu', compute_type='int8', download_root='/models/whisper'); \
print('KB-Whisper-large downloaded successfully')"

# Pre-download NeMo diarization models via Python API (NGC wget requires auth tokens)
# EncDecSpeakerLabelModel = TitaNet (speaker embeddings)
# EncDecDiarLabelModel    = MSDD multi-scale diarizer
RUN mkdir -p /models/nemo && python3 -c "\
import os; \
os.makedirs('/models/nemo', exist_ok=True); \
from nemo.collections.asr.models import EncDecSpeakerLabelModel, EncDecDiarLabelModel; \
print('Downloading TitaNet Large...'); \
titanet = EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large'); \
titanet.save_to('/models/nemo/titanet-large.nemo'); \
print('TitaNet saved to /models/nemo/titanet-large.nemo'); \
del titanet; \
print('Downloading MSDD telephony...'); \
msdd = EncDecDiarLabelModel.from_pretrained('diar_msdd_telephony'); \
msdd.save_to('/models/nemo/diar_msdd_telephony.nemo'); \
print('MSDD saved to /models/nemo/diar_msdd_telephony.nemo'); \
del msdd; \
print('All NeMo models downloaded')"

# Pre-download Demucs htdemucs model
RUN python3 -c "\
import demucs.pretrained; \
print('Downloading Demucs htdemucs...'); \
demucs.pretrained.get_model('htdemucs'); \
print('Demucs htdemucs downloaded successfully')"

# Copy application code
COPY handler.py .
COPY transcribe.py .
COPY diarize.py .
COPY merge.py .
COPY preprocess.py .

CMD ["python3", "-u", "handler.py"]
