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
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python3 /usr/bin/python3.10 1

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

# Pre-download NeMo diarization models from NGC
RUN mkdir -p /models/nemo && \
    wget -q -O /models/nemo/diar_msdd_telephony.nemo \
        "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/diar_msdd_telephony/versions/1.0.1/files/diar_msdd_telephony.nemo" && \
    echo "MSDD telephony model downloaded" && \
    wget -q -O /models/nemo/titanet-large.nemo \
        "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/titanet_large/versions/v1/files/titanet-large.nemo" && \
    echo "TitaNet-large downloaded"

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
