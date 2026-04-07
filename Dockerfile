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

# Install PyTorch with CUDA 12.1 support FIRST (must come before nemo_toolkit
# so the CUDA build is not overwritten by a CPU build from PyPI)
RUN pip install --no-cache-dir \
    "torch>=2.2.0" \
    "torchaudio>=2.2.0" \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining requirements
# torch/torchaudio intentionally excluded from requirements.txt to avoid
# pip pulling the CPU build from PyPI on top of the CUDA build above
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model download script and run it
# Separate file (not inline -c) so errors show the exact line that failed
COPY download_models.py .
RUN python3 download_models.py

# Copy application code
COPY handler.py .
COPY transcribe.py .
COPY diarize.py .
COPY merge.py .
COPY preprocess.py .

CMD ["python3", "-u", "handler.py"]
