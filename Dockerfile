FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_DATASETS_CACHE=/models

WORKDIR /

# System deps
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends \
    sudo ca-certificates git wget curl bash \
    ffmpeg build-essential \
    python3.10 python3.10-dev python3.10-venv python3-pip -y && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Pre-download KB-Whisper model at build time (critical for FlashBoot)
COPY builder/fetch_model.py /fetch_model.py
RUN python /fetch_model.py && rm /fetch_model.py

# Copy handler
COPY handler.py /handler.py
COPY test_input.json /test_input.json

CMD ["python", "-u", "/handler.py"]
