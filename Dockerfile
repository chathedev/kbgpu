FROM runpod/base:0.6.2-cuda12.1.0

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_DATASETS_CACHE=/models

WORKDIR /

# Ensure python -> python3 symlink exists
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ffmpeg (base image already has python3, pip, CUDA, cuDNN)
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends ffmpeg && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Pre-download KB-Whisper model at build time (critical for FlashBoot)
COPY builder/fetch_model.py /fetch_model.py
RUN python /fetch_model.py || (echo "=== FETCH MODEL FAILED ===" && echo "--- script content ---" && cat /fetch_model.py && echo "--- pip list ---" && pip list 2>&1 | grep -i -E "hugging|faster|whisper" && exit 1)
RUN rm -f /fetch_model.py

# Copy handler
COPY handler.py /handler.py
COPY test_input.json /test_input.json

CMD ["python", "-u", "/handler.py"]
