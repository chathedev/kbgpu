FROM runpod/base:0.6.2-cuda12.1.0

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_DATASETS_CACHE=/models

WORKDIR /

# Ensure python -> python3 symlink
RUN ln -sf $(which python3) /usr/bin/python 2>/dev/null || true

# ffmpeg
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends ffmpeg && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /requirements.txt

# Skip model download at build — download on first boot instead.
# This avoids build failures from GPU-dependent imports during docker build.
# FlashBoot snapshot will cache the model after first successful boot.

# Copy handler
COPY handler.py /handler.py
COPY test_input.json /test_input.json

CMD ["python3", "-u", "/handler.py"]
