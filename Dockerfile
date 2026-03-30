FROM runpod/base:0.6.2-cuda12.1.0

ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_DATASETS_CACHE=/models

WORKDIR /

# Debug: find python location in this base image
RUN which python3 || echo "NO python3" && \
    which python || echo "NO python" && \
    ls -la /usr/bin/python* || true

# Ensure python -> python3 symlink
RUN if command -v python3 &>/dev/null; then ln -sf "$(which python3)" /usr/bin/python; fi

# ffmpeg
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends ffmpeg && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /requirements.txt

# Verify installs
RUN python3 -c "import huggingface_hub; print('huggingface_hub OK:', huggingface_hub.__version__)"
RUN python3 -c "import faster_whisper; print('faster_whisper OK')"

# Pre-download KB-Whisper model at build time (critical for FlashBoot)
COPY builder/fetch_model.py /fetch_model.py
RUN python3 -u /fetch_model.py
RUN rm -f /fetch_model.py

# Copy handler
COPY handler.py /handler.py
COPY test_input.json /test_input.json

CMD ["python3", "-u", "/handler.py"]
