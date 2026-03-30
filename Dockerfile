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

# Download model at BUILD TIME — baked into image for instant worker starts
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('KBLab/kb-whisper-large', local_dir='/models/kb-whisper-large', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*','onnx/*'])"

# Verify model files exist
RUN ls -la /models/kb-whisper-large/ && python3 -c "import os; files=os.listdir('/models/kb-whisper-large'); print(f'Model files: {len(files)}'); assert len(files)>3, 'Model download incomplete'"

# Copy handler
COPY handler.py /handler.py
COPY test_input.json /test_input.json

CMD ["python3", "-u", "/handler.py"]
