"""Pre-download KB-Whisper Large at build time so it's baked into the image."""
from huggingface_hub import snapshot_download
import os

os.makedirs("/models", exist_ok=True)
print("Downloading KBLab/kb-whisper-large to /models ...")
snapshot_download(
    repo_id="KBLab/kb-whisper-large",
    local_dir="/models/kb-whisper-large",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "onnx/*"],
)
print("Model downloaded and cached.")
