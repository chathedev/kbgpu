"""Pre-download KB-Whisper Large at build time so it's baked into the image."""
import sys
import os

os.makedirs("/models", exist_ok=True)

try:
    from huggingface_hub import snapshot_download
    print("Downloading KBLab/kb-whisper-large to /models ...", flush=True)
    snapshot_download(
        repo_id="KBLab/kb-whisper-large",
        local_dir="/models/kb-whisper-large",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "onnx/*"],
    )
    print("Model downloaded successfully.", flush=True)
except Exception as e:
    print(f"FETCH MODEL ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
