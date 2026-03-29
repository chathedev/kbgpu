"""Pre-download KB-Whisper Large at build time so it's baked into the image."""
import os
os.environ["HF_HOME"] = "/models"

from huggingface_hub import snapshot_download

print("Downloading KBLab/kb-whisper-large to /models ...")
snapshot_download(
    repo_id="KBLab/kb-whisper-large",
    cache_dir="/models",
    local_dir="/models/KBLab/kb-whisper-large",
)
print("Model downloaded and cached.")
