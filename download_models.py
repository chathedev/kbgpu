"""
Build-time model pre-download script.
Bakes KB-Whisper-large into the image layer so cold starts don't wait.

pyannote community-1 is downloaded at RUNTIME (first container boot) because
its HuggingFace access token must not live in the image. FlashBoot captures
the post-download state so subsequent wake-ups load the cached pyannote from
local disk — zero network calls after the first boot per image version.

CUDA_VISIBLE_DEVICES='' forces CPU-only at build time (no GPU on GHA).
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.makedirs("/models/whisper", exist_ok=True)
os.makedirs("/models/hf_cache", exist_ok=True)

# ── KB-Whisper-large ─────────────────────────────────────────────────────────
print("=== Downloading KB-Whisper-large ===", flush=True)
try:
    from faster_whisper import WhisperModel
    WhisperModel(
        "KBLab/kb-whisper-large",
        device="cpu",
        compute_type="int8",
        download_root="/models/whisper",
    )
    print("KB-Whisper-large: OK", flush=True)
except Exception as e:
    print(f"KB-Whisper-large FAILED: {e}", flush=True)
    sys.exit(1)

print("=== Build-time model download complete ===", flush=True)
