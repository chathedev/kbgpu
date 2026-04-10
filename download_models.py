"""
Build-time model pre-download.

Bakes two models into the image so cold starts don't wait on HF:
    1. KB-Whisper-large (faster-whisper CTranslate2 format) → /models/whisper
    2. DiariZen WavLM speaker-diarization model             → /models/diar

CUDA_VISIBLE_DEVICES='' forces CPU-only at build time (no GPU on CI runners).
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.makedirs("/models/whisper", exist_ok=True)
os.makedirs("/models/diar", exist_ok=True)

os.environ.setdefault("HF_HOME", "/models/diar")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/models/diar")

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

# ── DiariZen pretrained diarization model ───────────────────────────────────
# We use HuggingFace snapshot_download so the weights live on disk without
# having to actually construct the pipeline at build time (the pipeline
# needs a GPU-compatible torch setup that we don't want to spin up here).
DIAR_MODEL_ID = os.environ.get("DIARIZEN_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md")
print(f"=== Downloading DiariZen model: {DIAR_MODEL_ID} ===", flush=True)
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=DIAR_MODEL_ID,
        cache_dir="/models/diar",
        local_dir=None,
    )
    print(f"DiariZen model {DIAR_MODEL_ID}: OK", flush=True)
except Exception as e:
    # Non-fatal: the container will still serve transcription. The diarize
    # pipeline will try to download on first boot and either succeed or
    # return a clear error.
    print(f"DiariZen pre-download FAILED (will retry at boot): {e}", flush=True)

# Also pre-download the WavLM base that DiariZen's WavLM backbone depends on,
# so the first boot doesn't have to fetch it.
try:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="microsoft/wavlm-base-plus", cache_dir="/models/diar")
    print("microsoft/wavlm-base-plus: OK", flush=True)
except Exception as e:
    print(f"wavlm-base-plus pre-download skipped: {e}", flush=True)

print("=== Build-time model download complete ===", flush=True)
