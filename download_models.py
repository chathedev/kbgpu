"""
Build-time model pre-download script.
Run once during Docker build; bakes all model weights into the image layer.
"""
import os
import sys

os.makedirs("/models/whisper", exist_ok=True)
os.makedirs("/models/nemo", exist_ok=True)

# ── 1. KB-Whisper-large ──────────────────────────────────────────────────────
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

# ── 2. TitaNet Large (speaker embeddings) ────────────────────────────────────
print("=== Downloading TitaNet Large ===", flush=True)
try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    titanet = EncDecSpeakerLabelModel.from_pretrained("titanet_large")
    titanet.save_to("/models/nemo/titanet-large.nemo")
    del titanet
    print("TitaNet Large: OK -> /models/nemo/titanet-large.nemo", flush=True)
except Exception as e:
    print(f"TitaNet FAILED: {e}", flush=True)
    sys.exit(1)

# ── 3. MSDD telephony (multi-scale diarization decoder) ──────────────────────
print("=== Downloading MSDD telephony ===", flush=True)
try:
    from nemo.collections.asr.models import EncDecDiarLabelModel
    msdd = EncDecDiarLabelModel.from_pretrained("diar_msdd_telephony")
    msdd.save_to("/models/nemo/diar_msdd_telephony.nemo")
    del msdd
    print("MSDD telephony: OK -> /models/nemo/diar_msdd_telephony.nemo", flush=True)
except Exception as e:
    print(f"MSDD FAILED: {e}", flush=True)
    sys.exit(1)

# ── 4. Demucs htdemucs (vocal separation) ────────────────────────────────────
print("=== Downloading Demucs htdemucs ===", flush=True)
try:
    from demucs.pretrained import get_model
    get_model("htdemucs")
    print("Demucs htdemucs: OK", flush=True)
except Exception as e:
    print(f"Demucs FAILED: {e}", flush=True)
    sys.exit(1)

print("=== All models downloaded successfully ===", flush=True)
