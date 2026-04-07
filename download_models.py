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

# ── 3. VAD model (MarbleNet — used by NeMo diarizer) ─────────────────────────
# Without pre-baking this, every cold-start downloads ~50MB from NGC at job time.
print("=== Downloading VAD model (vad_multilingual_marblenet) ===", flush=True)
try:
    from nemo.collections.asr.models import MarbleNetModel
    vad = MarbleNetModel.from_pretrained("vad_multilingual_marblenet")
    vad.save_to("/models/nemo/vad_multilingual_marblenet.nemo")
    del vad
    print("VAD model: OK -> /models/nemo/vad_multilingual_marblenet.nemo", flush=True)
except Exception as e:
    # Non-fatal: NeMo will fall back to downloading at runtime
    print(f"VAD model download failed (non-fatal): {e}", flush=True)

# ── 4. MSDD (multi-scale diarization decoder) — optional ─────────────────────
# NeMo 2.x renamed/restructured MSDD models. Try known names, warn on failure.
print("=== Downloading MSDD model (optional) ===", flush=True)
MSDD_CANDIDATES = [
    "diar_msdd_telephony",   # NeMo 1.x name
    "diar_msdd_meeting",     # NeMo 2.x name
]
msdd_ok = False
for model_name in MSDD_CANDIDATES:
    try:
        from nemo.collections.asr.models import EncDecDiarLabelModel
        msdd = EncDecDiarLabelModel.from_pretrained(model_name)
        msdd.save_to("/models/nemo/diar_msdd.nemo")
        del msdd
        print(f"MSDD ({model_name}): OK -> /models/nemo/diar_msdd.nemo", flush=True)
        msdd_ok = True
        break
    except Exception as e:
        print(f"MSDD ({model_name}) not available: {e}", flush=True)

if not msdd_ok:
    print("MSDD: No model available — diarizer will use clustering-only (still effective)", flush=True)

print("=== All required models downloaded successfully ===", flush=True)
