"""
Build-time model pre-download script.
Run once during Docker build; bakes all model weights into the image layer.

Note: CUDA_VISIBLE_DEVICES='' forces CPU-only mode because GHA runners have no GPU.
NeMo CUDA libs (libcudart.so.13 etc.) must be registered via ldconfig before this
script runs — see Dockerfile step 3c.
"""
import os
import sys
import json
import tempfile
import shutil

# Force CPU-only at build time — GHA runners have no GPU.
# This prevents NeMo from attempting CUDA device access during model download.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
    titanet = EncDecSpeakerLabelModel.from_pretrained("titanet_large", map_location="cpu")
    titanet.save_to("/models/nemo/titanet-large.nemo")
    del titanet
    print("TitaNet Large: OK -> /models/nemo/titanet-large.nemo", flush=True)
except Exception as e:
    print(f"TitaNet FAILED: {e}", flush=True)
    sys.exit(1)

# ── 3. VAD model (vad_multilingual_marblenet) ─────────────────────────────────
# Pre-baking prevents ~50MB NGC download + model init on every cold start.
# NeMo 2.x uses EncDecClassificationModel or EncDecFrameClassificationModel
# for MarbleNet VAD — try both since the class changed between versions.
print("=== Downloading VAD model (vad_multilingual_marblenet) ===", flush=True)
vad_ok = False
VAD_CLASSES = [
    ("nemo.collections.asr.models", "EncDecFrameClassificationModel"),
    ("nemo.collections.asr.models", "EncDecClassificationModel"),
]
for mod_name, cls_name in VAD_CLASSES:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        vad = cls.from_pretrained("vad_multilingual_marblenet", map_location="cpu")
        vad.save_to("/models/nemo/vad_multilingual_marblenet.nemo")
        del vad
        print(f"VAD ({cls_name}): OK -> /models/nemo/vad_multilingual_marblenet.nemo", flush=True)
        vad_ok = True
        break
    except Exception as e:
        print(f"VAD ({cls_name}): {e}", flush=True)

if not vad_ok:
    # Fallback: trigger download via a dummy ClusteringDiarizer init
    print("Trying VAD download via ClusteringDiarizer init...", flush=True)
    try:
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ClusteringDiarizer

        tmp_dir = tempfile.mkdtemp()
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        # Write a minimal manifest pointing to /dev/null (0 bytes, just triggers model load)
        with open(manifest_path, "w") as f:
            json.dump({
                "audio_filepath": "/dev/null",
                "offset": 0, "duration": 0.01,
                "label": "infer", "text": "-",
                "num_speakers": None, "rttm_filepath": None, "uem_filepath": None,
            }, f)
            f.write("\n")

        cfg = OmegaConf.create({
            "num_workers": 0, "sample_rate": 16000, "batch_size": 1,
            "device": "cpu", "verbose": False,
            "diarizer": {
                "manifest_filepath": manifest_path,
                "out_dir": tmp_dir, "oracle_vad": False,
                "collar": 0.25, "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "external_vad_manifest": None,
                    "parameters": {
                        "window_length_in_sec": 0.15,
                        "shift_length_in_sec": 0.01,
                        "smoothing": "median", "overlap": 0.5,
                        "onset": 0.8, "offset": 0.6,
                        "pad_onset": 0.1, "pad_offset": -0.1,
                        "min_duration_on": 0.2, "min_duration_off": 0.2,
                        "filter_speech_first": True,
                    },
                },
                "speaker_embeddings": {
                    "model_path": "/models/nemo/titanet-large.nemo" if os.path.exists("/models/nemo/titanet-large.nemo") else "titanet_large",
                    "parameters": {
                        "window_length_in_sec": [1.5, 1.0, 0.5],
                        "shift_length_in_sec": [0.75, 0.5, 0.25],
                        "multiscale_weights": [1, 1, 1],
                        "save_embeddings": False,
                    },
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False,
                    },
                },
            },
        })

        # Just instantiating ClusteringDiarizer triggers VAD model download
        ClusteringDiarizer(cfg=cfg)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("VAD model downloaded via ClusteringDiarizer init: OK", flush=True)
        vad_ok = True
    except Exception as e:
        print(f"VAD fallback also failed (non-fatal): {e}", flush=True)
        print("VAD will download from NGC at first job runtime", flush=True)

# ── 4. MSDD (multi-scale diarization decoder) — optional ─────────────────────
print("=== Downloading MSDD model (optional) ===", flush=True)
MSDD_CANDIDATES = [
    "diar_msdd_telephony",
    "diar_msdd_meeting",
]
msdd_ok = False
for model_name in MSDD_CANDIDATES:
    try:
        from nemo.collections.asr.models import EncDecDiarLabelModel
        msdd = EncDecDiarLabelModel.from_pretrained(model_name, map_location="cpu")
        msdd.save_to("/models/nemo/diar_msdd.nemo")
        del msdd
        print(f"MSDD ({model_name}): OK -> /models/nemo/diar_msdd.nemo", flush=True)
        msdd_ok = True
        break
    except Exception as e:
        print(f"MSDD ({model_name}) not available: {e}", flush=True)

if not msdd_ok:
    print("MSDD: No model available — clustering-only diarization", flush=True)

print("=== All required models downloaded successfully ===", flush=True)
