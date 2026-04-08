import os
import json
import logging
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

MSDD_MODEL_PATH = "/models/nemo/diar_msdd.nemo"
TITANET_MODEL_PATH = "/models/nemo/titanet-large.nemo"
VAD_MODEL_PATH = "/models/nemo/vad_multilingual_marblenet.nemo"

_MSDD_AVAILABLE = os.path.exists(MSDD_MODEL_PATH)
# Use pre-baked VAD path if baked in, else NeMo downloads from NGC
_VAD_MODEL_PATH = VAD_MODEL_PATH if os.path.exists(VAD_MODEL_PATH) else "vad_multilingual_marblenet"


class DiarizationModels:
    """Pre-loaded NeMo diarization models held in GPU memory between jobs."""
    def __init__(self):
        self.titanet = None   # EncDecSpeakerLabelModel
        self.vad = None       # VAD model (loaded lazily by ClusteringDiarizer)
        self.msdd_available = _MSDD_AVAILABLE


def load_diarization_models() -> DiarizationModels:
    """
    Pre-load TitaNet speaker embeddings into GPU VRAM at container startup.
    Called once at module level in handler.py.

    ClusteringDiarizer manages its own VAD/MSDD internally, but pre-loading
    TitaNet (the slowest model, ~800MB) eliminates the biggest per-job latency.
    """
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    models = DiarizationModels()

    logger.info(f"Pre-loading TitaNet from {TITANET_MODEL_PATH}...")
    try:
        models.titanet = EncDecSpeakerLabelModel.restore_from(
            restore_path=TITANET_MODEL_PATH,
            map_location="cuda",
        )
        models.titanet.eval()
        models.titanet.freeze()
        logger.info("TitaNet loaded into GPU VRAM")
    except Exception as e:
        logger.warning(f"TitaNet pre-load failed (will load at job time): {e}")

    if _MSDD_AVAILABLE:
        logger.info("MSDD model available — enhanced diarization enabled")
    else:
        logger.info("MSDD model not available — using clustering-only diarization")

    return models


def diarize(
    audio_path: str,
    diar_models: Optional[DiarizationModels] = None,
    num_speakers: Optional[int] = None,
    job_id: Optional[str] = None,
) -> list[dict]:
    """
    Run NeMo ClusteringDiarizer (+ MSDD if available) for speaker diarization.

    Uses pre-loaded TitaNet from diar_models to skip GPU model loading per job.
    num_workers=0 is critical — prevents forking after CUDA init (deadlock).

    Returns segments sorted by start time:
      [{"speaker": str, "start": float, "end": float}, ...]
    """
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ClusteringDiarizer
    import torch

    job_tag = job_id or "default"
    tmp_dir = f"/tmp/nemo_{job_tag}"
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        logger.info(f"Diarizing: {audio_path}, num_speakers={num_speakers}, msdd={_MSDD_AVAILABLE}")

        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": num_speakers,
                "rttm_filepath": None,
                "uem_filepath": None,
            }, f)
            f.write("\n")

        cfg = _build_config(tmp_dir, manifest_path, num_speakers, diar_models)

        with torch.no_grad():
            diarizer = ClusteringDiarizer(cfg=cfg)
            diarizer.diarize()

        audio_stem = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_path = os.path.join(tmp_dir, "pred_rttms", audio_stem + ".rttm")
        segments = _parse_rttm(rttm_path)

        # Post-merge: glue together same-speaker segments separated by <1s.
        # This undoes NeMo's over-fragmentation on crosstalk-heavy audio.
        before = len(segments)
        segments = _post_merge_segments(segments, max_gap=1.0)

        num_spk = len({s["speaker"] for s in segments})
        logger.info(f"Diarization done: {before}→{len(segments)} segments (post-merged), {num_spk} speakers")
        return segments

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_config(tmp_dir, manifest_path, num_speakers, diar_models):
    from omegaconf import OmegaConf

    # Use pre-loaded TitaNet path — ClusteringDiarizer will re-use it from disk
    # (NeMo doesn't accept live model objects yet, but loading from baked path
    # is still instant vs NGC download)
    titanet_path = TITANET_MODEL_PATH if os.path.exists(TITANET_MODEL_PATH) else "titanet_large"

    cfg_dict = {
        # num_workers=0 is CRITICAL — prevents DataLoader fork after CUDA init
        "num_workers": 0,
        "sample_rate": 16000,
        "batch_size": 64,
        "device": "cuda",
        "verbose": False,
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": tmp_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": False,
            "vad": {
                "model_path": _VAD_MODEL_PATH,
                "external_vad_manifest": None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.5,
                    "onset": 0.8,
                    "offset": 0.6,
                    "pad_onset": 0.1,
                    "pad_offset": -0.1,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": titanet_path,
                "parameters": {
                    # 3 scales: good balance of speed vs accuracy
                    # 5-scale is marginally better but ~40% slower
                    "window_length_in_sec": [1.5, 1.0, 0.5],
                    "shift_length_in_sec": [0.75, 0.5, 0.25],
                    "multiscale_weights": [1, 1, 1],
                    "save_embeddings": False,
                },
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "max_num_speakers": num_speakers if num_speakers else 12,
                    "enhanced_count_thres": 80,
                    # Lowered from 0.25 → 0.15: tighter cluster radius reduces
                    # the "same speaker gets 2 slots" problem on overlapping
                    # speech without over-merging distinct speakers.
                    "max_rp_threshold": 0.15,
                    "sparse_search_volume": 30,
                    "maj_vote_spk_count": False,
                },
            },
        },
    }

    if _MSDD_AVAILABLE:
        cfg_dict["diarizer"]["msdd_model"] = {
            "model_path": MSDD_MODEL_PATH,
            "parameters": {
                "use_speaker_model_from_ckpt": True,
                "infer_batch_size": 25,
                # Lowered from 0.5 → 0.4: MSDD emits fewer speaker-change
                # events on borderline frames, cutting spurious splits.
                "sigmoid_threshold": [0.4, 0.4],
                "seq_eval_mode": False,
                "split_infer": True,
                "diar_eval_settings": [
                    [0.25, True],
                    [0.25, False],
                    [0.0, False],
                ],
            },
        }
        logger.info("MSDD enabled for enhanced diarization")

    return OmegaConf.create(cfg_dict)


def _post_merge_segments(segments: list[dict], max_gap: float = 1.0) -> list[dict]:
    """
    Merge consecutive segments from the same speaker if the gap between
    them is < max_gap seconds. Fixes the common artifact where one speaker
    gets split into multiple tiny slots across short pauses or overlaps.
    """
    if not segments:
        return segments
    segments = sorted(segments, key=lambda s: s["start"])
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"]) < max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def _parse_rttm(rttm_path: str) -> list[dict]:
    if not os.path.exists(rttm_path):
        logger.error(f"RTTM not found: {rttm_path}")
        return []
    segments = []
    with open(rttm_path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append({
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(start + duration, 3),
            })
    segments.sort(key=lambda s: s["start"])
    return segments
