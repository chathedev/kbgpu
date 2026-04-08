import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


class DiarizationModels:
    """Pre-loaded pyannote pipeline held in GPU VRAM between jobs."""
    def __init__(self):
        self.pipeline = None  # pyannote.audio.Pipeline


def load_diarization_models() -> DiarizationModels:
    """
    Load pyannote speaker-diarization-community-1 once at container startup.
    The pipeline weights are already in the HF cache (baked by download_models.py),
    so this is just a local file load — no network calls.
    """
    import torch
    from pyannote.audio import Pipeline

    models = DiarizationModels()

    logger.info(f"Loading {PYANNOTE_MODEL} from local HF cache...")
    try:
        token = os.environ.get("HF_TOKEN")
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            use_auth_token=token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info("pyannote pipeline moved to CUDA")
        else:
            logger.warning("CUDA not available — pyannote running on CPU")
        models.pipeline = pipeline
        logger.info("pyannote community-1 loaded")
    except Exception as e:
        logger.exception(f"pyannote load failed: {e}")
        raise

    return models


def diarize(
    audio_path: str,
    diar_models: Optional[DiarizationModels] = None,
    num_speakers: Optional[int] = None,
    job_id: Optional[str] = None,
) -> list[dict]:
    """
    Run pyannote speaker diarization on a mono 16kHz WAV.

    Args:
        audio_path: path to audio file (any format pyannote can read)
        diar_models: pre-loaded DiarizationModels (required for fast path)
        num_speakers: optional exact number of speakers. If None, pyannote
                      auto-detects.
        job_id: for logging only.

    Returns segments sorted by start time:
        [{"speaker": str, "start": float, "end": float}, ...]
    """
    import torch

    if diar_models is None or diar_models.pipeline is None:
        raise RuntimeError("diar_models must be pre-loaded via load_diarization_models()")

    logger.info(f"Diarizing: {audio_path}, num_speakers={num_speakers}")

    kwargs = {}
    if num_speakers and num_speakers >= 1:
        kwargs["num_speakers"] = int(num_speakers)

    with torch.no_grad():
        annotation = diar_models.pipeline(audio_path, **kwargs)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "speaker": str(speaker),       # "SPEAKER_00", "SPEAKER_01", ...
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
        })

    segments.sort(key=lambda s: s["start"])

    # Post-merge: glue same-speaker segments with gap <1s. pyannote is cleaner
    # than NeMo here but some fragmentation still happens on crosstalk.
    before = len(segments)
    segments = _post_merge_segments(segments, max_gap=1.0)

    num_spk = len({s["speaker"] for s in segments})
    logger.info(
        f"Diarization done: {before}→{len(segments)} segments (post-merged), "
        f"{num_spk} speakers"
    )
    return segments


def _post_merge_segments(segments: list[dict], max_gap: float = 1.0) -> list[dict]:
    """
    Merge consecutive same-speaker segments if the gap between them is
    < max_gap seconds. Removes the "same speaker gets 2 slots" artifact.
    """
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and (seg["start"] - last["end"]) < max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged
