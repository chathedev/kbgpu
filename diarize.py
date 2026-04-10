"""
Speaker diarization module for kbgpu — powered by DiariZen
(BUT-FIT/diarizen-wavlm-large-s80-md).

The pipeline runs on the same GPU the Whisper model is loaded on. Model is
downloaded at build time via download_models.py and cached under /models/diar.

Public surface:
    load_diarization_pipeline() -> pipeline object (load once at boot)
    diarize(audio_path, pipeline, num_speakers=None) -> list[dict]

Segment output shape:
    [{"speaker": "speaker_0", "start": 0.12, "end": 3.45}, ...]
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

DIAR_MODEL_ID = os.environ.get("DIARIZEN_MODEL", "BUT-FIT/diarizen-wavlm-large-s80-md")
DIAR_CACHE_DIR = "/models/diar"


def load_diarization_pipeline():
    """
    Load the DiariZen pipeline once at container boot. Returns a callable
    pipeline object. Raises on failure — caller decides whether diarize
    mode is usable.
    """
    os.makedirs(DIAR_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", DIAR_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", DIAR_CACHE_DIR)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — diarization requires GPU.")

    t0 = time.time()
    logger.info(f"Loading DiariZen pipeline: {DIAR_MODEL_ID}")

    try:
        from diarizen.pipelines.inference import DiariZenPipeline  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            f"diarizen package not installed in container: {err}"
        )

    pipeline = DiariZenPipeline.from_pretrained(DIAR_MODEL_ID)

    # Push whatever torch modules live inside the pipeline onto CUDA.
    try:
        pipeline.to(torch.device("cuda"))
    except Exception as move_err:
        logger.warning(f"pipeline.to(cuda) failed (non-fatal): {move_err}")

    logger.info(f"DiariZen pipeline loaded in {time.time()-t0:.1f}s")
    return pipeline


def _normalize_speaker_label(raw_label) -> str:
    """
    DiariZen yields labels like "SPEAKER_00" or plain ints. Normalize to
    "speaker_<N>" so downstream merge logic is consistent.
    """
    if raw_label is None:
        return "speaker_0"
    s = str(raw_label).strip()
    # "SPEAKER_00" → "speaker_0"
    lower = s.lower()
    if lower.startswith("speaker_"):
        tail = lower[len("speaker_"):]
        try:
            return f"speaker_{int(tail)}"
        except ValueError:
            return lower
    # pure int
    try:
        return f"speaker_{int(s)}"
    except ValueError:
        return lower.replace(" ", "_") or "speaker_0"


def diarize(audio_path: str, pipeline, num_speakers=None, min_speakers=None, max_speakers=None) -> list:
    """
    Run diarization on an audio file and return normalized speaker segments.

    Inputs:
        audio_path: path to a 16kHz mono WAV (or anything torchaudio reads).
        pipeline: object returned by load_diarization_pipeline().
        num_speakers / min_speakers / max_speakers: optional hints.

    Returns:
        [{"speaker": "speaker_0", "start": float, "end": float}, ...]
        sorted by start time.
    """
    t0 = time.time()

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = int(num_speakers)
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    # DiariZenPipeline is callable on a file path and returns a pyannote
    # Annotation object (itertracks yields (Segment, track_id, label)).
    try:
        annotation = pipeline(audio_path, **kwargs) if kwargs else pipeline(audio_path)
    except TypeError:
        # Some pipeline builds don't take the hint kwargs — retry without.
        annotation = pipeline(audio_path)

    segments = []
    try:
        for turn, _track, label in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": _normalize_speaker_label(label),
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
            })
    except AttributeError:
        # If the pipeline returns a plain list of segments already.
        for seg in annotation:
            start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
            end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
            label = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "label", None)
            if start is None or end is None:
                continue
            segments.append({
                "speaker": _normalize_speaker_label(label),
                "start": round(float(start), 3),
                "end": round(float(end), 3),
            })

    segments.sort(key=lambda s: s["start"])

    speaker_set = {s["speaker"] for s in segments}
    logger.info(
        f"Diarization complete: {len(segments)} segments, "
        f"{len(speaker_set)} speakers in {time.time()-t0:.1f}s"
    )
    return segments
