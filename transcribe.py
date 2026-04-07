import logging
from typing import Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

WHISPER_MODEL_PATH = "/models/whisper"
MIN_WORD_PROBABILITY = 0.3


def load_whisper_model() -> WhisperModel:
    """Load KB-Whisper-large from pre-cached path. Call once at module level."""
    logger.info("Loading KB-Whisper-large from /models/whisper...")
    model = WhisperModel(
        "KBLab/kb-whisper-large",
        device="cuda",
        compute_type="float16",
        download_root=WHISPER_MODEL_PATH,
    )
    logger.info("KB-Whisper-large loaded successfully")
    return model


def transcribe(audio_path: str, model: WhisperModel) -> list[dict]:
    """
    Transcribe Swedish audio using KB-Whisper-large with word-level timestamps.

    Returns list of word dicts:
    {"word": str, "start": float, "end": float, "probability": float}
    """
    logger.info(f"Transcribing: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        language="sv",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=3,           # 5→3: ~25% faster, negligible quality loss for Swedish
        temperature=0.0,
        # best_of is ignored when temperature=0.0 — omitted for clarity
    )

    logger.info(f"Detected language: {info.language} (prob={info.language_probability:.2f}), duration={info.duration:.1f}s")

    words = []
    for segment in segments:
        if segment.words is None:
            continue
        for word in segment.words:
            text = word.word.strip()
            if not text:
                continue
            prob = word.probability if word.probability is not None else 1.0
            if prob < MIN_WORD_PROBABILITY:
                logger.debug(f"Filtered low-prob word '{text}' (prob={prob:.3f})")
                continue
            words.append({
                "word": text,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "probability": round(prob, 4),
            })

    logger.info(f"Transcription complete: {len(words)} words")
    return words
