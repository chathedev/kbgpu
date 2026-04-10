import logging
import os
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline

logger = logging.getLogger(__name__)

WHISPER_MODEL_PATH = "/models/whisper"
MIN_WORD_PROBABILITY = 0.3

# Batch size for BatchedInferencePipeline.
# Auto-detect based on available VRAM: more VRAM = bigger batches = faster.
# KB-Whisper-large uses ~3GB base + ~0.5GB per batch item.
def _auto_batch_size():
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb >= 70: return 24   # H100/A100 80GB
            if vram_gb >= 40: return 16   # A40/L40S/A6000 48GB
            if vram_gb >= 20: return 8    # RTX 4090/3090 24GB
        return 8
    except Exception:
        return 8

BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "0")) or _auto_batch_size()


def _verify_cuda_or_die():
    """
    Hard-fail at startup if CTranslate2 can't see a CUDA device.
    Without this, faster-whisper happily runs on CPU at 100% load and
    0% GPU utilization — silent degradation.
    """
    import ctranslate2
    n = ctranslate2.get_cuda_device_count()
    if n <= 0:
        raise RuntimeError(
            "CTranslate2 sees 0 CUDA devices — faster-whisper would fall "
            "back to CPU and run 50x slower. Aborting worker boot."
        )
    compute_types = ctranslate2.get_supported_compute_types("cuda")
    logger.info(
        f"CTranslate2 CUDA OK: {n} device(s), compute_types={sorted(compute_types)}"
    )
    return n


def load_whisper_model() -> BatchedInferencePipeline:
    """
    Load KB-Whisper-large + wrap in BatchedInferencePipeline for fast
    parallel chunk inference. Call once at module level.
    """
    _verify_cuda_or_die()

    logger.info(f"Loading KB-Whisper-large (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    model = WhisperModel(
        "KBLab/kb-whisper-large",
        device="cuda",
        compute_type="float16",
        download_root=WHISPER_MODEL_PATH,
        # num_workers: how many parallel audio streams can the model process.
        # BatchedInferencePipeline handles intra-audio batching itself; we keep
        # num_workers=1 to avoid double-scheduling that hurts cache locality.
        num_workers=1,
        # cpu_threads controls the tokenizer + feature extractor CPU pool.
        # Auto (0) defaults to num cores (16 on RunPod) which creates heavy
        # thread contention during GPU inference. 4 is a sweet spot.
        cpu_threads=4,
    )
    pipeline = BatchedInferencePipeline(model=model)
    logger.info(f"KB-Whisper-large + BatchedInferencePipeline loaded in {time.time()-t0:.1f}s")
    return pipeline


def transcribe(audio_path: str, pipeline: BatchedInferencePipeline) -> list[dict]:
    """
    Transcribe Swedish audio using KB-Whisper-large with word-level timestamps.

    Uses BatchedInferencePipeline so multiple audio chunks run through the
    GPU in parallel → ~10-15x faster than the default serial inference.

    Returns list of word dicts:
      [{"word": str, "start": float, "end": float, "probability": float}, ...]
    """
    t0 = time.time()
    logger.info(f"Transcribing: {audio_path} (batch_size={BATCH_SIZE})")

    segments, info = pipeline.transcribe(
        audio_path,
        language="sv",
        word_timestamps=True,
        # VAD is still on — but with batching it runs much faster since
        # the batched pipeline amortizes VAD overhead across chunks.
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 200,
        },
        beam_size=1,                        # greedy decode: ~2x faster than beam=3, negligible quality loss for Swedish
        temperature=0.0,
        # Quality knobs — suppress hallucination / repetition loops on messy audio
        condition_on_previous_text=False,   # don't let prior garbage poison next chunk
        no_speech_threshold=0.6,            # higher → skip more non-speech regions
        compression_ratio_threshold=2.4,    # reject repeated-token hallucinations
        log_prob_threshold=-1.0,            # reject low-confidence segments
        batch_size=BATCH_SIZE,
    )

    logger.info(
        f"Detected language: {info.language} (prob={info.language_probability:.2f}), "
        f"duration={info.duration:.1f}s"
    )

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
                continue
            words.append({
                "word": text,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "probability": round(prob, 4),
            })

    elapsed = time.time() - t0
    rtf = (elapsed / info.duration) if info.duration > 0 else 0
    logger.info(
        f"Transcription complete: {len(words)} words in {elapsed:.1f}s "
        f"(audio {info.duration:.0f}s, RTF {rtf:.3f}x)"
    )
    return words
