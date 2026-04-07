import os
import time
import logging
import requests
import runpod
import soundfile as sf
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model loading — runs once per warm container, never inside handler
# ---------------------------------------------------------------------------
from transcribe import load_whisper_model

logger.info("Loading models...")
_whisper_model = load_whisper_model()
logger.info("Models loaded successfully")

# ---------------------------------------------------------------------------
# Import pipeline modules (after models loaded so logging is set up)
# ---------------------------------------------------------------------------
from preprocess import preprocess_audio
from transcribe import transcribe
from diarize import diarize
from merge import merge


def handler(job: dict) -> dict:
    """
    RunPod serverless handler for Swedish meeting transcription.

    Input:
      audio_url: str         — URL to download audio file
      num_speakers: int|null — optional speaker count hint
      job_id: str|null       — optional job ID for logging/temp file naming

    Output:
      utterances: list of {speaker, start, end, text}
      words: list of {word, start, end, speaker}
      num_speakers: int
      duration_seconds: float
      processing_time_seconds: float
      audio_url: str
    """
    job_input = job.get("input", {})
    audio_url = job_input.get("audio_url")
    num_speakers = job_input.get("num_speakers")
    job_id = job_input.get("job_id") or job.get("id") or "unknown"

    logger.info(f"[{job_id}] Job received: audio_url={audio_url}, num_speakers={num_speakers}")

    if not audio_url:
        return {"error": "Missing required field: audio_url", "audio_url": None}

    start_time = time.time()
    downloaded_path = None
    preprocessed_path = None

    try:
        # 1. Download audio
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        logger.info(f"[{job_id}] Download: {time.time()-t0:.1f}s")

        # 2. Preprocess: ffmpeg → 16kHz mono WAV (~5s regardless of file size)
        t0 = time.time()
        preprocessed_path = preprocess_audio(downloaded_path)
        logger.info(f"[{job_id}] Preprocess: {time.time()-t0:.1f}s")

        # Get duration
        with sf.SoundFile(preprocessed_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)
        logger.info(f"[{job_id}] Audio duration: {duration_seconds:.1f}s")

        # 3. Transcribe with KB-Whisper-large (GPU, CTranslate2)
        t0 = time.time()
        words = transcribe(preprocessed_path, _whisper_model)
        logger.info(f"[{job_id}] Transcribe: {time.time()-t0:.1f}s → {len(words)} words")

        # 4. Diarize with NeMo (GPU, PyTorch)
        # NOTE: Sequential after transcribe to avoid CUDA context conflicts between
        # CTranslate2 (faster-whisper) and PyTorch (NeMo) in concurrent threads.
        t0 = time.time()
        diar_segments = diarize(preprocessed_path, num_speakers=num_speakers, job_id=job_id)
        logger.info(f"[{job_id}] Diarize: {time.time()-t0:.1f}s → {len(diar_segments)} segments")

        # 5. Merge
        utterances = merge(words, diar_segments)

        # 6. Build output
        processing_time = round(time.time() - start_time, 2)
        unique_speakers = list({u["speaker"] for u in utterances})
        num_speakers_detected = len(unique_speakers)
        word_speaker_map = _build_word_speaker_map(words, utterances)

        logger.info(
            f"[{job_id}] Done in {processing_time}s: "
            f"{len(utterances)} utterances, {num_speakers_detected} speakers, "
            f"{duration_seconds}s audio"
        )

        return {
            "utterances": [
                {"speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"]}
                for u in utterances
            ],
            "words": word_speaker_map,
            "num_speakers": num_speakers_detected,
            "duration_seconds": duration_seconds,
            "processing_time_seconds": processing_time,
            "audio_url": audio_url,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"[{job_id}] Job failed: {e}")
        # Include traceback in output so RunPod shows it in job details
        return {"error": str(e), "traceback": tb, "audio_url": audio_url}

    finally:
        for path in [downloaded_path, preprocessed_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


def _download_audio(url: str, job_id: str) -> str:
    parsed = urlparse(url)
    original_name = os.path.basename(parsed.path) or "audio"
    safe_name = "".join(c for c in original_name if c.isalnum() or c in "._-")
    if not safe_name:
        safe_name = "audio"

    out_path = f"/tmp/{job_id}_{safe_name}"
    logger.info(f"Downloading audio from {url} -> {out_path}")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f}MB: {out_path}")
    return out_path


def _build_word_speaker_map(words: list[dict], utterances: list[dict]) -> list[dict]:
    result = []
    for utt in utterances:
        for w in utt.get("words", []):
            result.append({"word": w["word"], "start": w["start"], "end": w["end"], "speaker": utt["speaker"]})
    result.sort(key=lambda w: w["start"])
    return result


runpod.serverless.start({"handler": handler})
