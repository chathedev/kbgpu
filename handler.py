import os
import time
import logging
import tempfile
import numpy as np
import requests
import runpod
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model loading — runs ONCE per container lifetime.
# This is the key to fast response: models are hot in GPU VRAM between jobs.
# ---------------------------------------------------------------------------
from transcribe import load_whisper_model
from preprocess import preprocess_audio
from transcribe import transcribe
from diarize import load_diarization_models, diarize
from merge import merge

logger.info("=== kbgpu startup: loading models into GPU VRAM ===")
_t0 = time.time()

_whisper_model = load_whisper_model()           # CTranslate2 float16 on CUDA
_diar_models = load_diarization_models()        # NeMo VAD + TitaNet on CUDA

logger.info(f"=== Models loaded in {time.time()-_t0:.1f}s — running GPU warmup ===")

# ---------------------------------------------------------------------------
# GPU warmup: run both pipelines on 0.5s of silence.
# This forces CUDA context initialization for BOTH CTranslate2 AND PyTorch
# BEFORE any real job arrives, so parallel thread execution is safe.
# Without warmup, concurrent CUDA init in threads → deadlock / CPU spin.
# ---------------------------------------------------------------------------
def _warmup():
    silence = np.zeros(8000, dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, silence, 16000)
    try:
        transcribe(path, _whisper_model)
        diarize(path, diar_models=_diar_models, num_speakers=1, job_id="warmup")
        logger.info("GPU warmup complete — CTranslate2 + NeMo CUDA contexts initialized")
    except Exception as e:
        logger.warning(f"GPU warmup error (non-fatal): {e}")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

_warmup()
logger.info("=== kbgpu ready — all models warm in GPU VRAM ===")


def handler(job: dict) -> dict:
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
        # 1. Download
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        logger.info(f"[{job_id}] Download: {time.time()-t0:.1f}s ({os.path.getsize(downloaded_path)/1e6:.1f}MB)")

        # 2. Preprocess: ffmpeg → 16kHz mono WAV (~5s for any file size)
        t0 = time.time()
        preprocessed_path = preprocess_audio(downloaded_path)
        with sf.SoundFile(preprocessed_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)
        logger.info(f"[{job_id}] Preprocess: {time.time()-t0:.1f}s → {duration_seconds:.1f}s audio")

        # 3+4. Transcribe + Diarize IN PARALLEL on the A40/A6000.
        #
        # This is safe because:
        #  a) GPU warmup above pre-initialized CUDA contexts for both engines
        #  b) NeMo uses num_workers=0 (no process forking after CUDA init)
        #  c) CTranslate2 and PyTorch share the CUDA device, CUDA scheduler
        #     interleaves their kernels — both see high GPU utilization
        #
        # Expected wall time: max(whisper_time, nemo_time) instead of sum.
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            transcribe_fut = executor.submit(
                transcribe, preprocessed_path, _whisper_model
            )
            diarize_fut = executor.submit(
                diarize, preprocessed_path,
                diar_models=_diar_models,
                num_speakers=num_speakers,
                job_id=job_id,
            )
            words = transcribe_fut.result()
            diar_segments = diarize_fut.result()

        logger.info(
            f"[{job_id}] Transcribe+Diarize parallel: {time.time()-t0:.1f}s "
            f"→ {len(words)} words, {len(diar_segments)} segments"
        )

        # 5. Merge
        utterances = merge(words, diar_segments)

        # 6. Output
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
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
    return out_path


def _build_word_speaker_map(words: list[dict], utterances: list[dict]) -> list[dict]:
    result = []
    for utt in utterances:
        for w in utt.get("words", []):
            result.append({"word": w["word"], "start": w["start"], "end": w["end"], "speaker": utt["speaker"]})
    result.sort(key=lambda w: w["start"])
    return result


runpod.serverless.start({"handler": handler})
