"""
kbgpu worker — TRANSCRIPTION ONLY.

Diarization is done in the TIVLY backend via the pyannoteAI Precision-2 API.
This worker's sole job is: download audio → 16kHz mono WAV → KB-Whisper → words.

Output shape:
    {
        "words": [ {word, start, end, probability}, ... ],
        "language": "sv",
        "duration_seconds": float,
        "processing_time_seconds": float,
        "audio_url": str,
    }
"""
import os
import time
import logging
import tempfile
import numpy as np
import requests
import runpod
import soundfile as sf
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model loading — runs ONCE per container lifetime.
# Model hot in VRAM across jobs → sub-second dispatch after FlashBoot wake-up.
# ---------------------------------------------------------------------------
from transcribe import load_whisper_model, transcribe
from preprocess import preprocess_audio

logger.info("=== kbgpu startup: loading KB-Whisper into GPU VRAM ===")
_t0 = time.time()
_whisper_model = load_whisper_model()
logger.info(f"=== Whisper loaded in {time.time()-_t0:.1f}s — running GPU warmup ===")


def _warmup():
    """Force CUDA context init on 1s of near-silence before the first real job."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000).astype(np.float32) * 1e-4)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, noise, 16000)
    try:
        transcribe(path, _whisper_model)
        logger.info("GPU warmup complete")
    except Exception as e:
        logger.warning(f"warmup error (non-fatal): {e}")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


_warmup()
logger.info("=== kbgpu ready — whisper warm in GPU VRAM ===")


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    mode = job_input.get("mode", "transcribe")
    audio_url = job_input.get("audio_url")
    job_id = job_input.get("job_id") or job.get("id") or "unknown"

    logger.info(f"[{job_id}] Job received: mode={mode} audio_url={audio_url}")

    if not audio_url:
        return {"error": "Missing required field: audio_url", "audio_url": None}

    if mode == "separate":
        return _handle_separate(job_input, job_id)

    return _handle_transcribe(job_input, job_id)


def _handle_transcribe(job_input: dict, job_id: str) -> dict:
    audio_url = job_input["audio_url"]
    start_time = time.time()
    downloaded_path = None
    preprocessed_path = None

    try:
        # 1. Download
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        size_mb = os.path.getsize(downloaded_path) / 1e6
        logger.info(f"[{job_id}] Download: {time.time()-t0:.1f}s ({size_mb:.1f}MB)")

        # 2. Preprocess: ffmpeg → 16kHz mono WAV
        t0 = time.time()
        preprocessed_path = preprocess_audio(downloaded_path)
        with sf.SoundFile(preprocessed_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)
        logger.info(f"[{job_id}] Preprocess: {time.time()-t0:.1f}s → {duration_seconds:.1f}s audio")

        # 3. Transcribe (no parallelism needed — only one task)
        t0 = time.time()
        words = transcribe(preprocessed_path, _whisper_model)
        logger.info(f"[{job_id}] Transcribe: {time.time()-t0:.1f}s → {len(words)} words")

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"[{job_id}] Done in {processing_time}s ({len(words)} words, {duration_seconds}s audio)")

        return {
            "words": words,
            "language": "sv",
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


def _handle_separate(job_input: dict, job_id: str) -> dict:
    """
    Vocal isolation mode: download a short audio clip, run Demucs htdemucs
    to separate vocals, return the isolated vocal track as base64 WAV.

    Input:  { mode: "separate", audio_url: str, start?: float, end?: float }
    Output: { vocals_base64: str, duration_seconds: float, processing_time_seconds: float }
    """
    import base64
    import subprocess

    audio_url = job_input["audio_url"]
    clip_start = job_input.get("start")
    clip_end = job_input.get("end")
    start_time = time.time()
    downloaded_path = None
    clip_path = None
    separated_dir = None

    try:
        # 1. Download full audio
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        logger.info(f"[{job_id}] Separate: download {time.time()-t0:.1f}s")

        # 2. Extract clip if start/end specified
        if clip_start is not None and clip_end is not None:
            clip_path = f"/tmp/{job_id}_clip.wav"
            duration = float(clip_end) - float(clip_start)
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(clip_start), "-t", str(duration),
                "-i", downloaded_path, "-ar", "44100", "-ac", "2",
                "-acodec", "pcm_s16le", clip_path,
            ], capture_output=True, timeout=30, check=True)
            input_path = clip_path
        else:
            input_path = downloaded_path

        # 3. Run Demucs vocal separation (GPU-accelerated, ~2-3s for 15s clip)
        t0 = time.time()
        separated_dir = f"/tmp/{job_id}_separated"
        os.makedirs(separated_dir, exist_ok=True)

        subprocess.run([
            "python3", "-m", "demucs",
            "--two-stems", "vocals",  # only separate vocals vs rest
            "-n", "htdemucs",         # fast model
            "-d", "cuda",             # GPU
            "-o", separated_dir,
            input_path,
        ], capture_output=True, timeout=120, check=True)

        # Find the vocals output file
        vocals_path = None
        for root, dirs, files in os.walk(separated_dir):
            for f in files:
                if "vocals" in f.lower():
                    vocals_path = os.path.join(root, f)
                    break
            if vocals_path:
                break

        if not vocals_path or not os.path.exists(vocals_path):
            raise RuntimeError("Demucs did not produce a vocals file")

        logger.info(f"[{job_id}] Demucs separation: {time.time()-t0:.1f}s")

        # 4. Convert vocals to mono 16kHz WAV + apply speech enhancement
        clean_path = f"/tmp/{job_id}_clean.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", vocals_path,
            "-af", "highpass=f=80,lowpass=f=8000,acompressor=threshold=-25dB:ratio=4:attack=5:release=100:makeup=8",
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", clean_path,
        ], capture_output=True, timeout=30, check=True)

        # 5. Read and base64 encode
        with open(clean_path, "rb") as f:
            vocals_data = f.read()

        with sf.SoundFile(clean_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"[{job_id}] Separate done: {processing_time}s, {duration_seconds}s audio, {len(vocals_data)/1024:.0f}KB")

        return {
            "vocals_base64": base64.b64encode(vocals_data).decode("ascii"),
            "duration_seconds": duration_seconds,
            "processing_time_seconds": processing_time,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"[{job_id}] Separate failed: {e}")
        return {"error": str(e), "traceback": tb}

    finally:
        import shutil
        for p in [downloaded_path, clip_path]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
        if separated_dir and os.path.exists(separated_dir):
            try: shutil.rmtree(separated_dir)
            except: pass


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


runpod.serverless.start({"handler": handler})
