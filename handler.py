"""
kbgpu worker — Swedish ASR + speaker diarization on a single GPU.

Pipelines on RunPod serverless:
    mode="transcribe": download audio → 16kHz WAV → KB-Whisper → words
    mode="diarize":    download audio → DiariZen speaker segmentation
    mode="separate":   extract a denoised short clip for speaker samples

Transcription and diarization are loaded into VRAM at container boot so
FlashBoot wake-ups start sub-second. Backend calls them in parallel over
two RunPod requests.
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
# ---------------------------------------------------------------------------
from transcribe import load_whisper_model, transcribe
from preprocess import preprocess_audio

logger.info("=== kbgpu startup: loading KB-Whisper into GPU VRAM ===")
_t0 = time.time()
_whisper_model = load_whisper_model()
logger.info(f"=== Whisper loaded in {time.time()-_t0:.1f}s ===")

# Diarization pipeline loaded lazily-but-eagerly. If it fails we still want
# the container to serve transcription — diarization mode will return an
# error but the worker remains useful.
_diar_pipeline = None
_diar_load_error = None
try:
    from diarize import load_diarization_pipeline, diarize
    _t1 = time.time()
    _diar_pipeline = load_diarization_pipeline()
    logger.info(f"=== Diarization pipeline loaded in {time.time()-_t1:.1f}s ===")
except Exception as diar_err:
    _diar_load_error = str(diar_err)
    logger.exception(f"Diarization pipeline failed to load: {diar_err}")


def _warmup():
    """Force CUDA context init on 1s of near-silence before the first real job."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000).astype(np.float32) * 1e-4)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, noise, 16000)
    try:
        transcribe(path, _whisper_model)
        logger.info("Whisper GPU warmup complete")
    except Exception as e:
        logger.warning(f"warmup error (non-fatal): {e}")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


_warmup()
logger.info("=== kbgpu ready ===")


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

    if mode == "diarize":
        return _handle_diarize(job_input, job_id)

    return _handle_transcribe(job_input, job_id)


def _handle_transcribe(job_input: dict, job_id: str) -> dict:
    audio_url = job_input["audio_url"]
    start_time = time.time()
    downloaded_path = None
    preprocessed_path = None

    try:
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        size_mb = os.path.getsize(downloaded_path) / 1e6
        logger.info(f"[{job_id}] Download: {time.time()-t0:.1f}s ({size_mb:.1f}MB)")

        t0 = time.time()
        preprocessed_path = preprocess_audio(downloaded_path)
        with sf.SoundFile(preprocessed_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)
        logger.info(f"[{job_id}] Preprocess: {time.time()-t0:.1f}s → {duration_seconds:.1f}s audio")

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


def _handle_diarize(job_input: dict, job_id: str) -> dict:
    """
    Speaker diarization on a full audio file.

    Input:
        { mode: "diarize", audio_url: str,
          num_speakers?: int, min_speakers?: int, max_speakers?: int }

    Output:
        { segments: [{speaker, start, end}, ...],
          speaker_count: int,
          duration_seconds: float,
          processing_time_seconds: float }
    """
    if _diar_pipeline is None:
        return {
            "error": f"Diarization pipeline unavailable: {_diar_load_error}",
            "audio_url": job_input.get("audio_url"),
        }

    audio_url = job_input["audio_url"]
    start_time = time.time()
    downloaded_path = None
    preprocessed_path = None

    try:
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        size_mb = os.path.getsize(downloaded_path) / 1e6
        logger.info(f"[{job_id}] diar download: {time.time()-t0:.1f}s ({size_mb:.1f}MB)")

        t0 = time.time()
        preprocessed_path = preprocess_audio(downloaded_path)
        with sf.SoundFile(preprocessed_path) as f:
            duration_seconds = round(len(f) / f.samplerate, 2)
        logger.info(f"[{job_id}] diar preprocess: {time.time()-t0:.1f}s → {duration_seconds:.1f}s audio")

        t0 = time.time()
        segments = diarize(
            preprocessed_path,
            _diar_pipeline,
            num_speakers=job_input.get("num_speakers"),
            min_speakers=job_input.get("min_speakers"),
            max_speakers=job_input.get("max_speakers"),
        )
        logger.info(f"[{job_id}] diar inference: {time.time()-t0:.1f}s → {len(segments)} segments")

        speaker_count = len({s["speaker"] for s in segments})
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"[{job_id}] diar done in {processing_time}s ({speaker_count} speakers, {duration_seconds}s audio)")

        return {
            "segments": segments,
            "speaker_count": speaker_count,
            "duration_seconds": duration_seconds,
            "processing_time_seconds": processing_time,
            "audio_url": audio_url,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"[{job_id}] Diarize failed: {e}")
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
    Speaker sample isolation: extract a clip with aggressive noise gate +
    bandpass filtering to isolate the dominant voice.

    Input:  { mode: "separate", audio_url: str, start?: float, end?: float }
    Output: { vocals_base64, duration_seconds, processing_time_seconds }
    """
    import base64
    import subprocess

    audio_url = job_input["audio_url"]
    clip_start = job_input.get("start")
    clip_end = job_input.get("end")
    start_time = time.time()
    downloaded_path = None

    try:
        t0 = time.time()
        downloaded_path = _download_audio(audio_url, job_id)
        logger.info(f"[{job_id}] Separate: download {time.time()-t0:.1f}s")

        clean_path = f"/tmp/{job_id}_clean.wav"
        args = ["ffmpeg", "-y"]
        if clip_start is not None:
            args += ["-ss", str(clip_start)]
        if clip_end is not None and clip_start is not None:
            args += ["-t", str(min(float(clip_end) - float(clip_start), 10))]
        args += [
            "-i", downloaded_path,
            "-af", (
                "highpass=f=120,lowpass=f=6000,"
                "agate=threshold=-30dB:ratio=4:attack=5:release=50,"
                "acompressor=threshold=-25dB:ratio=4:attack=5:release=100:makeup=8,"
                "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB"
            ),
            "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", clean_path,
        ]
        subprocess.run(args, capture_output=True, timeout=30, check=True)

        with open(clean_path, "rb") as f:
            vocals_data = f.read()

        with sf.SoundFile(clean_path) as fh:
            duration_seconds = round(len(fh) / fh.samplerate, 2)

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"[{job_id}] Separate done: {processing_time}s, {duration_seconds}s audio")

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
        for p in [downloaded_path, f"/tmp/{job_id}_clean.wav"]:
            if p and os.path.exists(p):
                try: os.remove(p)
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
