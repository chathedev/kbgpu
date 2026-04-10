import os
import logging
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

TARGET_SR = 16000

# Audio enhancement filter chain for speech clarity:
#   highpass=f=80       — remove low-frequency rumble/hum
#   lowpass=f=8000      — remove high-frequency noise above speech
#   anlmdn=s=7:p=0.002:r=0.002 — non-local means denoiser
#   acompressor=...     — level compression (quiet parts louder, loud parts softer)
#   loudnorm=...        — EBU R128 loudness normalization
ENHANCE_FILTER = (
    "highpass=f=80,"
    "lowpass=f=8000,"
    "anlmdn=s=7:p=0.002:r=0.002,"
    "acompressor=threshold=-32dB:ratio=8:attack=2:release=80:makeup=10,"
    "loudnorm=I=-16:TP=-1.5:LRA=6"
)

# Simpler fallback filter if the full chain fails (anlmdn can be slow on edge cases)
FALLBACK_FILTER = "highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11"


def preprocess_audio(input_path: str) -> str:
    """
    Convert audio to enhanced 16kHz mono WAV using ffmpeg.

    Pipeline:
      1. Try full enhancement (denoise + compress + loudnorm) — runs at ~10-15x
         realtime on GPU worker CPUs (much faster than a small VPS).
      2. On failure, fall back to lightweight filter (highpass + lowpass + loudnorm).
      3. On double failure, plain conversion (no filters).

    Returns path to 16kHz mono PCM WAV file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)

    logger.info(f"Preprocessing (enhanced): {input_path}")

    # Try full enhancement first
    t0 = time.time()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", "-map", "0:a:0",
        "-af", ENHANCE_FILTER,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=900)
    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        elapsed = time.time() - t0
        logger.info(f"Enhanced preprocessing done: {out_path} ({size_mb:.1f}MB, {elapsed:.1f}s)")
        return out_path

    logger.warning(
        f"Full enhancement failed (exit {result.returncode}), trying fallback filter. "
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )

    # Fallback: lightweight filter
    t0 = time.time()
    cmd_fallback = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn", "-map", "0:a:0",
        "-af", FALLBACK_FILTER,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd_fallback, capture_output=True, timeout=600)
    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        elapsed = time.time() - t0
        logger.info(f"Fallback preprocessing done: {out_path} ({size_mb:.1f}MB, {elapsed:.1f}s)")
        return out_path

    logger.warning(
        f"Fallback filter also failed (exit {result.returncode}), using plain conversion. "
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )

    # Last resort: plain conversion, no filters
    cmd_plain = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd_plain, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg preprocessing failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Plain preprocessed: {out_path} ({size_mb:.1f}MB)")
    return out_path
