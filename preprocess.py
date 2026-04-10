import os
import logging
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

TARGET_SR = 16000

# Fast audio enhancement — ALL filters are single-pass, instant (no 2-pass analysis).
# NO anlmdn (O(n²) denoiser), NO loudnorm (2-pass, ~1x realtime on long files).
# Compressor + makeup gain handles level normalization in a single pass.
ENHANCE_FILTER = (
    "highpass=f=80,"
    "lowpass=f=8000,"
    "acompressor=threshold=-25dB:ratio=4:attack=5:release=100:makeup=8"
)


def preprocess_audio(input_path: str) -> str:
    """
    Convert audio to enhanced 16kHz mono WAV using ffmpeg.

    Uses fast filters only (no denoiser) — total preprocessing takes ~5-15s
    even for 2+ hour meetings. Falls back to plain conversion on failure.

    Returns path to 16kHz mono PCM WAV file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)

    logger.info(f"Preprocessing (enhanced): {input_path}")

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
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode == 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        elapsed = time.time() - t0
        logger.info(f"Enhanced preprocessing done: {out_path} ({size_mb:.1f}MB, {elapsed:.1f}s)")
        return out_path

    logger.warning(
        f"Enhancement failed (exit {result.returncode}), using plain conversion. "
        f"stderr: {result.stderr.decode(errors='replace')[:500]}"
    )

    # Fallback: plain conversion, no filters
    cmd_plain = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
        out_path,
    ]
    result = subprocess.run(cmd_plain, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg preprocessing failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Plain preprocessed: {out_path} ({size_mb:.1f}MB)")
    return out_path
