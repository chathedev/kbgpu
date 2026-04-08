import os
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def preprocess_audio(input_path: str) -> str:
    """
    Convert audio to 16kHz mono WAV using ffmpeg.

    Fast path: ffmpeg decodes and resamples any container/codec in ~5s
    regardless of input size. No Demucs — meetings are pure speech, the
    extra 2-3min for vocal separation provides no measurable quality win.

    Returns path to 16kHz mono PCM WAV file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)

    logger.info(f"Preprocessing: {input_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
        # Light loudness normalization — helps Whisper on quiet recordings
        # without distorting quiet speakers relative to loud ones.
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg preprocessing failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"Preprocessed: {out_path} ({size_mb:.1f}MB)")
    return out_path
