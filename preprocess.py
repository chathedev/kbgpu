import os
import logging
import subprocess
import tempfile

logger = logging.getLogger(__name__)

TARGET_SR = 16000

# Demucs is only useful for music/noise separation.
# Meeting recordings are pure speech — skip by default for 2-3 min speedup.
ENABLE_DEMUCS = os.environ.get("ENABLE_DEMUCS", "0").strip() == "1"


def preprocess_audio(input_path: str) -> str:
    """
    Convert audio to 16kHz mono WAV using ffmpeg.

    Replaces the old librosa.load approach which took 60-90s on CPU for
    large m4a files. ffmpeg decodes and resamples in ~5s regardless of
    input format or file size.

    Returns path to 16kHz mono PCM WAV file.
    """
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)

    logger.info(f"Preprocessing: {input_path} (demucs={ENABLE_DEMUCS})")

    # Fast path: ffmpeg handles any container/codec directly
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        "-acodec", "pcm_s16le",
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

    # Optional Demucs vocal separation (default OFF — adds 2-3 min for no benefit on meetings)
    if ENABLE_DEMUCS:
        try:
            out_path = _run_demucs(out_path)
            logger.info("Demucs separation successful")
        except Exception as e:
            logger.warning(f"Demucs failed, using raw audio: {e}")

    return out_path


def _run_demucs(wav_path: str) -> str:
    """
    Run Demucs htdemucs on an already-converted 16kHz mono WAV.
    Returns path to vocals-only WAV (same directory).
    """
    import numpy as np
    import soundfile as sf
    import torch
    import demucs.pretrained
    import demucs.apply
    from demucs.audio import convert_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running Demucs on {device}")

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]  # (1, T)
    wav = torch.from_numpy(audio).float().unsqueeze(0)  # (1, C, T)

    model = demucs.pretrained.get_model("htdemucs")
    model.to(device)
    model.eval()

    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    with torch.no_grad():
        sources = demucs.apply.apply_model(
            model, wav, device=device, shifts=1, split=True, overlap=0.25, progress=False
        )

    vocals_idx = model.sources.index("vocals")
    vocals = sources[0, vocals_idx].mean(dim=0).cpu().numpy()  # mono

    # Resample back to 16kHz if needed
    if model.samplerate != TARGET_SR:
        import librosa
        vocals = librosa.resample(vocals, orig_sr=model.samplerate, target_sr=TARGET_SR)

    # Normalize
    peak = abs(vocals).max()
    if peak > 0:
        vocals = vocals / peak * 0.95

    # Overwrite the existing wav
    sf.write(wav_path, vocals.astype("float32"), TARGET_SR, subtype="PCM_16")
    return wav_path
