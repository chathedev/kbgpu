import os
import logging
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch

logger = logging.getLogger(__name__)

TARGET_SR = 16000


def preprocess_audio(input_path: str) -> str:
    """
    Load audio, run Demucs vocal separation, resample to 16kHz mono WAV.
    Falls back to raw audio if Demucs fails.
    Returns path to cleaned WAV file.
    """
    logger.info(f"Preprocessing audio: {input_path}")

    # Load audio with librosa - handles any format ffmpeg can decode
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    # Ensure 2D shape: (channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    duration = audio.shape[-1] / sr
    logger.info(f"Audio loaded: {duration:.1f}s, {audio.shape[0]}ch, {sr}Hz")

    # Try Demucs vocal separation
    try:
        vocals = _run_demucs(audio, sr)
        logger.info("Demucs separation successful")
    except Exception as e:
        logger.warning(f"Demucs failed, falling back to raw audio: {e}")
        vocals = audio

    # Mix to mono
    if vocals.ndim > 1 and vocals.shape[0] > 1:
        mono = vocals.mean(axis=0)
    else:
        mono = vocals.squeeze()

    # Resample to 16kHz
    if sr != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=TARGET_SR)
        logger.info(f"Resampled from {sr}Hz to {TARGET_SR}Hz")

    # Normalize to prevent clipping
    peak = np.abs(mono).max()
    if peak > 0:
        mono = mono / peak * 0.95

    # Save to temp WAV
    fd, out_path = tempfile.mkstemp(suffix=".wav", dir="/tmp")
    os.close(fd)
    sf.write(out_path, mono.astype(np.float32), TARGET_SR, subtype="PCM_16")
    logger.info(f"Preprocessed audio saved: {out_path} ({len(mono)/TARGET_SR:.1f}s)")

    return out_path


def _run_demucs(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Run Demucs htdemucs to extract vocals.
    audio: (channels, samples) float32 numpy array
    Returns: (channels, samples) vocals numpy array
    """
    import demucs.pretrained
    import demucs.apply
    from demucs.audio import convert_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running Demucs on {device}")

    model = demucs.pretrained.get_model("htdemucs")
    model.to(device)
    model.eval()

    # Convert audio to tensor: (1, channels, samples)
    wav = torch.from_numpy(audio).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.unsqueeze(0)  # batch dim

    # Resample to model's expected sr if needed
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)

    with torch.no_grad():
        sources = demucs.apply.apply_model(model, wav, device=device, shifts=1, split=True, overlap=0.25, progress=False)

    # sources shape: (batch, sources, channels, samples)
    # htdemucs sources: drums, bass, other, vocals
    source_names = model.sources
    vocals_idx = source_names.index("vocals")
    vocals = sources[0, vocals_idx]  # (channels, samples)

    # Resample back to original sr if model changed it
    if model.samplerate != sr:
        vocals_np = vocals.cpu().numpy()
        vocals_resampled = np.stack([
            librosa.resample(vocals_np[c], orig_sr=model.samplerate, target_sr=sr)
            for c in range(vocals_np.shape[0])
        ])
        return vocals_resampled

    return vocals.cpu().numpy()
