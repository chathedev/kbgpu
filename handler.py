"""
RunPod Serverless — KB-Whisper Large Transcription
Model loaded at module level for FlashBoot cold-start snapshot.
Model MUST be pre-downloaded at /models/kb-whisper-large during Docker build.
"""
import os
import sys
import time
import base64
import tempfile
import runpod

MODEL_PATH = "/models/kb-whisper-large"

# Verify model exists before importing (fail fast with clear error)
if not os.path.isdir(MODEL_PATH) or len(os.listdir(MODEL_PATH)) < 3:
    print(f"[FATAL] Model not found at {MODEL_PATH}. Must be baked into Docker image at build time.", flush=True)
    sys.exit(1)

from faster_whisper import WhisperModel

# ── Load model at module level (FlashBoot snapshots this state) ──────────────
_t = time.time()
print(f"[BOOT] Loading KB-Whisper Large from {MODEL_PATH}...", flush=True)
model = WhisperModel(
    MODEL_PATH,
    device="cuda",
    compute_type="float16",
)
print(f"[BOOT] Model loaded in {time.time() - _t:.1f}s — worker ready", flush=True)


def handler(job):
    t0 = time.time()
    job_input = job["input"]
    audio_base64 = job_input.get("audio_base64")
    audio_url = job_input.get("audio_url")
    chunk_index = job_input.get("chunk_index", 0)
    language = job_input.get("language", "sv")
    audio_path = None
    _tmp_path = None

    try:
        t1 = time.time()
        if audio_base64:
            # Decode base64 audio to a temp file
            raw = base64.b64decode(audio_base64)
            fd, _tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            audio_path = _tmp_path
            print(f"[chunk {chunk_index}] Decoded base64 ({len(raw)} bytes) in {time.time() - t1:.2f}s", flush=True)
        elif audio_url:
            import requests as _req
            resp = _req.get(audio_url, timeout=30)
            resp.raise_for_status()
            fd, _tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)
            audio_path = _tmp_path
            print(f"[chunk {chunk_index}] Downloaded ({len(resp.content)} bytes) in {time.time() - t1:.2f}s", flush=True)
        else:
            return {"error": "audio_base64 or audio_url required", "chunk_index": chunk_index}

        # Transcribe
        t2 = time.time()
        segments_gen, info = model.transcribe(
            audio_path,
            language=language if language else None,
            condition_on_previous_text=False,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = []
        words = []
        full_text_parts = []
        for seg in segments_gen:
            text = seg.text.strip()
            segments.append({
                "text": text,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            })
            full_text_parts.append(text)
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3) if hasattr(w, "probability") else None,
                    })

        elapsed = time.time() - t2
        print(f"[chunk {chunk_index}] Transcribed in {elapsed:.2f}s — {len(segments)} segs, {len(words)} words, total {time.time() - t0:.2f}s", flush=True)

        return {
            "chunk_index": chunk_index,
            "text": " ".join(full_text_parts),
            "segments": segments,
            "words": words,
            "language": info.language,
            "language_probability": round(info.language_probability, 3) if info.language_probability else None,
            "duration_sec": round(info.duration, 3) if info.duration else None,
            "transcribe_time_sec": round(elapsed, 3),
        }

    except Exception as e:
        print(f"[chunk {chunk_index}] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return {"error": str(e), "chunk_index": chunk_index}

    finally:
        if _tmp_path and os.path.exists(_tmp_path):
            try:
                os.unlink(_tmp_path)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
