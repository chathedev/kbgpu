"""
RunPod Serverless Handler — KB-Whisper Large Transcription
Transcribes a single audio chunk and returns segments + word-level timestamps.
Optimized for FlashBoot: model loaded at module level.
"""

import os
import time
import requests
import runpod
from faster_whisper import WhisperModel

# ── Model loading (module level for FlashBoot snapshot) ──────────────────────
print(f"[{time.strftime('%H:%M:%S')}] Loading KB-Whisper Large model...")
model = WhisperModel(
    "KBLab/kb-whisper-large",
    device="cuda",
    compute_type="float16",
    download_root="/models",
)
print(f"[{time.strftime('%H:%M:%S')}] Worker ready - KB-Whisper loaded")


def handler(job):
    job_input = job["input"]
    audio_url = job_input["audio_url"]
    chunk_index = job_input.get("chunk_index", 0)
    tmp_path = f"/tmp/chunk_{chunk_index}_{int(time.time())}.wav"

    try:
        # Download audio
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_index}: downloading {audio_url}")
        resp = requests.get(audio_url, timeout=120)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_index}: downloaded ({len(resp.content)} bytes)")

        # Transcribe
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_index}: transcribing...")
        segments_gen, info = model.transcribe(
            tmp_path,
            language="sv",
            condition_on_previous_text=False,
            word_timestamps=True,
        )

        segments = []
        words = []
        for seg in segments_gen:
            segments.append({
                "text": seg.text.strip(),
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
            })
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                    })

        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_index}: done — {len(segments)} segments, {len(words)} words")

        return {
            "chunk_index": chunk_index,
            "segments": segments,
            "words": words,
        }

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_index}: ERROR — {e}")
        return {"error": str(e), "chunk_index": chunk_index}

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


runpod.serverless.start({"handler": handler})
