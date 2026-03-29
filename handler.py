"""
RunPod Serverless — KB-Whisper Large Transcription
Model loaded at module level for FlashBoot cold-start snapshot.
"""
import os
import time
import runpod
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from faster_whisper import WhisperModel

# ── Load model at module level (FlashBoot snapshots this state) ──────────────
_t = time.time()
print("[BOOT] Loading KB-Whisper Large onto GPU...")
model = WhisperModel(
    "KBLab/kb-whisper-large",
    device="cuda",
    compute_type="float16",
    download_root="/models",
)
print(f"[BOOT] Model loaded in {time.time() - _t:.1f}s — worker ready")


def handler(job):
    t0 = time.time()
    job_input = job["input"]
    audio_url = job_input["audio_url"]
    chunk_index = job_input.get("chunk_index", 0)

    try:
        # Download audio via RunPod's optimized downloader
        t1 = time.time()
        local_paths = download_files_from_urls(job["id"], [audio_url])
        audio_path = local_paths[0]
        print(f"[chunk {chunk_index}] Downloaded in {time.time() - t1:.2f}s")

        # Transcribe
        t2 = time.time()
        segments_gen, info = model.transcribe(
            audio_path,
            language="sv",
            condition_on_previous_text=False,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
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

        print(f"[chunk {chunk_index}] Transcribed in {time.time() - t2:.2f}s — {len(segments)} segs, {len(words)} words, total {time.time() - t0:.2f}s")

        return {
            "chunk_index": chunk_index,
            "segments": segments,
            "words": words,
        }

    except Exception as e:
        print(f"[chunk {chunk_index}] ERROR: {e}")
        return {"error": str(e), "chunk_index": chunk_index}

    finally:
        rp_cleanup.clean(["input_objects"])


runpod.serverless.start({"handler": handler})
