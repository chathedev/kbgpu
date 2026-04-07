# KB-Whisper Large + NeMo MSDD — RunPod Serverless Pipeline

Production Swedish meeting transcription pipeline with speaker diarization.

**Stack:**
- **ASR**: [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) via faster-whisper
- **Diarization**: NeMo ClusteringDiarizer + MSDD (Multi-Scale Diarization Decoder)
- **Speaker embeddings**: TitaNet-Large
- **Preprocessing**: Demucs htdemucs vocal separation → 16kHz mono

All models are baked into the Docker image at build time. Cold starts are fast via FlashBoot.

---

## RunPod Setup

1. **Serverless → New Endpoint → Custom Source**
2. Container image: `ghcr.io/chathedev/kbgpu:latest`
3. GPU: **48GB** (A40, A6000, or RTX 6000 Ada — High Supply tier)
4. Max Workers: set based on load (recommend 5–10 for meetings)
5. **Enable FlashBoot** in Endpoint Settings after creation

---

## Input Schema

```json
{
  "input": {
    "audio_url": "https://example.com/meeting.mp3",
    "num_speakers": 4,
    "job_id": "optional-for-logging"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `audio_url` | string | Yes | URL to audio file (mp3, wav, m4a, ogg, flac, etc.) |
| `num_speakers` | int | No | Speaker count hint. Omit for auto-detection (up to 12) |
| `job_id` | string | No | Identifier for logging and temp file naming |

---

## Output Schema

```json
{
  "utterances": [
    {
      "speaker": "Talare 1",
      "start": 0.24,
      "end": 4.88,
      "text": "God morgon allihopa, ska vi börja?"
    }
  ],
  "words": [
    {
      "word": "God",
      "start": 0.24,
      "end": 0.44,
      "speaker": "Talare 1"
    }
  ],
  "num_speakers": 3,
  "duration_seconds": 3612.5,
  "processing_time_seconds": 187.3,
  "audio_url": "https://example.com/meeting.mp3"
}
```

On error:
```json
{
  "error": "description of what went wrong",
  "audio_url": "https://example.com/meeting.mp3"
}
```

---

## Test with curl

```bash
# Submit job
curl -X POST https://api.runpod.io/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_url": "https://example.com/meeting.mp3",
      "num_speakers": 4
    }
  }'

# Poll for result (use job ID from above response)
curl https://api.runpod.io/v2/<ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

---

## Build & Push

```bash
chmod +x build_and_push.sh
./build_and_push.sh
```

Requires Docker with `buildx` and push access to `ghcr.io/chathedev/kbgpu`.

---

## Pipeline Details

1. **Download** — Audio fetched from `audio_url` to `/tmp/`
2. **Preprocess** — Demucs htdemucs extracts vocal stem → resampled to 16kHz mono WAV (falls back to raw audio if separation fails)
3. **Transcribe** — KB-Whisper-large with `language=sv`, word timestamps, VAD filter, beam_size=5
4. **Diarize** — NeMo MarbleNet VAD → TitaNet-Large embeddings → MSDD overlap-aware clustering
5. **Merge** — Words assigned to speakers by maximum time overlap; consecutive same-speaker words grouped into utterances; short utterances merged into neighbors
6. **Cleanup** — All `/tmp/` files deleted after every job

Speaker labels are in Swedish: **Talare 1**, **Talare 2**, etc., numbered by order of first appearance.

Handles files up to 3+ hours. All GPU inference runs in float16.
