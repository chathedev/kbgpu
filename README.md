# KB-Whisper Large — RunPod Serverless Transcription Worker

Transcribes a single audio chunk using [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) (Swedish). Returns segment + word-level timestamps. Runs 19 instances in parallel for fast meeting transcription.

**FlashBoot optimized**: Model loaded at module level, pre-cached in Docker image at build time. Cold starts < 200ms.

## Deploy on RunPod

1. **Serverless → New Endpoint → Deploy from GitHub**
2. Connect repo: `chathedev/kbgpu`, branch: `main`
3. Dockerfile path: `Dockerfile` (auto-detected)
4. GPU: **RTX 4090**
5. Max Workers: **20**
6. **Enable FlashBoot** in Endpoint Settings after creation

### How FlashBoot works here

The model is baked into the Docker image at build time (`builder/fetch_model.py` runs during `docker build`). At runtime, `handler.py` loads it at module level — outside the handler function. FlashBoot snapshots this loaded state. Subsequent cold starts restore the snapshot: zero model download, zero model load.

```python
# Module level — FlashBoot snapshots this
model = WhisperModel("KBLab/kb-whisper-large", device="cuda", compute_type="float16", download_root="/models")

def handler(job):
    # model already in GPU memory
    ...
```

## API

**Input:**
```json
{
  "input": {
    "audio_url": "https://storage.example.com/chunk_0.wav",
    "chunk_index": 0,
    "total_chunks": 19
  }
}
```

**Output:**
```json
{
  "chunk_index": 0,
  "segments": [{ "text": "Hej och välkommen", "start": 0.0, "end": 1.5 }],
  "words": [
    { "word": "Hej", "start": 0.0, "end": 0.3 },
    { "word": "och", "start": 0.35, "end": 0.5 },
    { "word": "välkommen", "start": 0.55, "end": 1.5 }
  ]
}
```

## Example: Fire 19 chunks in parallel (Node.js)

```javascript
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const ENDPOINT_ID = "your-endpoint-id";

async function transcribeAllChunks(chunkUrls) {
  const results = await Promise.all(
    chunkUrls.map((url, i) =>
      fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${RUNPOD_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: { audio_url: url, chunk_index: i, total_chunks: chunkUrls.length },
        }),
      }).then((r) => r.json())
    )
  );

  return results.map((r) => r.output).sort((a, b) => a.chunk_index - b.chunk_index);
}
```

## Repo structure

```
├── Dockerfile              # CUDA 12.3 + cuDNN 9, installs deps, pre-caches model
├── builder/fetch_model.py  # Downloads KB-Whisper at build time
├── handler.py              # RunPod handler, model at module level
├── requirements.txt        # Pinned deps
├── runpod.toml             # RunPod project config
└── test_input.json         # Local testing
```
