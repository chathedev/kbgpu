# KB-Whisper Large — RunPod Serverless Transcription Worker

Transcribes a single audio chunk using [KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large) (Swedish) and returns segment-level + word-level timestamps. Designed to run 19 instances in parallel (one per audio chunk) for fast meeting transcription.

## Deploy on RunPod (Deploy from GitHub)

1. Go to **RunPod Console → Serverless → New Endpoint**
2. Select **Deploy from GitHub**
3. Connect this repo: `https://github.com/chathedev/kbgpu`
4. RunPod will detect `runpod.toml` and build automatically — no Dockerfile needed

### Enable FlashBoot

1. After creating the endpoint, go to **Endpoint Settings**
2. Toggle **FlashBoot** ON
3. The model loads at module level so FlashBoot snapshots the loaded state — subsequent cold starts skip model loading entirely and boot in <200ms

### Environment Variables

These are already set in `runpod.toml` but verify they appear in your endpoint config:

| Variable | Value |
|---|---|
| `HF_HOME` | `/models` |
| `TRANSFORMERS_CACHE` | `/models` |
| `HF_DATASETS_CACHE` | `/models` |

### GPU & Scaling

- **GPU**: RTX 4090 recommended
- **Max Workers**: 20 (handles 19 parallel chunks + 1 buffer)
- **Idle Timeout**: 5s recommended (FlashBoot makes cold starts cheap)

## API

### Input

```json
{
  "input": {
    "audio_url": "https://example.com/chunk_0.wav",
    "chunk_index": 0,
    "total_chunks": 19
  }
}
```

### Output

```json
{
  "chunk_index": 0,
  "segments": [
    { "text": "Hej och välkommen", "start": 0.0, "end": 1.5 }
  ],
  "words": [
    { "word": "Hej", "start": 0.0, "end": 0.3 },
    { "word": "och", "start": 0.35, "end": 0.5 },
    { "word": "välkommen", "start": 0.55, "end": 1.5 }
  ]
}
```

## Example: 19 Parallel Chunk Requests (Node.js)

```javascript
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
const ENDPOINT_ID = "your-endpoint-id";

async function transcribeAllChunks(chunkUrls) {
  const requests = chunkUrls.map((url, i) =>
    fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${RUNPOD_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: {
          audio_url: url,
          chunk_index: i,
          total_chunks: chunkUrls.length,
        },
      }),
    }).then((r) => r.json())
  );

  // Fire all 19 chunks simultaneously
  const results = await Promise.all(requests);

  // Sort by chunk_index and assemble
  return results
    .map((r) => r.output)
    .sort((a, b) => a.chunk_index - b.chunk_index);
}
```

## FlashBoot Details

The model is loaded at **module level** (outside the handler function):

```python
model = WhisperModel("KBLab/kb-whisper-large", device="cuda", ...)
```

When FlashBoot is enabled, RunPod snapshots the worker state after the first boot. On subsequent requests, the snapshot is restored instead of re-running the import chain — the model is already in GPU memory, so there is zero model loading time.
