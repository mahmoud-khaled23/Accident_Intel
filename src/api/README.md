# Accident-Intel video API

Run from `Accident_Intel/`:

```bash
uv sync --locked
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Use one backend worker: the inference queue and local model caches live in that process.
Jobs run after the upload request returns, with persistent status and intermediate results in SQLite.
On restart, interrupted jobs become failed and can be retried from the UI.

The React UI (port 5173) proxies `/api` to port 8000. Leave `VITE_API_BASE_URL` empty when using
Lightning Studio so the browser uses the Studio URL rather than its own localhost.
For a separate frontend host, set it to the publicly reachable backend origin and configure CORS.

## Model configuration

All `TRACKING_MOCK`, `VLM_MOCK`, `RAG_MOCK`, `REASONING_MOCK`, and `DAMAGE_MOCK` flags default to
`false`. Explicit mock stages label the resulting accident record as demo data. Failures never
substitute sample accident results. The existing project `.env` is read from the project root.

- Tracking: YOLO11x + BoT-SORT, `TRACKING_MODEL_PATH=weights/vehicle_tracking/yolo11x.pt`. Weights download on first use.
  `lap` is required by BoT-SORT and is declared in the dependency files.
- Auto visual model (default): `VIDEO_VLM_API=auto` uses CUDA when available, otherwise the configured hosted vision API.
- Local visual model: `VIDEO_VLM_API=local`, `VLM_MODEL_ID=Qwen/Qwen2-VL-7B-Instruct`.
  Calls `src/video_understanding/analyze_video.py` with the stored upload and real tracking context. CUDA GPU required.
- Hosted visual model: `VIDEO_VLM_API=openai`, `VIDEO_VLM_BASE_URL`, `VIDEO_VLM_MODEL`, `VIDEO_VLM_API_KEY`. Video-specific provider fields override legacy `VLM_*` provider fields.
  The model must accept images. `VLM_N_FRAMES` (default 4) controls sampled frames.
  Hosted frames are grouped into timestamped contact sheets with at most `VLM_MAX_IMAGES`
  images (default 1). The hosted prompt and tracking context are deliberately bounded for providers
  with a 7k input-token limit. `VLM_MAX_TOKENS` defaults to 700; optional `VLM_EXTRA_PARAMS` is JSON.
- Retrieval: `RAG_OUT_DIR=data/traffic_laws/chunks` must contain `parents.json` and `children.jsonl`.
  Hybrid dense/BM25 retrieval with reranking is the default; an empty dense index is built on first use.
  `RAG_MODE=bm25` and `RAG_RERANK=false` explicitly select a lighter configuration without embedding downloads.
- Reasoning: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, and `REASONING_MODEL` (default `gpt-4o-mini`).
  Existing OpenRouter keys automatically use its endpoint. The visual description, bounded tracking
  summary, retrieved passages, and unverified user context are passed to the reasoning model.

`VLM_API` no longer selects the video backend; it may remain in `.env` for repair-cost analysis.
Use `VIDEO_VLM_API` to select local or hosted video analysis without changing that separate pipeline.

`DATA_DIR` and `DB_PATH` configure storage; defaults are `data` and `DATA_DIR/accident_intel.db` under
this project. `MAX_UPLOAD_MB` defaults to 500. The backend checks size, format, metadata, and
frame decoding before accepting MP4, MOV, or WebM uploads. H.264 MP4 or WebM is recommended
for browser playback; unsupported browser codecs show a playback error without hiding analysis.

## Upload workflow

1. `POST /api/accidents/upload` with multipart `file` and optional `location`, `incident_date`,
   `description`, `notes`, and `video_vlm_api` (`auto`, `local`, `openai`). Returns a saved record and video metadata.
2. `POST /api/accidents/{id}/analyze` queues the job. Duplicate starts return 409.
3. `GET /api/accidents/{id}/status` reports `processing`, `tracking`, `vlm_analysis`,
   `law_retrieval`, `reasoning`, `completed`, or `failed`.
4. `GET /api/accidents/{id}` includes available results, demo provenance, and any stage error.
   `GET /api/accidents/{id}/video` supports browser range requests and seeking.
5. Retry a failed job with the same analyze endpoint; optional `?video_vlm_api=openai` changes the backend for that run. Its stale results and error are cleared.

`GET /api/health` reports server health and real/demo configuration for each pipeline stage.
Its `video_vlm` object identifies the requested/resolved backend, model, source module, GPU availability, and configuration errors. `ready` means basic configuration is present, not that a model or provider has been successfully called. Health also reports corpus availability, reasoning credentials presence, and the upload limit.
Interactive endpoint documentation: `/docs`.

## Verification

```bash
uv run pytest -q
uv run ruff check src tests
cd frontend && npm run build
```

Integration tests decode generated videos and exercise the real adapter contracts with model
boundaries replaced by test doubles; they require no GPU, provider requests, or model downloads.
