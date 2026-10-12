# Accident-Intel FastAPI Backend

## Running the server

From the project root (`Accident_Intel/`):

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at **http://localhost:8000/docs** once the server is running.

---

## Mock flags

All heavy ML components have a mock mode controlled by environment variables (or a `.env` file).
The defaults are safe for local development without GPU or model weights.

| Env var | Default | What it does when `true` |
|---|---|---|
| `VLM_MOCK` | `true` | Returns a realistic pre-written accident description instead of running Qwen2-VL |
| `TRACKING_MOCK` | `true` | Returns synthetic vehicle trajectories instead of running YOLO tracking |
| `REASONING_MOCK` | `false` | Returns a pre-written liability assessment instead of calling OpenAI |
| `RAG_MOCK` | `false` | Returns three hardcoded law sections instead of querying the vector index |
| `DAMAGE_MOCK` | `true` | Returns a synthetic cost report instead of running the damage pipeline |

Set any flag to `false` to enable the real model.

---

## Required environment variables (real mode)

| Var | Used by |
|---|---|
| `OPENAI_API_KEY` | Reasoning service (GPT-4o-mini by default) |
| `GEMINI_API_KEY` | Optional PriceAgent in damage estimation |
| `REASONING_MODEL` | OpenAI model name (default: `gpt-4o-mini`) |
| `GEMINI_MODEL` | Gemini model name (default: `gemini-1.5-flash`) |
| `VLM_MODEL_ID` | Hugging Face model ID (default: `Qwen/Qwen2-VL-7B-Instruct`) |
| `VLM_N_FRAMES` | Frames sampled from video (default: `32`) |
| `VLM_MAX_TOKENS` | Max tokens for VLM response (default: `700`) |
| `RAG_OUT_DIR` | Path to the RAG index directory (default: `out`) |
| `DB_PATH` | SQLite database path (default: `out/accident_intel.db`) |

---

## API endpoints

### Health
| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server health check |

### Accidents
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/accidents` | List accidents (paginated; query: `page`, `limit`, `status`) |
| `POST` | `/api/accidents` | Create a new accident record |
| `GET` | `/api/accidents/{id}` | Get a single accident record |
| `POST` | `/api/accidents/{id}/video` | Upload a video file (multipart) |
| `POST` | `/api/accidents/{id}/analyze` | Trigger the full analysis pipeline (background) |
| `GET` | `/api/accidents/{id}/tracking` | Retrieve vehicle tracking data |
| `GET` | `/api/accidents/{id}/description` | Retrieve VLM accident description |
| `GET` | `/api/accidents/{id}/laws` | Retrieve retrieved traffic law sections |
| `GET` | `/api/accidents/{id}/report` | Retrieve liability reasoning report |

### Damage assessments
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/damage-assessments` | Create a new damage assessment |
| `POST` | `/api/damage-assessments/{id}/images` | Upload one or more images (multipart) |
| `POST` | `/api/damage-assessments/{id}/estimate` | Trigger cost estimation (background) |
| `GET` | `/api/damage-assessments/{id}` | Get assessment result |

### Laws
| Method | Path | Description |
|---|---|---|
| `GET` | `/api/laws` | Search law sections (query: `q`, `jurisdiction`, `topic`, `limit`) |
| `GET` | `/api/laws/{section_id}` | Look up a specific law section |

---

## Swagger UI

Visit **http://localhost:8000/docs** for the interactive Swagger UI.
ReDoc is available at **http://localhost:8000/redoc**.
