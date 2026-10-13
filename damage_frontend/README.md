# Damage Frontend

React + Vite + TypeScript frontend for the Car Damage Cost Estimator.

This app is **completely separate** from the existing `frontend/` directory. It lives at `damage_frontend/` and communicates with the FastAPI backend in `damage_backend/`.

## Prerequisites

- Node 18+
- Python 3.13
- Backend dependencies: `pip install -r requirements.txt` from the repo root

## Backend startup

```bash
cd /home/ma7moud-5aled/Projects/Accident_Intel
PYTHONPATH=src/Car_damage_cost uvicorn damage_backend.main:app --host 0.0.0.0 --port 8001 --reload
```

## Frontend startup

```bash
cd damage_frontend
cp .env.example .env
npm install
npm run dev
```

The dev server starts on port 5174 and proxies `/api` requests to `http://localhost:8001`.

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Backend base URL (used for axios baseURL) | `http://localhost:8001` |
| `DAMAGE_WEIGHTS` | Absolute path to `damage_segmentation.pt` | auto-resolved |
| `PARTS_WEIGHTS` | Absolute path to `parts_segmentation.pt` | auto-resolved |
| `VLM_BASE_URL` | VLM API base URL | — |
| `VLM_MODEL` | VLM model name | — |
| `VLM_API` | VLM API type | — |
| `VLM_API_KEY` | VLM API key | — |
| `GEMINI_API_KEY` | Gemini API key for web part-price lookup | — |

## Lightning AI notes

Set `VITE_API_BASE_URL` to the Lightning AI proxy URL for port 8001 before building. The `vite.config.ts` already sets `allowedHosts: 'all'` and `host: '0.0.0.0'` so the Vite dev server is reachable through the Lightning AI proxy.

## Production build

```bash
npm run build
```

Output is placed in `dist/` and can be served by any static host or the FastAPI backend itself.
