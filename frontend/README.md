# Accident-Intel Frontend

A React + TypeScript single-page application for AI-assisted traffic accident analysis. It lets investigators upload accident footage, view vehicle tracking overlays, read VLM-generated scene descriptions, retrieve relevant traffic laws, and generate liability reports — connected to the backend model pipeline by default.

---

## Prerequisites

- Node.js ≥ 18
- npm ≥ 9

---

## Setup & Development

```bash
cd frontend
npm ci
npm run dev
```

The development server runs at **http://localhost:5173** with hot module reloading.

---

## Build

```bash
npm run build
```

Output is written to `dist/`. `npm run preview` proxies `/api` to the backend like development. For deployment, route `/api` to the backend or set `VITE_API_BASE_URL` before building.

---

## Environment Variables

Copy `.env.example` to `.env.local` and adjust values before starting:

```bash
cp .env.example .env.local
```

| Variable            | Default                    | Description                                         |
|---------------------|----------------------------|-----------------------------------------------------|
| `VITE_API_BASE_URL` | `empty (same origin)`     | Base URL of the FastAPI backend                     |
| `VITE_USE_MOCK`     | `false`                    | `true` = local mock data; `false` = real backend    |

---

## Mock / Demo Mode

When `VITE_USE_MOCK=true` (explicit demo mode), all data comes from local mock files under `src/mocks/`. Reports are marked as demo data and video upload is blocked. Pipeline settings still inspects server health. To connect to the real FastAPI backend:

1. Set `VITE_USE_MOCK=false` in `.env.local`
2. Leave `VITE_API_BASE_URL` empty for the Vite proxy, including Lightning Studio. Set a public backend URL only for a separate deployment.
3. Restart the dev server

---

## Routes

| Path              | Page                   | Description                                  |
|-------------------|------------------------|----------------------------------------------|
| `/dashboard`      | DashboardPage          | Overview stats, recent analyses, quick links |
| `/analysis/new`   | NewAnalysisPage        | Real upload → tracking → vision → RAG → reasoning   |
| `/analysis/:id`   | AnalysisDetailsPage    | Tabbed report for a specific accident        |
| `/history`        | AnalysisHistoryPage    | Searchable, filterable analysis history      |
| `/damage`         | DamageEstimatorPage    | 5-step vehicle damage + repair cost estimator|
| `/laws`           | TrafficLawsPage        | Searchable NYC traffic law reference library |
| `/settings`       | SettingsPage           | Live pipeline configuration + VLM preference      |
| `/*`              | NotFoundPage           | 404 catch-all                                |

---

## API Endpoint Reference

When `VITE_USE_MOCK=false`, the frontend calls the following FastAPI endpoints:

| Method | Endpoint                      | Description                              |
|--------|-------------------------------|------------------------------------------|
| POST   | `/api/accidents/upload`       | Upload a video + metadata to start analysis |
| GET    | `/api/accidents/:id`          | Fetch a single accident record           |
| GET    | `/api/accidents`              | List all accident records                |
| DELETE | `/api/accidents/:id`          | Delete an accident record                |
| POST   | `/api/damage-assessments/`    | Create a damage assessment |
| POST   | `/api/damage-assessments/:id/images` | Upload assessment images |
| POST   | `/api/damage-assessments/:id/estimate` | Start repair-cost estimation |
| GET    | `/api/laws/`                  | Search traffic laws (query params: q, jurisdiction, topic) |

---

## Project Structure

```
src/
├── app/           # Router and App entry
├── components/
│   ├── analysis/  # VLMDescription, LegalCitationCard, LiabilityReport, EvidenceCard
│   ├── common/    # PageHeader, DemoBadge, StatusBadge, EmptyState, LoadingState, ErrorState, ConfirmDialog
│   ├── damage/    # DamageAssessment, RepairCostBreakdown
│   ├── layout/    # AppShell, Sidebar, Topbar
│   ├── tracking/  # VehicleList, TrackingOverlay, Timeline
│   ├── ui/        # shadcn-style Radix UI primitives
│   └── upload/    # VideoUploader, VideoPreview, ImageUploader
├── hooks/         # Custom React hooks (useDamageAssessment, etc.)
├── mocks/         # Static demo data (accidentMocks, damageMocks, lawMocks)
├── pages/         # Top-level page components
├── services/      # API + mock service layer
├── styles/        # Global CSS
├── types/         # TypeScript type definitions
└── utils/         # pdfExport and other utilities
```

---

## Tech Stack

| Library              | Version  | Purpose                                   |
|----------------------|----------|-------------------------------------------|
| React                | 18       | UI rendering                              |
| React Router         | v6       | Client-side routing                       |
| TypeScript           | 5        | Type safety                               |
| Vite                 | 5        | Build tool & dev server                   |
| Tailwind CSS         | 3        | Utility-first styling                     |
| Radix UI             | various  | Accessible component primitives           |
| Recharts             | 3        | Data visualization (bar chart, pie chart) |
| jsPDF + html2canvas  | latest   | PDF export                                |
| Lucide React         | latest   | Icon set                                  |
| Axios                | latest   | HTTP client (real mode)                   |

## Real video analysis

Upload a video and optional incident details on `/analysis/new`. After the upload starts the
model pipeline, the app opens `/analysis/:id`. This saved page follows live status, shows partial
results, and supports retries. History restores the same video and results after refresh.
The Tracking tab plays the uploaded video and draws model detections at its actual timestamps.

The model selector offers Auto (CUDA or API), Local GPU, and API. The chosen mode is persisted with each upload and may be changed on retry. Keys remain on the server; the settings page exposes only configuration status.
