# Accident-Intel Frontend

A React + TypeScript single-page application for AI-assisted traffic accident analysis. It lets investigators upload accident footage, view vehicle tracking overlays, read VLM-generated scene descriptions, retrieve relevant traffic laws, and generate liability reports — all backed by mock data in demo mode.

---

## Prerequisites

- Node.js ≥ 18
- npm ≥ 9

---

## Setup & Development

```bash
cd frontend
npm install
npm run dev
```

The development server runs at **http://localhost:5173** with hot module reloading.

---

## Build

```bash
npm run build
```

Output is written to `dist/`. Serve with any static file server (e.g. `npx serve dist`).

---

## Environment Variables

Copy `.env.example` to `.env.local` and adjust values before starting:

```bash
cp .env.example .env.local
```

| Variable            | Default                    | Description                                         |
|---------------------|----------------------------|-----------------------------------------------------|
| `VITE_API_BASE_URL` | `http://localhost:8000`    | Base URL of the FastAPI backend                     |
| `VITE_MOCK_MODE`    | `true`                     | `true` = local mock data; `false` = real backend    |

---

## Mock / Demo Mode

When `VITE_MOCK_MODE=true` (the default), all data comes from local mock files under `src/mocks/`. No network requests are made. Every page shows a **Demo Data** badge. To connect to the real FastAPI backend:

1. Set `VITE_MOCK_MODE=false` in `.env.local`
2. Set `VITE_API_BASE_URL` to your backend URL
3. Restart the dev server

---

## Routes

| Path              | Page                   | Description                                  |
|-------------------|------------------------|----------------------------------------------|
| `/dashboard`      | DashboardPage          | Overview stats, recent analyses, quick links |
| `/analysis/new`   | NewAnalysisPage        | 6-step analysis workflow (upload → report)   |
| `/analysis/:id`   | AnalysisDetailsPage    | Tabbed report for a specific accident        |
| `/history`        | AnalysisHistoryPage    | Searchable, filterable analysis history      |
| `/damage`         | DamageEstimatorPage    | 5-step vehicle damage + repair cost estimator|
| `/laws`           | TrafficLawsPage        | Searchable NYC traffic law reference library |
| `/settings`       | SettingsPage           | User preferences stored in localStorage      |
| `/*`              | NotFoundPage           | 404 catch-all                                |

---

## API Endpoint Reference

When `VITE_MOCK_MODE=false`, the frontend calls the following FastAPI endpoints:

| Method | Endpoint                      | Description                              |
|--------|-------------------------------|------------------------------------------|
| POST   | `/api/accidents/upload`       | Upload a video + metadata to start analysis |
| GET    | `/api/accidents/:id`          | Fetch a single accident record           |
| GET    | `/api/accidents`              | List all accident records                |
| DELETE | `/api/accidents/:id`          | Delete an accident record                |
| POST   | `/api/damage/upload`          | Upload images to start damage assessment |
| GET    | `/api/laws/search`            | Search traffic laws (query params: q, jurisdiction, topic) |

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
