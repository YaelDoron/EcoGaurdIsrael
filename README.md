# EcoGaurdIsrael

## Backend API

The backend exposes a FastAPI HTTP API (`backend/src/api/`).

**Install dependencies** (from `backend/`):

```bash
pip install -r requirements.txt
```

**Environment variable** — origins allowed to call the API (CORS), used by the local React/Vite dev server:

```text
FRONTEND_ORIGINS=http://localhost:5173
```

See `.env.example` for the full list of environment variables.

**Run the API locally** (from `backend/`):

```bash
python -m uvicorn src.api.app:app --reload
```

**Health check:**

```text
GET http://localhost:8000/health
```

```json
{"status": "ok"}
```

**Active wildfire events** (Epic 6, US 6.1) — currently active (suspected/confirmed) FireEvents, each with its latest persisted severity assessment if one exists:

```text
GET http://localhost:8000/api/v1/fire-events/active
```

```json
{
  "as_of": "2026-09-17T14:00:00Z",
  "items": [
    {
      "fire_event_id": 12,
      "status": "confirmed",
      "latitude": 32.731,
      "longitude": 35.046,
      "detection_confidence": 0.91,
      "detected_at": "2026-09-17T13:20:00Z",
      "updated_at": "2026-09-17T13:28:00Z",
      "severity": {
        "assessment_id": 44,
        "status": "valid",
        "score": 81.4,
        "level": "critical",
        "assessed_at": "2026-09-17T13:27:00Z"
      }
    }
  ]
}
```

## Frontend Development

The frontend is a React + TypeScript + Vite application (`frontend/`) — currently an application shell/foundation (Epic 6, US 6.1 Task 4): header, navigation, routing, and a shared API client. It does not yet render the Active Wildfires dashboard itself.

**Prerequisites:** Node.js 20.19+ (or 22.12+) and npm.

**Install dependencies:**

```bash
cd frontend
npm install
```

**Environment variable** — backend API base URL (falls back to `http://localhost:8000` if omitted, see `frontend/src/api/config.ts`):

```text
VITE_API_BASE_URL=http://localhost:8000
```

See `frontend/.env.example`.

**Run the dev server** (from `frontend/`):

```bash
npm run dev
```

Opens at `http://localhost:5173`.

**Build and test:**

```bash
npm run build
npm test
```