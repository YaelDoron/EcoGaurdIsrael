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
## Demo simulation and presentation mode

The dashboard's **Start Simulation / Run Again** button runs the `operations_demo` preset with a new
backend-chosen seed every time (mandatory demo-state reset first), so normal runs vary.

For a rehearsable presentation, start the same preset with an explicit seed through the API:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulation/runs \
  -H "Content-Type: application/json" \
  -d '{"preset":"operations_demo","seed":893,"reset_demo_state":true}'
```

Same preset + same seed reproduces the incidents, schedule, weather values, hotspot properties,
news text and therefore the AI feature values and statuses; only absolute timestamps follow the run
start. Two conditions: rehearse and present in the same satellite day/night window (the day/night
flag uses the event's UTC hour, day = 06:00-18:00 UTC, and is an AI feature), and with the same
News-LLM (Groq) availability.

Recorded seed-893 run (T+ = seconds after the run starts; a few seconds of processing lag):

| T+ | Galilee (Har Kamon) | Judean Hills | Golan Heights |
|---|---|---|---|
| ~55 | | | 1st satellite pass: AI 0.32, below SUSPECTED threshold, no event |
| ~95 | 1st pass: **SUSPECTED**, AI 0.58 | | |
| ~115 | | 1st pass: **SUSPECTED**, AI 0.58 | |
| ~175 | | News (Hebrew): AI 0.61 | |
| ~215 | | 2nd pass: **CONFIRMED**, AI 0.97 | |
| ~225 | | Severity critical; spread **propagates** (grassland): 48 cells at 30 min, 168 at 60 min | |
| ~240 | | Response targets + response plan | |
| ~250 | News: AI 0.61 | | |
| ~310 | | | 2nd pass: **SUSPECTED**, AI 0.77 (stays monitoring-only) |
| ~355 | 2nd pass: **CONFIRMED**, AI 0.81 | | |
| ~370 | Severity critical; spread **risk only** (Tree cover → GENERIC_TREE): 8 cells, 0 propagated | | |
| ~395 | Response plan; Global Response Plan covers Galilee + Judean Hills only | | |

## Known limitations

1. Fire Detection ML (HGB V5) was trained and validated on **synthetic** data; AI likelihood is a model estimate, not real-world certainty.
2. Fire spread uses a **homogeneous fuel grid**: one fuel class for the whole prediction grid.
3. That fuel class comes from the **dominant** Copernicus land-cover class only (no fraction-weighted fuel).
4. Spread assumes **flat terrain** (no slope).
5. Copernicus reports generic tree cover only, so tree-dominant areas use the EcoGuard-derived **GENERIC_TREE** fuel (no tree-species identification).
6. News translation uses a Groq LLM that may be **unavailable** (e.g. HTTP 401); readable Hebrew is then shown as-is.
7. Road networks come from a Neon cache with live OpenStreetMap/Overpass fallback; an Overpass outage slows planning for uncached areas.
