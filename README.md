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

The dashboard has one simulation control:

* **Start Simulation** (shown when no run is active, including after a completed or stopped run)
  runs the `presentation_demo` preset: the mandatory demo-state reset, then a fixed, paced
  30-minute timeline with the approved seed **594** (pinned by the backend), so every run replays
  the same story.
* **Stop Simulation** (shown while a run is active) is acknowledged immediately (the run becomes
  `stopping`, the button shows "Stopping..."): no new scheduled event starts, the event already
  executing finishes as one unit (evidence -> detection -> severity -> spread -> targets -> routing
  -> allocation -> plans, so a CONFIRMED fire never loses its downstream outputs), then the run is
  `stopped`. Everything generated stays available for inspection (nothing is deleted; the next
  start resets as usual).

Response plans are produced asynchronously after a fire is CONFIRMED, so the plan pages report the
lifecycle derived from persisted data: "Generating response plan..." (a confirmed fire has no plan
yet), "Updating response plan... Currently covers X of Y confirmed fires" (the shown plan does not
yet include every confirmed fire), and "No response plan available" only when no fire is confirmed.
Only CONFIRMED (response-eligible) fires count; SUSPECTED fires are monitoring-only and are listed as
such, never as pending. The Global Response Plan re-checks every 5 s while a run is live (in every
state, so an open page picks up a newly confirmed fire), shows nothing from the previous run while the
new one is still preparing, and after the run ends keeps checking only while a plan is still being
completed (at most ~1 minute).

The same through the API (the randomized `operations_demo` preset remains available here for
exploratory runs):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/simulation/runs \
  -H "Content-Type: application/json" \
  -d '{"preset":"presentation_demo","reset_demo_state":true}'
curl -X POST http://127.0.0.1:8000/api/v1/simulation/runs/current/stop
```

Only the schedule is designed; the seed drives every generated value (weather, hotspots, news), so the
same seed reproduces the values, AI feature values and statuses - for a daytime or a night-time start
(checked offline with the real model: `python -m scripts.select_presentation_seed`). Absolute
timestamps follow the run start.

Recorded run (T+ = seconds after the run starts; each confirmation's severity/spread/planning work
takes ~25-35 s and delays the following event):

| T+ | What happens |
|---|---|
| 4-19 | Quiet start: fire-danger updates for Carmel, Judean Hills, Galilee |
| ~24 | Judean Hills 1st satellite pass: **SUSPECTED**, AI 0.53 |
| ~32 | Galilee 1st pass: **SUSPECTED**, AI 0.63 |
| ~39 | Judean Hills news report (Hebrew): AI 0.57 |
| ~49 | Judean Hills 2nd pass: **CONFIRMED**, AI 0.95 |
| ~56-75 | Judean Hills severity critical; spread **propagates** (grassland): 48 cells at 30 min, 168 at 60 min; response plan |
| ~84 | Galilee 2nd pass: **CONFIRMED**, AI 0.98 |
| ~91-121 | Galilee severity critical; spread **risk only** (Tree cover -> GENERIC_TREE: 8 cells, 0 propagated); response plan |
| ~130 | Galilee news report |
| ~175+ | Golan Heights appears (**SUSPECTED**, monitoring only); Carmel escalates into a fire at ~7-8 min |
| until 1800 | One update every 26-36 s: weather, satellite passes and news for every fire, fire-danger updates |

Spread wording: under the current homogeneous-fuel PROPAGATOR model, tree-dominant land cover
(GENERIC_TREE) stays below the 0.45 propagation threshold at the simulated fuel moisture even in
strong wind, so forest fires show "risk only" cells, while grassland reaches the threshold with
~13 km/h of aligned wind.

## Known limitations

1. Fire Detection ML (HGB V5) was trained and validated on **synthetic** data; AI likelihood is a model estimate, not real-world certainty.
2. Fire spread uses a **homogeneous fuel grid**: one fuel class for the whole prediction grid.
3. That fuel class comes from the **dominant** Copernicus land-cover class only (no fraction-weighted fuel).
4. Spread assumes **flat terrain** (no slope).
5. Copernicus reports generic tree cover only, so tree-dominant areas use the EcoGuard-derived **GENERIC_TREE** fuel (no tree-species identification).
6. News translation uses a Groq LLM that may be **unavailable** (e.g. HTTP 401); readable Hebrew is then shown as-is.
7. Road networks come from a Neon cache with live OpenStreetMap/Overpass fallback; an Overpass outage slows planning for uncached areas.
8. Simulation events are processed one at a time: after a confirmation, downstream severity/spread/planning work delays the following scheduled updates by up to ~40 s, and a delayed update can arrive shortly after the previous one.
