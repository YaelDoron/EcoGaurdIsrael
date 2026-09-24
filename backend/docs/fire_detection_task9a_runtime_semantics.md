# Task 9A - demo-run isolation and response eligibility

Scope: two runtime blockers found in Task 8. The AI model is **not** integrated here (no HGB artifact in `FireDetectionAgent`, no `ai_hybrid` mode, locked Task 8 thresholds untouched).

## A. Demo-run isolation (mandatory reset)

Without a reset a new simulation run inherited FireEvents, evidence refs and Fire Detection history from the previous run (a hotspot inside the 6 h matching window attached to the old event, so the "first" observation had `satellite_pass_count = 3` / span 360 min).

### Start paths (all audited)

| Path | Before | Now |
|---|---|---|
| Dashboard "Start / Run again" (`POST /api/v1/simulation/runs`, sends `reset_demo_state: true`) | reset | reset (unchanged) |
| `POST /api/v1/simulation/runs` without the flag | **no reset** | reset (`reset_demo_state` defaults to `true`) |
| `POST ...` with `reset_demo_state: false` | no reset | **422 `SIMULATION_RESET_REQUIRED`** |
| `scripts/run_demo_simulation.py` (manual and automatic) | reset only with `--reset-demo-state` | always resets; `--reset-demo-state` flag removed |
| `SimulationRunManager.start_run` (direct) | reset only if asked | always resets; `reset_demo_state` other than `True` is refused |
| Tests / helpers | construct pieces directly | no supported starter exists besides the two above (guard test) |

`DemoSimulationRunner` is used by exactly two modules (the run manager and the CLI); a guard test fails if a third appears or if either stops going through `src/simulation/demo_run_preparation.prepare_clean_demo_state`.

### Fail-closed
`ENABLE_DEMO_DATA_RESET=false` -> no simulation starts: API returns **403 `SIMULATION_RESET_DISABLED`** (before a run is reserved), the CLI prints a refusal and exits 2, the manager raises `SimulationResetDisabledError`. Nothing is generated, run or deleted. A reset that fails mid-run fails the run (`FAILED` state) - it never falls through to the runner.

### API schema change
`StartSimulationRequest.reset_demo_state`: default `false` -> `true`; `false` is now rejected (422). The frontend already sends `true`; the field is retained for backward compatibility of request bodies.

### Reset scope (unchanged)
`DemoStateResetService._DELETE_ORDER_MODELS` remains the single source of truth (runtime tables incl. FireEvents, satellite/news evidence associations, ML assessments, hotspots, reports, plans, commitments, ...). Static data (stations, resources, road graph, weather stations) is preserved. A guard test fails if a new table is neither in the delete order nor in the declared preserved set. No `simulation_run_id` migration was made.

## B. Active vs response-eligible

Module: `src/models/fire_event_response_eligibility.py`.

| | SUSPECTED | CONFIRMED |
|---|---|---|
| Active for monitoring (`ACTIVE_FOR_MONITORING_STATUSES`) | yes | yes |
| Response eligible (`RESPONSE_ELIGIBLE_STATUSES`) | **no** | yes |

SUSPECTED remains persisted, matchable by later detections, has evidence history, is visible in the active-events API/dashboard and can be promoted to CONFIRMED on the **same event id** with its history intact. It never triggers response work.

| Subsystem | SUSPECTED | CONFIRMED | Mechanism |
|---|---|---|---|
| Severity (simulation-triggered) | no | yes | coordinators use eligible queries / filter detected ids |
| Spread | no | yes | operational refresh early exit |
| Response targets | no | yes | `NOT_RESPONSE_ELIGIBLE` status (in-memory) |
| Routing / road graph | no | yes | reached only through operational refresh / global planning |
| Dijkstra | no | yes | as above |
| Resource allocation / GA | no | yes | as above |
| Global planning | excluded | included | `get_response_eligible_fire_event_ids()` |
| Resource commitment | refused | allowed | activation services require CONFIRMED |
| Dashboard / active-events / history | shown | shown | monitoring set unchanged |

`OperationalRefreshOrchestrator.refresh_fire_event` returns `OperationalRefreshStatus.NOT_RESPONSE_ELIGIBLE` for an active but not eligible event before any severity/spread/target/planning work; the planning-refresh coordinator then skips global planning. No new persisted status or DB constraint was added. The low-level severity/spread agents stay status-agnostic primitives; the gate sits at their callers.

### Performance
A SUSPECTED event costs no road-graph build, route matrix, Dijkstra or GA work, per cycle or repeatedly. Global planning already short-circuits to NO_OP before building the route matrix when nothing changed; spies characterize this.

### Demo runner
The "scramble" (depleting resources near a new fire) applies only to events that become response-eligible during the run.

## C. Compatibility with future expiry / dismissal
SUSPECTED -> DISMISSED needs no change here: DISMISSED is already outside both sets, the lifecycle service dismisses suspected events and releases any (legacy) commitments. Not implemented in 9A.

## D. Secret hygiene
No tests print `Settings`, `.env` or credentials. `Settings` now excludes token/key/secret/`DATABASE_URL` fields from `repr` (values unchanged) so an assertion diff or log can no longer leak them.

## Known limitations
- Tests under `tests/integration` that use the live demo database (e.g. `test_simulation_response_planning_e2e.py`) were not run or updated. A satellite-only detection is SUSPECTED, so an end-to-end scenario that expected a response plan from it now needs a CONFIRMED event (satellite + news) - follow-up when a demo DB is available.
- Pytest run from `backend/` shows 18 known working-directory-dependent architecture-guard failures; they pass from the repository root.
