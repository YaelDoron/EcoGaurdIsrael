# Task 9C: final hardening, migration, end-to-end acceptance and demo activation of `ai_hybrid_v5`

Synthetic-data caveat first: the model was trained and confirmed only on **synthetic** data. `P(fire)` is a model score, not
real-world certainty. Early fires frequently stay SUSPECTED. SUSPECTED expiry is not implemented yet.

## 1. Atomic AI detection results
Before: `create_event` committed the FireEvent (+ evidence refs) and `upsert_ml_assessment` committed the assessment separately. If the
second write failed, `detect()` reported failure but the FireEvent stayed persisted (an event with no assessment).

Now `FireEventRepository` has two composite operations that use ONE session/transaction:
* `create_event_with_ml_assessment(event, refs, assessment_factory)` - event + evidence refs + assessment.
* `update_event_with_ml_assessment(id, event=..., new_evidence=..., assessment=...)` - status promotion / confidence + new evidence +
  latest assessment (any part may be absent; the parts present commit together or not at all).

The AI path of `FireDetectionAgent` uses only these. A failure raises `FireDetectionAIAssessmentPersistenceError` ("... rolled back ...")
and the cycle returns `success=False`. A `NO_EVENT` candidate with no matching event still writes nothing at all (no fake FireEvent);
only candidates tied to a persisted FireEvent get a persisted assessment row - the in-memory candidate assessment and the log are the trace
for negative candidates. Legacy modes keep their existing (separate-write) behaviour unchanged. Tests: `tests/acceptance/test_ai_atomicity_task9c.py`
(SQLite, failure injection incl. a partially promoted SUSPECTED->CONFIRMED event) and a live PostgreSQL rollback test.

## 2. Database migration
`scripts/migrate_add_fire_event_ml_assessment_ai_columns.py`: idempotent (adds only missing columns), one transaction, plain
cross-dialect types, all columns NULLABLE, existing rows untouched. Applied to the configured Neon demo database in Task 9C:
`policy_version VARCHAR`, `policy_status VARCHAR`, `history_available BOOLEAN`, `satellite_pass_count INTEGER`,
`current_satellite_pixel_count INTEGER` - all nullable. The 2 pre-existing (shadow) rows were byte-identical afterwards
(same content hash) and read back with all AI fields NULL.

## 3. Detection performance (found in the live run)
The first live run showed history retrieval costing ~3.4 s per call: `resolve_evidence_refs` did one query PER evidence ref. It now
issues one query per evidence family (`SatelliteHotspotRepository.get_by_ids`, new `NewsRepository.get_by_ids`; test doubles without
a batch method fall back to per-ref lookups; a missing ref still raises `ValueError`). History retrieval: 44 s -> 5 s over the scenario
(6 calls, ~0.9 s each on a remote database). This also speeds up the legacy 5B history path.

## 4. The acceptance scenario
The demo presets last <= 900 simulated seconds, so they can never contain several satellite passes. `src/simulation/ai_acceptance_scenario.py`
adds a dedicated multi-hour scenario that reuses the production per-event pipeline (`execute_simulation_event`) and only replaces the
satellite/news EVIDENCE SOURCE with a fixed observation plan (one growing overnight fire, four NOAA-20 passes 4 h apart, the front
advancing ~1 km per pass, plus reports). No model, threshold or V5-generator change; probabilities are never scripted.
Runners: `tests/acceptance/test_ai_hybrid_v5_final_acceptance_task9c.py` (SQLite; real HGB, real pipeline; test doubles = static
vegetation lookup + tiny pre-seeded road graph) and `scripts/run_ai_hybrid_acceptance.py` (live: production coordinator builders on the
configured database; refuses unless the mode is `ai_hybrid_v5`, the artifact validates and the migration is applied; starts with the
mandatory reset).

Discovery while building it: the approved model treats a hotspot that never moves and never grows as an industrial heat source, and needs
several pixels and a front that drifts between passes before it reaches P >= 0.80. Passes more than 6 h apart also become separate FireEvents
(event matching window, unchanged).

## 5. Readiness check
`python -m scripts.check_ai_hybrid_v5_readiness` (read-only, no secrets, no paths): mode, artifact/metadata/policy validation, migration present.

## 6. Activation
The demo `.env` (git-ignored) sets `FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5` only after every criterion passed. The application's source
default (`shadow`) is unchanged. To roll back: set the line back to `shadow` (or remove it).

## Known limitations
* Synthetic training and confirmatory validation; the probability is not real-world certainty.
* Early fires frequently remain SUSPECTED; there is one CONFIRMED branch (P >= 0.80 and >= 2 current pixels).
* SUSPECTED expiry / dismissal is not implemented.
* The response-plan hint on the event-details page treats a SUSPECTED event as "plan pending" although it never receives a plan (frontend unchanged).
* Detection cycles remain latency-bound on a remote database (~4 s each on Neon); HGB inference is ~0.01-0.2 s.
* The standard demo presets are time-dependent (their hotspots differ per run), so their SUSPECTED/CONFIRMED mix varies.
* The translation LLM (Groq) currently answers 401 in this environment; unrelated to detection.
