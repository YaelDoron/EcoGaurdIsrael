# Fire Detection: AI Hybrid V5 runtime mode (Task 9B)

`FIRE_DETECTION_DECISION_MODE=ai_hybrid_v5` makes the approved HistGradientBoosting V5 model plus the locked AI Hybrid Policy
v5.0 the **only** source of FireEvent status. It is opt-in: the project default (`shadow`) and `.env` are unchanged, and
`rule_only` / `shadow` / `hybrid` behave exactly as before and never load the V5 artifact.

All model training and confirmatory evaluation data is **synthetic**. A model probability is a model score within that
distribution, not a calibrated real-world probability of wildfire, and must never be presented as operational certainty.

## Runtime path

```
FireDetectionCandidate  (<= 60 min, 5 km; unchanged)
  -> matching ACTIVE FireEvent?            find_matching_active_event, 5 km / 6 h (unchanged)
  -> FireDetectionHistoryService           the event's attached evidence, 24 h window (or None for a new candidate)
  -> FireDetectionFeatureExtractorV5       the Task 6 extractor: 25 ordered features (candidate + de-duplicated history)
  -> FireDetectionModelV5Runtime           HGB V5 -> P(fire)         (validated, cached, immutable)
  -> fire_detection_ai_hybrid_policy_v5    P < 0.40 NO_EVENT | P >= 0.40 SUSPECTED | P >= 0.80 AND current pixels >= 2 CONFIRMED
  -> FireDetectionAIAssessment             probability, policy status, model / policy version, pixel + pass counts
  -> FireDetectionAgent persistence        create / update the FireEvent, upsert the latest ML assessment row
```

Modules: `src/ml/fire_detection/fire_detection_model_runtime_v5.py` (artifact loader), `fire_detection_ai_hybrid_runtime_v5.py`
(classifier), `src/models/fire_detection_ai_assessment.py` (result), and the `_detect_candidate_ai_v5` branch of
`FireDetectionAgent`. Neither features nor policy are reimplemented; thresholds are imported from
`fire_detection_policy_config_v5`, never copied. The V5 modules import no calculator, repository, agent, service or Fire Danger
code (guard test). **Fire Danger is not used for V5 inference.**

## Artifact validation (strict, no lenient mode)
`fire_detection_hgb_v5.joblib` + `fire_detection_hgb_v5_metadata.json` must satisfy: `model_type == HistGradientBoostingClassifier`,
model version `5.0`, feature schema `v5`; metadata `feature_names` equal `FIRE_DETECTION_FEATURE_NAMES_V5` exactly (names and
order); Task 7 gate recorded `FAILED`, Task 8 policy gate `PASSED`, policy version / thresholds / multi-pixel definition equal
the locked config; the deserialized object is a pipeline ending in an HGB classifier, fitted on 25 features, classes `[0, 1]`.
Any mismatch raises an explicit error. Paths: `FIRE_DETECTION_AI_V5_MODEL_PATH` / `FIRE_DETECTION_AI_V5_METADATA_PATH`
(defaults point at `backend/models/fire_detection/`); only read in this mode.

## Caching
The artifact is deserialized once per process (module-level cache keyed by path), lazily on first use, and shared by every
agent - including the agent each simulation run builds. The cache holds only the immutable model. Incident / evidence /
history state is never cached and is reset per demo run by the Task 9A mandatory reset. Failed loads are not cached.

## Decisions
* **New candidate:** `NO_EVENT` creates nothing (the in-memory candidate assessment and the log are the only trace - no fake
  FireEvent); `SUSPECTED` / `CONFIRMED` create an event with that status. `detection_confidence` = the probability.
* **Existing event:** `CONFIRMED` never downgrades; `SUSPECTED` is promoted only by a `CONFIRMED` result and is **never
  dismissed** by a `NO_EVENT` result (expiry / dismissal is a separate future lifecycle task). The candidate's evidence is
  attached whatever the verdict (a weak pass is still a pass in the history).
* **Confidence semantics:** `FireEvent.detection_confidence` is the monotonic **peak** probability. The **latest** probability
  lives in the event's ML-assessment row (`ml_probability`) together with `policy_status`, `policy_version`, `history_available`,
  `satellite_pass_count`, `current_satellite_pixel_count`. Example: T+0 0.61, T+3h 0.84, T+6h 0.55 -> event confidence 0.84,
  latest assessment 0.55.
* **Rules never decide.** The rule calculator runs for diagnostics only (recorded as `rule_status` / `agreement`) and cannot
  override NO_EVENT / SUSPECTED / CONFIRMED.
* **Response semantics (Task 9A, unchanged):** SUSPECTED is active for monitoring but triggers no severity, spread, targets,
  routing, Dijkstra, GA, global planning or commitment; CONFIRMED is response eligible.

## Failure handling (no silent fallback)
Missing / corrupt artifact, metadata or schema mismatch, feature-extraction or classifier failure, an invalid probability,
a history-loading failure, or an unmigrated audit table each fail the detection cycle with
`success=False` and `error_message="Fire detection (ai_hybrid_v5) failed: <path-free reason>"`. No rule-based event is
created in their place and a history failure is never treated as "no history".

## Persistence / migration
The existing `fire_event_ml_assessments` row (one per event, upserted) is reused. Five **nullable** columns were added for the
AI audit trail: `policy_version`, `policy_status`, `history_available`, `satellite_pass_count`, `current_satellite_pixel_count`.
They are mapped `deferred`, only written / read for `ai_hybrid_v5` rows, and legacy rows are inserted with an explicit column list -
so `rule_only` / `shadow` / `hybrid` keep working on a database that has **not** been migrated. Only `ai_hybrid_v5` needs it:

```
python -m scripts.migrate_add_fire_event_ml_assessment_ai_columns --inspect-only
python -m scripts.migrate_add_fire_event_ml_assessment_ai_columns --apply     # idempotent, additive
```
Task 9B did not run this migration; Task 9C applied it to the configured demo database (see `fire_detection_task9c_final_acceptance.md`).

## API visibility
Event details (`ml_assessment`) already exposes `mode`; it now may read `ai_hybrid_v5` and adds optional `policy_version`,
`policy_status`, `history_available`, `satellite_pass_count`, `current_satellite_pixel_count` (all `null` for other modes). The
25-feature vector is not exposed. No filesystem path or secret is returned.

## Known limitations
* Trained and confirmed only on synthetic data; no measure of real-world accuracy.
* No SUSPECTED expiry / decay yet: a SUSPECTED event stays active until resolved or dismissed by other means.
* The probability is model confidence, not real-world certainty.
* The locked policy has exactly one CONFIRMED branch (`P >= 0.80` and >= 2 current satellite pixels): news alone, or one pixel
  plus strong news, is never CONFIRMED, so early fires are often SUSPECTED.
* Candidates still inside the 120-minute evidence lookback are re-scored on repeated `detect()` cycles (the model is cheap;
  the latest-assessment row reflects the last candidate scored for the event).
* (Task 9B limitation, fixed in Task 9C) The FireEvent used to be written before its audit row; it is now stored atomically - see `fire_detection_task9c_final_acceptance.md`.
