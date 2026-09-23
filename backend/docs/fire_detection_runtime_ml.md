# Fire Detection - Runtime ML Integration (Task 5)

## Status

This integrates the trained V3 Logistic Regression model into the real runtime Fire Detection flow, as a second, subordinate signal alongside the unchanged deterministic `FireDetectionCalculator`. The deterministic calculator remains authoritative by default (`SHADOW` mode). No runtime classifier "replaces" it; `FireDetectionCalculator`'s methodology, weights, and thresholds are untouched, and all of its existing tests still pass with the exact same expected confidence values.

## Architecture

```text
                 FireDetectionCandidate
                          |
               +----------+----------+
               v                     v
     FireDetectionCalculator    MLFireDetectionClassifier
          (rule-based,               (Logistic Regression V3,
           unchanged)                 lazily loaded, cached)
               |                     |
               v                     v
        FireDetectionDecision  FireDetectionMLAssessment
               |                     |
               +----------+----------+
                          v
              FireDetectionHybridPolicy
              (decision_mode: RULE_ONLY / SHADOW / HYBRID)
                          v
              HybridFireDetectionDecision
                          v
                  FireDetectionAgent
                          v
        FireEvent (unchanged schema) + FireEventMLAssessment
        (new, separate table: rule/ML/decision trace for observability)
```

`FireDetectionAgent` never imports sklearn/joblib, never builds a raw feature array, and never references `StandardScaler` - all of that is contained in `MLFireDetectionClassifier` (`src/calculators/fire_detection/fire_detection_ml_classifier.py`). The Agent only ever sees `FireDetectionMLAssessment` and `HybridFireDetectionDecision`.

## Why Logistic Regression V3 (not Random Forest)

Both were compared in Task 4 on the same synthetic V3 dataset:

| | Standard held-out Accuracy/F1/ROC-AUC | Grouped (unseen scenario-family) ROC-AUC |
| --- | --- | --- |
| Logistic Regression | 0.758 / 0.754 / 0.849 | **0.590** |
| Random Forest | 0.831 / 0.838 / 0.928 | **0.378** (worse than chance is 0.5) |

Random Forest scores higher on ordinary random-split evaluation, but generalizes substantially worse to scenario families the model never saw during training, and showed a larger train/test overfitting gap (Task 4). Since the runtime model's job here is to support a real (if synthetic-trained) decision, and unseen-family robustness matters more than squeezing out standard-split accuracy, Logistic Regression V3 was selected as the more conservative choice. This is **not** a claim that Logistic Regression is "better" in general, or that either model is operationally validated - see "Probability interpretation" below.

## ML component

`MLFireDetectionClassifier` (`src/calculators/fire_detection/fire_detection_ml_classifier.py`):

1. Loads `fire_detection_logistic_v3.joblib` **lazily** (only on first `assess()` call or explicit `initialize()`, never merely by importing the module) and **caches** the loaded pipeline - the file is never reloaded per candidate, and a failed load is not retried on every call.
2. Loads and validates `fire_detection_logistic_v3_metadata.json` alongside it: `model_type` must be `"LogisticRegression"`, `feature_names` must exactly equal `FIRE_DETECTION_FEATURE_NAMES_V3` (order included), and `model_version` must be present. **Any mismatch refuses inference** - it never silently feeds features in the wrong order.
3. Uses `FireDetectionFeatureExtractorV3` (Task 4, unchanged) to turn candidate evidence into the V3 feature vector.
4. Calls `predict_proba(...)[1]` and wraps the result in a `FireDetectionMLAssessment`.

Configured via (`src/config/settings.py`, all overridable through environment variables):

- `FIRE_DETECTION_ML_MODEL_PATH` (default: `models/fire_detection/fire_detection_logistic_v3.joblib`, project-relative)
- `FIRE_DETECTION_ML_METADATA_PATH` (default: the matching `..._metadata.json`)

### Failure behavior

Any failure - missing/corrupt model file, metadata/schema mismatch, feature-extraction failure, `predict_proba` failure - is caught inside `MLFireDetectionClassifier.assess()` and turned into `FireDetectionMLAssessment(available=False, ..., failure_reason=...)`. It **never raises**, and `FireDetectionAgent.detect()` continues using the deterministic calculator alone. ML failure never stops Fire Detection.

## Runtime modes

`FireDetectionDecisionMode` (`src/models/fire_detection_decision_mode.py`), configured via `FIRE_DETECTION_DECISION_MODE` (default **`shadow`**):

- **`RULE_ONLY`**: historical, pre-Task-5 behavior. ML is never loaded and never called - not even constructed by default. No `FireEventMLAssessment` row is ever written for events created/updated in this mode (true backward compatibility: zero new writes).
- **`SHADOW`** (default): both the rule calculator and ML run. The final status/confidence/location **always** equal the rule decision's - ML can never change what `FireEvent` is created or updated. The ML assessment and rule/ML agreement are recorded (in the new `fire_event_ml_assessments` table) for observability.
- **`HYBRID`**: both run, and a conservative policy (below) may let ML promote a `NO_EVENT` rule decision to `SUSPECTED`. Not the default.

### Why SHADOW is default

Logistic Regression V3 is real supervised ML, but it is trained entirely on synthetic data, and Task 3/4's grouped (unseen-scenario-family) evaluation showed limited generalization (ROC-AUC as low as ~0.38-0.59 depending on version/model, well below the ~0.85-0.93 seen on ordinary random splits). Treating its probability as operational truth would be premature. `SHADOW` gives real runtime AI inference - genuinely computed on every candidate, recorded, and inspectable - without letting an incompletely-validated synthetic model become the sole decision mechanism for an emergency-relevant system.

## Hybrid policy (conservative, `HYBRID` mode only)

`FireDetectionHybridPolicy` (`src/calculators/fire_detection/fire_detection_decision_policy.py`) enforces, unconditionally:

- **`CONFIRMED` rule decision -> always final `CONFIRMED`.** ML cannot downgrade it.
- **`SUSPECTED` rule decision -> always at least final `SUSPECTED`.** ML cannot downgrade it, and cannot alone promote it to `CONFIRMED`.
- **`NO_EVENT` rule decision -> stays `NO_EVENT` unless ML is available and `probability >= FIRE_DETECTION_ML_SUSPECT_THRESHOLD`, in which case it may be promoted to `SUSPECTED` - never directly to `CONFIRMED`.**
- **ML unavailable -> final always equals the rule decision**, in every mode.
- **`final_confidence` always equals the rule decision's confidence**, in every mode, including an escalated `NO_EVENT -> SUSPECTED` case - ML never fabricates a confidence number. The escalation is explained by the recorded `ml_probability`/`decision_mode`/`agreement`, not by a manufactured confidence value.

ML can support, flag disagreement, or (in `HYBRID`, only for `NO_EVENT`) conservatively escalate to `SUSPECTED`. **It can never suppress or downgrade direct rule evidence.**

## Agreement classification

`FireDetectionMLRuleAgreement` (`src/models/fire_detection_ml_rule_agreement.py`), computed using `FIRE_DETECTION_ML_CLASSIFICATION_THRESHOLD` (default `0.50` - the standard binary threshold used during model evaluation, **not** tuned for operational escalation, used here purely to interpret "does ML say fire or not" for comparison):

- `AGREE_FIRE`: rule says fire (SUSPECTED/CONFIRMED), ML >= threshold.
- `AGREE_NO_FIRE`: rule says NO_EVENT, ML < threshold.
- `RULE_STRONGER`: rule says fire, ML < threshold (rule alone drives the outcome).
- `ML_STRONGER`: rule says NO_EVENT, ML >= threshold (ML alone would say fire).
- `ML_UNAVAILABLE`: no ML assessment could be produced.

## Escalation threshold selection

`FIRE_DETECTION_ML_SUSPECT_THRESHOLD` (the `NO_EVENT -> SUSPECTED` HYBRID escalation threshold) was chosen by `scripts/analyze_fire_detection_ml_threshold.py`, which fits the exact same Logistic Regression pipeline configuration used for the saved V3 model and gets **out-of-fold predicted probabilities via `cross_val_predict`** on `training_v3.csv` (5-fold stratified CV, `random_state=42`) - not the held-out test split, and not the saved production model artifact itself (which is untouched).

| Threshold | Precision | Recall | FPR | Predicted positive |
| --- | --- | --- | --- | --- |
| 0.50 | 0.767 | 0.719 | 0.218 | 1874 |
| 0.60 | 0.831 | 0.622 | 0.126 | 1495 |
| **0.70** | **0.908** | 0.540 | 0.055 | 1189 |
| 0.80 | 0.966 | 0.431 | 0.015 | 892 |
| 0.90 | 0.997 | 0.312 | 0.001 | 625 |
| 0.95 | 1.000 | 0.239 | 0.000 | 477 |

Target: precision >= 0.90 (a design target, per Task 5's brief - not a mandatory fabricated result). **0.70 is the smallest candidate threshold reaching it** (precision 0.908), so escalation is **enabled** at `FIRE_DETECTION_ML_SUSPECT_THRESHOLD=0.70`. Full results: `models/fire_detection/fire_detection_logistic_v3_threshold_analysis.json`. If no candidate threshold had reached the target, `FIRE_DETECTION_ML_SUSPECT_THRESHOLD` would be left unset (empty string), and `FireDetectionHybridPolicy` treats that as escalation disabled (HYBRID then behaves exactly like `SHADOW` for `NO_EVENT` rule decisions) - the code never invents a threshold.

## Probability interpretation

`FireDetectionMLAssessment.probability` is `predict_proba(...)[1]`: a **model-estimated probability within the synthetic V3 training distribution**. It is **not** a calibrated real-world probability of an active wildfire, and must never be presented as operational certainty (e.g. never "AI certainty = 86%" in any UI/API/log copy - prefer "ML probability" or "ML model score"). It reflects what the model learned from synthetic scenario families, not validated real-world statistics.

## Persistence

**No new fields were added to `FireEvent` or `fire_events`** (the widely-used, ~35-call-site domain model and table) - `FireEvent.status`/`detection_confidence` continue to mean exactly what they meant before Task 5 (the *final* decision's status/confidence; `detection_confidence` is always the rule decision's own confidence, even when `HYBRID` escalates status). Instead, a small new table, `fire_event_ml_assessments` (`FireEventMLAssessmentDB`), holds one row per `FireEvent` (upserted on reevaluation, not a growing history log), with a unique FK to `fire_events.id` (`ON DELETE CASCADE`):

- `decision_mode`, `rule_status`, `rule_confidence` (a snapshot of the rule decision at the last evaluation - useful because in `HYBRID` mode `FireEvent.status` can diverge from `rule_status`)
- `ml_available`, `ml_probability`, `ml_model_name`, `ml_model_version`, `ml_feature_schema_version`, `ml_failure_reason`
- `agreement`

This answers "what did the rule say / what did ML say / what final decision was used / did they agree / which model" without duplicating the V3 feature vector (recomputable from already-persisted evidence) or any existing `FireEvent` field.

Historical `FireEvent`s simply have **no matching row** in the new table - never backfilled, never a fabricated "ML not evaluated" row inserted retroactively. `RULE_ONLY` mode never writes a row here either (true backward compatibility).

### Migration

`scripts/migrate_add_fire_event_ml_assessments_table.py --inspect-only` / `--apply`. Purely additive (one new table via plain `CREATE TABLE`, fully portable, tested against real SQLite); `fire_events` itself is never touched. **Not applied automatically to Neon** - see the implementation report for the exact manual commands.

The earlier Task 4 migration (`wildfire_reports.wildfire_signal_strength`) has already been applied to Neon manually and is **not** re-touched by this task.

### Update-only-when-changed

`FireDetectionAgent._record_ml_assessment` compares the freshly computed assessment against the currently stored one and skips the write when nothing meaningful changed (same `decision_mode`, `rule_status`, `agreement`, `ml_model_version`, `ml_available`, and confidence/probability within a small tolerance: `RULE_CONFIDENCE_CHANGE_TOLERANCE = 1e-9`, `ML_PROBABILITY_CHANGE_TOLERANCE = 0.01`). Re-running `detect()` on unchanged evidence does not produce noisy DB writes.

### Reevaluation

When an existing `FireEvent` matches new evidence, `FireDetectionAgent._update_existing_event` recomputes **both** the rule decision and the ML assessment on the normalized union of evidence (existing + new) before persisting - an ML probability is never left stale after new evidence arrives. (Note: because the Agent also computes a candidate-level rule+ML pass first, to decide whether to skip an all-`NO_EVENT` candidate, ML ends up being called twice per merge-into-existing-event cycle - once on the raw candidate, once on the actual combined evidence that gets persisted. This is a minor inefficiency, not a correctness issue: the combined-evidence result is always what gets used/stored.)

## Simulation output

`FireDetectionResult` gained an optional `candidate_assessments: tuple[FireDetectionCandidateAssessment, ...] = ()` field (default empty - existing callers unaffected). `scripts/run_demo_simulation.py`'s `print_fire_detection_result` now prints one compact block per evaluated candidate:

```text
FIRE DETECTION
success=True
candidates_processed=1
...
  candidate: event_id=77 rule=confirmed(0.80) final=confirmed mode=shadow
    ml: probability=0.86 model_version=3.0 agreement=agree_fire
```

This is deliberately compact (2 lines per candidate), not verbose - but real: it reflects the actual `MLFireDetectionClassifier.assess()` output for that run, so a demo/presentation can show ML inference genuinely happening, not a silently-loaded, never-invoked model.

## API (ML Task 6)

`GET /api/v1/fire-events/{fire_event_id}/details` now exposes the persisted `FireEventMLAssessment`, read-only, as an optional nested `ml_assessment` object on `EventDetailsResult` (`src/api/schemas/event_details.py`, `FireEventMLAssessmentResponse`). `EventDetailsService._load_ml_assessment` calls the existing `FireEventRepository.get_ml_assessment(fire_event_id)` - a pure SELECT, no write, no ML inference triggered by an API read.

Fields (snake_case, matching this project's existing API convention - no camelCase aliasing anywhere in this router):

- `available`, `mode`, `rule_status`, `rule_confidence`, `agreement`, `model_version`, `feature_schema_version`, `updated_at` - copied verbatim from the persisted row.
- `model_score` (renamed from the domain field's `ml_probability`) - deliberately a distinct field from `rule_confidence`, never merged into one generic "confidence" value: they are different signals (deterministic rule score vs. a model-estimated score from Logistic Regression V3). See "Probability interpretation" above - `model_score` is **not** a calibrated real-world probability and the API/frontend must never describe it as e.g. "75% certainty of wildfire."
- `model_name` - the existing persisted `ml_model_name` field (e.g. `"fire_detection_logistic_v3"`); no new `model_type` field was invented, since this existing field already identifies the model.
- `failure_reason` - a **sanitized** category string, never the raw persisted `ml_failure_reason` text. `MLFireDetectionClassifier` sometimes embeds raw exception text (including filesystem paths, e.g. a missing model artifact's full path) in the persisted reason; `_sanitize_ml_failure_reason` (`event_details_service.py`) maps known failure-text prefixes to a short generic message and falls back to a fully generic one for anything unrecognized. The raw persisted text itself is untouched (still useful server-side/in logs) - only the API-facing copy is sanitized.

Optionality: `ml_assessment` is `null` for any FireEvent with no `fire_event_ml_assessments` row (pre-ML-integration FireEvents, and any FireEvent created while `FIRE_DETECTION_DECISION_MODE=rule_only`) - never a fabricated zero-valued object. When a row exists but `ml_available=false` (classifier failed), `model_score`/`model_name`/`model_version`/`feature_schema_version` are `null` and `agreement` is `ml_unavailable` - never a substituted `0`.

**`FireEvent.status` (the final operational decision) is always read independently of `ml_assessment`** - `EventDetailsResult.fire_event` is sourced from the persisted `FireEvent` row exactly as before this task, never from the ML assessment. In `shadow` mode this means `ml_assessment.mode == "shadow"` tells the frontend ML could not have influenced `fire_event.status`, without a manufactured field like `decision_source`. The schema also serializes `mode == "hybrid"` without any special-casing, even though `hybrid` is not the runtime default - no policy code changed to support this.

The primary `GET /api/v1/fire-events/active` list endpoint was intentionally left unchanged - ML exposure was scoped to the detail endpoint only, per this task's brief, to avoid payload noise on the list view. The full 16-value V3 feature vector is also intentionally not exposed here (no debug endpoint existed to reuse, and adding one was out of scope) - feature-level explainability remains a possible later task.

## AI-agent framing

`FireDetectionAgent` now performs:

- **Perception**: multi-source evidence retrieval/correlation (`FireDetectionEvidenceService`, unchanged).
- **AI reasoning**: trained Logistic Regression V3 inference (`MLFireDetectionClassifier`).
- **Deterministic reasoning**: the existing rule-based methodology (`FireDetectionCalculator`, unchanged).
- **Decision**: the hybrid policy (`FireDetectionHybridPolicy`).
- **Action**: create/update `FireEvent` (+ its `FireEventMLAssessment` trace).

This is the intended AI-agent architecture for this project. It remains a final-year-project proof of concept - not a claim of autonomous operational emergency infrastructure. The ML component is trained on synthetic data, its runtime default (`SHADOW`) cannot affect real decisions, and even `HYBRID`'s escalation path is capped at `SUSPECTED` with a precision-vetted threshold.
