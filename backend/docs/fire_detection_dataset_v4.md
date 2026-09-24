# Fire Detection ML - V4 Training Dataset

## Status

Offline dataset infrastructure only. **No model is trained here**, and nothing in
runtime behaviour changed: `FireDetectionAgent`, `FireDetectionHybridPolicy`,
`FIRE_DETECTION_DECISION_MODE`, the V3 classifier/artifacts, `.env` and FireEvent
persistence are untouched. V3 files (`training_v3.csv`, V3 generator, models) are
unchanged; V4 sits next to them.

Files:

| | |
| --- | --- |
| Schema (features / metadata / target) | `src/ml/fire_detection/fire_detection_features_v4.py` |
| Generator + scenario families | `src/ml/fire_detection/fire_detection_training_data_generator_v4.py` |
| CSV I/O + grouped splits | `src/ml/fire_detection/fire_detection_dataset_v4.py` |
| Validation + statistics | `src/ml/fire_detection/fire_detection_dataset_validation_v4.py` |
| Scripts | `scripts/generate_fire_detection_training_data_v4.py`, `scripts/validate_fire_detection_dataset_v4.py` |
| Output | `data/fire_detection/training_v4.csv`, `data/fire_detection/training_v4_validation_report.json` |

```bash
python -m scripts.generate_fire_detection_training_data_v4   # seed 42, 9000 rows; validates before writing
python -m scripts.validate_fire_detection_dataset_v4         # prints + writes the report
```

## What was wrong with V3 (why V4 exists)

The V3 label is ground truth (each scenario family has a fixed `label`), not a rule
output - that part was right. The problems are in what the families let a model
learn:

1. **Geometry shortcut.** Positive families were configured with wider spreads
   than negative ones (e.g. `high_confidence_satellite_false_positive` is always one
   hotspot at radius 0.3 km / 0 min). `max_pairwise_distance_km` alone reached
   AUC 0.70 (the top single feature) and candidates spread over >3 km were 80% fire.
   Distance/time-span are properties of the generator's family config, not of
   wildfire physics.
2. **Evidence shape nearly decides the label.** Knowing only whether satellite and
   news are present gives 69% majority-vote accuracy; single-item candidates were
   only 26% fire.
3. **No Fire Danger context**, so the model could not learn how weather modulates
   ambiguous evidence.
4. **Easy negatives.** Many negatives are pure noise (all-`low` hotspots, `NONE`
   news) - the rule detector already handles those; the hard cases are fire-looking
   evidence with no fire.
5. **Family = label.** Every family has exactly one label, so grouped CV by family
   trains and tests on folds with skewed class balance and gives a very pessimistic,
   high-variance signal (LR ROC-AUC 0.59 +/- 0.22), while standard CV leaks the
   family generator (0.84). Neither is a clean estimate.
6. **Simulator weather is a label proxy.** In the demo simulator, ACTIVE_FIRE weather
   is >= HIGH danger while the no-fire types are LOW/MODERATE/HIGH, so a naive reuse of
   `WEATHER_SCENARIO_PROFILES` would make FFWI a near-perfect shortcut.

## V4 label definition

`label = 1` iff a wildfire **actually exists** in the scenario. It is decided first,
from the scenario family's ground truth, and mapped through the simulation's own
`ScenarioType` (`ACTIVE_FIRE` -> 1; `LOW/MODERATE/HIGH_RISK_NO_FIRE` -> 0). Evidence is
generated afterwards and is noisy, missing or misleading independently of that truth.
The generator never imports or calls `FireDetectionCalculator`, the decision policy,
the ML classifier, `FireEvent`, or any rule/ML confidence; an automated test checks the
imports and another monkeypatches `FireDetectionCalculator.evaluate` to raise during
generation. Tests also show the rule detector disagrees with the label in both
directions (real fires it calls `NO_EVENT`, non-fires it `CONFIRM`s) - the very
signal the future model must learn.

## Scenario families (20; 450 rows each)

Each row is one independent candidate at one assessment time (`sample_id` doubles as
incident id, so no incident appears in two rows). Archetype = dominant evidence shape;
every archetype contains both labels.

**Fire exists (`label=1`)**

| Family | Archetype | Story |
| --- | --- | --- |
| P1_strong_satellite_strong_news_high_danger | satellite_news | the easy case; mostly HIGH/EXTREME danger |
| P2_satellite_only_fire | satellite_only | satellite sees it, no news yet |
| P3_news_first_satellite_later | news_led | news early; hotspot (if any yet) arrives later |
| P4_fire_moderate_danger | satellite_news | real fire under MODERATE FFWI |
| P5_fire_low_ffwi_despite_ignition | satellite_news | same evidence as P4 under LOW/MODERATE FFWI |
| P6_fire_missing_fire_danger | satellite_news | ~85% no Fire Danger assessment |
| P7_fire_weak_satellite_measurements | satellite_only | low/nominal, low FRP/brightness, often unmeasured |
| P8_fire_evidence_spread_over_time | multi_satellite | evidence accumulates over 30-58 min |
| P9_fire_persistent_multi_satellite | multi_satellite | 3-5 hotspots |
| P10_fire_news_led_multiple_reports | news_led | 2-4 reports, at most one late weak hotspot |

**No fire (`label=0`)**

| Family | Archetype | Story |
| --- | --- | --- |
| N1_extreme_danger_no_fire | satellite_only | EXTREME weather, weak ambiguous hotspots |
| N2_high_danger_no_fire_rumor | news_led | HIGH weather, smoke rumours |
| N3_isolated_low_confidence_hotspot_noise | satellite_only | isolated low-confidence hotspots |
| N4_false_or_weak_news_report | news_led | news-only false alarm, sometimes convincing |
| N5_satellite_plus_unrelated_news | satellite_news | hotspot + news not about a fire there |
| N6_multiple_weak_pieces_no_incident | multi_satellite | several weak hotspots + stray reports |
| N7_no_fire_missing_fire_danger | satellite_news | ambiguous evidence, ~85% no Fire Danger |
| N8_realistic_strong_looking_false_positive | satellite_news | nominal/high hotspots, sizeable FRP, convincing news |
| N9_persistent_non_wildfire_heat_source | multi_satellite | stationary industrial-type source seen repeatedly |
| N10_repeated_reports_of_same_false_alarm | news_led | several reports of the same unfounded alarm |

## Feature schema

Every CSV column is exactly one of (order: metadata, features, label):

* **ML features (19)** = the 16 unchanged V3 features
  (`FIRE_DETECTION_FEATURE_NAMES_V3`) + `fire_danger_available`, `fire_danger_score`,
  `fire_danger_age_minutes` (the Task 1 `FireDetectionContext` semantics). Since Task 3
  every value is produced by the canonical `FireDetectionFeatureExtractorV4` - the same
  extractor runtime inference will use - see `fire_detection_feature_pipeline_v4.md`.
* **Metadata only (never model input):** `sample_id`, `seed`, `scenario_family`,
  `scenario_archetype`, `ground_truth_scenario_type`, `fire_danger_band`
  (analysis only; not a feature), `as_of_utc`.
* **Target:** `label`.

`FORBIDDEN_FEATURE_NAMES_V4` / `forbidden_feature_names()` list every leakage-prone name
(scenario, family, archetype, ground truth, rule/ML outputs, SUSPECTED/CONFIRMED, event
status, timestamps, level/band); a test asserts none appear among the 19 features and
that the checker really flags leaky names.

**Missing Fire Danger is preserved, not imputed.** `fire_danger_available=0` and the score/age
cells are EMPTY (loaded as `None`); a real FFWI of `0.0` stays `0.0` (tested). Any
imputation belongs to a later preprocessing step, which must keep
`fire_danger_available` as a feature. `fire_danger_level` is not a feature (it is derived
from the score).

## How the data is generated

* Independent draws per row (`random.Random` seeded from `(seed, row index)`): centre in
  Israel, timestamp uniform over one year (label-independent by construction), counts,
  satellite confidence, FRP/brightness (sharing a latent per-scenario intensity, with
  per-item noise and missingness), news signal strength, evidence timing/offsets and
  spatial spread.
* Correlation semantics match runtime: every row must be a valid `FireDetectionCandidate`
  (max 5 km / 60 min; the generator stays inside 4.8 km / 58 min) and the features are
  extracted from that candidate.
* **Fire Danger** comes from the project's own weather profile boxes
  (`WEATHER_SCENARIO_PROFILES`) scored by the real `FFWICalculator`; EXTREME is the HIGH
  box restricted to EXTREME scores. The weather regime is chosen per family
  **independently of the ground truth**, then availability (about 12% missing, ~85%
  in P6/N7) and age (uniform 0-60 min, the Task 1 freshness limit) are drawn
  independently of the label. Fire baselines lean slightly hotter than no-fire baselines
  so FFWI is a weak prior (univariate AUC about 0.54), not a rule.
* Deterministic, offline: no network, DB, LLM or wall-clock dependency.

## Grouped evaluation design (for the next task)

Do not optimise for a random train/test split.

| Purpose | Field | Helper |
| --- | --- | --- |
| Unseen scenario family | `scenario_family` | `grouped_family_folds(rows, 5)` |
| Unseen evidence shape | `scenario_archetype` | `leave_one_archetype_out(rows)` |

`grouped_family_folds` deals fire and no-fire families round-robin (sorted by name), so
each of the 5 folds holds out 2 fire + 2 no-fire whole families (900 + 900 rows) - plain
`GroupKFold` would produce single-class folds because each family has one label. Every
held-out archetype contains both labels. Report metrics per fold and per held-out
group, not just the mean; compare against the rule-based decision on the same folds.

## Known limits

The data is still synthetic. Family parameters encode our assumptions about how real
evidence looks; grouped evaluation measures generalisation across those assumptions, not
real-world accuracy. Real-fire validation (e.g. historic FIRMS/news with labels) would be
needed before any claim about operational performance. The validation report lists the
remaining strongest signals honestly: FRP/brightness (AUC ~0.69) and news-signal
counts, which is where a real relationship should live.
