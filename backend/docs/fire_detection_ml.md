# Fire Detection - ML Foundation (V1)

## Status

This is an offline ML training foundation only. No runtime path uses a trained model yet. `FireDetectionAgent.detect()` and `FireDetectionCalculator` are unchanged and remain the production Fire Detection mechanism. See [fire_detection.md](fire_detection.md) for the existing deterministic system.

## Existing system

Fire Detection (User Story 2.2) currently uses a deterministic, manually configured evidence-scoring methodology: `FireDetectionCalculator` assigns fixed confidence weights per evidence source (satellite low/nominal/high, news), combines them with a noisy-or formula, and classifies the result against fixed thresholds (`NO_EVENT` / `SUSPECTED` / `CONFIRMED`). Those weights and thresholds are manually chosen EcoGuard methodology parameters, not learned from data.

## Why ML is being introduced

A manually configured scoring formula can only express relationships someone already thought to encode. Machine learning instead lets the relationship between evidence features and ground-truth wildfire state be *learned* from labeled examples. This does not replace the rule-based methodology - it gives EcoGuard a second, data-driven candidate mechanism that can be compared against it before any runtime decision is made about which one to trust.

## Why Logistic Regression was selected first

Logistic Regression is:

- a legitimate supervised machine-learning classifier for binary outcomes
- suitable for binary classification ("active wildfire" vs "no active wildfire")
- capable of returning calibrated probabilities (`predict_proba`), not just a hard label
- relatively interpretable - its coefficients show each feature's learned direction and relative weight
- a strong first baseline for a small, structured/tabular dataset like this one

**Logistic Regression is the initial ML baseline, not automatically the final production classifier.** A later task is expected to train `RandomForestClassifier` (and possibly others) on the same dataset and compare.

## Feature representation

Features are computed by `FireDetectionFeatureExtractor.extract()` (`src/ml/fire_detection/fire_detection_feature_extractor.py`) from an already-valid `FireDetectionEvidence` candidate group (the same connectivity/identity rules `FireDetectionCandidate` and `FireDetectionEvidenceService` use). The extractor is pure: no repositories, no Neon, no external APIs, and no dependency on `FireDetectionCalculator`.

The feature set and its column order are declared once, centrally, in `FIRE_DETECTION_FEATURE_NAMES` (`src/ml/fire_detection/fire_detection_features.py`):

| Feature | Meaning |
| --- | --- |
| `satellite_count` | Number of satellite evidence items in the candidate |
| `news_count` | Number of news evidence items in the candidate |
| `satellite_low_count` | Satellite items with normalized confidence `low` |
| `satellite_nominal_count` | Satellite items with normalized confidence `nominal` |
| `satellite_high_count` | Satellite items with normalized confidence `high` |
| `time_span_minutes` | Latest `observed_at` minus earliest `observed_at`; `0.0` for a single evidence item |
| `max_pairwise_distance_km` | Maximum Haversine distance between any two evidence items; `0.0` for a single evidence item |

Every script and test imports this constant rather than declaring its own copy, so a future runtime inference path and a future classifier comparison both stay compatible with the same representation.

### Deliberately excluded from V1

Raw latitude/longitude, database/evidence ids, location names, scenario family, and sample id are excluded so the classifier cannot memorize specific Israeli locations or synthetic scenario identifiers instead of learning the evidence pattern.

FFWI, Fire Danger level, Fire Severity, and FRP are also excluded, on architectural grounds: `FireDangerAssessmentAgent` answers "are conditions favorable for a wildfire," while `FireDetectionAgent` answers "do we have direct evidence one has started." Mixing those responsibilities into one feature set would blur that boundary. FRP may become a useful feature in a later model version, but not in this initial classifier.

## Synthetic training data

`FireDetectionTrainingDataGenerator` (`src/ml/fire_detection/fire_detection_training_data_generator.py`) produces a reproducible, seeded synthetic dataset of `(evidence, label)` samples, where `label = 1` means the synthetic scenario's ground truth contains an active wildfire and `label = 0` means it does not.

**The label never comes from `FireDetectionCalculator`.** The generator has no import of, and no dependency on, the rule-based calculator - using its output as a label would only teach a classifier to reproduce the existing manually chosen weights and thresholds, not to learn anything new. Labels come directly from which synthetic scenario family produced the sample.

### Why not reuse the demo simulation

The existing demo/simulation generators (`SatelliteDataGenerator`, `NewsDataGenerator`) only emit evidence for `ScenarioType.ACTIVE_FIRE`; every non-fire scenario emits none. Training directly on that simulation output would make the label trivially recoverable as "evidence exists -> fire, evidence absent -> no fire," so the classifier would learn almost nothing. The demo/simulation generators are unchanged in this task; they still drive existing demos and acceptance tests. A dedicated `FireDetectionTrainingDataGenerator` was built instead, which emits real `FireDetectionEvidence` for **both** labels.

### Scenario families

Every sample is generated from one of 13 named scenario families (6 positive, 7 negative), chosen so evidence source-type combinations deliberately overlap between the two labels - satellite-only, news-only, and satellite+news examples all occur under both `label=1` and `label=0` - and so satellite confidence alone does not determine the label (e.g. an occasional high-confidence satellite false positive, and a real fire first seen only at nominal confidence, both exist).

| Family | Label | Idea |
| --- | --- | --- |
| `high_confidence_satellite_with_news` | 1 | High-confidence satellite + nearby news |
| `nominal_satellite_with_news` | 1 | Nominal satellite + nearby news |
| `multi_satellite_real_fire` | 1 | Multiple correlated satellite detections, no news yet |
| `satellite_only_before_news` | 1 | Real fire seen by satellite before any news exists |
| `news_only_before_satellite` | 1 | Real fire reported in news before satellite catches it |
| `multi_satellite_plus_multi_news` | 1 | Multiple satellite detections + multiple news reports |
| `low_confidence_satellite_noise` | 0 | Low-confidence thermal anomaly, no wildfire |
| `nominal_satellite_false_heat_source` | 0 | Nominal-confidence non-fire heat source |
| `high_confidence_satellite_false_positive` | 0 | Occasional high-confidence satellite false positive |
| `news_rumor_no_fire` | 0 | Wildfire-related rumor/report, no actual fire |
| `multiple_news_same_false_alarm` | 0 | Several news reports describing the same false alarm |
| `correlated_satellite_news_false_alarm` | 0 | Satellite + news false alarm together |
| `persistent_non_wildfire_heat_source` | 0 | Several nearby satellite anomalies from a persistent non-fire heat source |

### Candidate validity

Every generated sample is a valid `FireDetectionCandidate`: all of a sample's evidence items are placed within a bounded radius of one shared center point and a bounded window around one shared base timestamp, so every pair correlates directly - a strictly stronger, and therefore safe, condition than the connected-component rule `FireDetectionCandidate` actually requires. The generator does not reimplement or loosen correlation; it just never produces evidence that could fail it. Candidate correlation itself remains `FireDetectionEvidenceService`/`FireDetectionCandidate`'s responsibility.

### Determinism

`FireDetectionTrainingDataGenerator(seed=...)` draws all randomness from one explicitly seeded `random.Random` instance - no uncontrolled global randomness. The same seed and sample count always produce equivalent output. Default seed: `42`. Default sample count: `2000`, split ~50/50 between labels by construction (exact split when the count is even).

## Dataset artifact

`python -m scripts.generate_fire_detection_training_data` writes `backend/data/fire_detection/training_v1.csv` with columns:

```text
sample_id, scenario_family, satellite_count, news_count,
satellite_low_count, satellite_nominal_count, satellite_high_count,
time_span_minutes, max_pairwise_distance_km, label
```

`sample_id` and `scenario_family` are metadata only - useful for inspecting the dataset - and must never be used as a model feature. The full column order (metadata + `FIRE_DETECTION_FEATURE_NAMES` + `label`) is declared once as `TRAINING_DATA_CSV_COLUMNS`, alongside the feature list.

The CSV is the ML training artifact. It is **not** stored in Neon, and this task adds no database migration for it - Neon remains the operational runtime database for satellite hotspots, wildfire reports, `FireEvent`s, and operational assessments.

## Training

```text
features + known label
        |
      fit()
        |
learned coefficients
```

`python -m scripts.train_fire_detection_model`:

1. Loads `training_v1.csv`, selecting only the `FIRE_DETECTION_FEATURE_NAMES` columns as `X` and `label` as `y`.
2. Splits 80% train / 20% test, stratified by label, `random_state=42`.
3. Fits an `sklearn.pipeline.Pipeline` of `StandardScaler -> LogisticRegression(max_iter=1000, random_state=42)`. Scaling is fit only on the training split; features are never manually pre-scaled before the pipeline.
4. Evaluates on the held-out test split: accuracy, precision, recall, F1, ROC-AUC, and the full confusion matrix (TP/TN/FP/FN) - not accuracy alone, since false negatives matter most for wildfire detection.
5. Prints the learned coefficients per feature.
6. Saves the fitted `Pipeline` (scaler + classifier together) with `joblib` to `backend/models/fire_detection/fire_detection_logistic_v1.joblib`.
7. Saves training metadata (feature names, sample counts, metrics, coefficients) to `backend/models/fire_detection/fire_detection_logistic_v1_metadata.json`.

### Reading the coefficients

Because `StandardScaler` is part of the pipeline, the learned coefficients correspond to **standardized** features (zero mean, unit variance) - not raw evidence counts/minutes/kilometers. A coefficient's sign and relative magnitude indicate direction and relative importance; it is not a direct "+1 satellite item changes the odds by X" statement in raw units.

## Offline training vs. future runtime (not implemented in this task)

```text
OFFLINE TRAINING (this task)

synthetic labeled candidates
        |
feature extraction (FireDetectionFeatureExtractor)
        |
StandardScaler
        |
LogisticRegression.fit()
        |
saved Pipeline (.joblib)
```

```text
RUNTIME (future task, NOT implemented here)

Satellite + News
        |
FireDetectionEvidenceService
        |
FireDetectionCandidate
        |
Feature Extractor
        |
trained classifier
        |
P(active wildfire)
        |
FireDetectionDecision
```

`FireDetectionAgent.detect()` is unchanged and does not load, call, or reference the trained model. No API, frontend, database migration, or simulation-orchestration code was touched.

## Dataset limitation

**The V1 dataset is synthetic.** It is a proof-of-concept dataset built to exercise the feature representation and training pipeline, not real satellite/news history. This model is not validated as an operational wildfire-detection model, and metrics measured on synthetic held-out data must not be presented or treated as real-world operational accuracy.

## Observed V1 results

A run with the default seed (`42`) and 2000 samples produced accuracy ≈0.72, ROC-AUC ≈0.85 on the held-out test split (see the metadata JSON for the exact run's numbers). This is a real, imperfect, learned classifier - not a trivial 100%/1.0 result - which is expected: the scenario families deliberately overlap in evidence composition between the two labels (see "Scenario families" above), so the dataset should not be perfectly separable. If a future re-run of this generator/trainer instead reaches near-100% accuracy/F1/ROC-AUC, that should be treated as a signal to inspect for label leakage (e.g. a feature or scenario-family artifact uniquely determining the label) rather than as a success.

## Baselines and future comparison

```text
              same dataset (training_v1.csv)
                     |
            +--------+--------+
            |                 |
   Logistic Regression   Random Forest / others
            |                 |
            +--------+--------+
                     |
              compare metrics
                     |
           select runtime model (future task)
```

This task trains only Logistic Regression. `FireDetectionCalculator` remains the deterministic Rule-Based Fire Detection baseline, untouched, for future comparison. The dataset (`training_v1.csv`) and feature representation (`FireDetectionFeatures` / `FireDetectionFeatureExtractor`) are reusable as-is by a future task training `RandomForestClassifier` or other classifiers - no regeneration or feature-representation change is required to do so. Runtime model selection is not part of this task.

**Update:** the Logistic Regression vs. Random Forest comparison described above has since been carried out on this same dataset. See [fire_detection_model_comparison.md](fire_detection_model_comparison.md) for the full comparison (dataset statistics by class, correlation analysis, held-out and cross-validated metrics for both models, feature importances, and an explanation of the `satellite_count` coefficient's counter-intuitive sign via feature multicollinearity). Runtime model selection is still not decided - that comparison also stops short of it.

**Second update:** that comparison also surfaced a V1 dataset artifact - `satellite_low_count` was 0 for every positive row, an unintended generator shortcut rather than a realistic property. A `training_v2.csv` with mixed-confidence evidence (including LOW satellite readings for some positive examples) was created to fix this, alongside a harder "unseen scenario family" grouped cross-validation. See [fire_detection_model_comparison_v2.md](fire_detection_model_comparison_v2.md) - notably, Random Forest's standard-evaluation advantage over Logistic Regression does not hold under grouped evaluation. `training_v1.csv` remains unchanged and reproducible.

**Third update:** V2's grouped evaluation exposed a deeper *feature representation* problem, not an algorithm problem - some V2 "twin" scenario families (e.g. a real early fire reported only by one news article vs. a false rumor) reduced to identical feature vectors. Task 4 exposed richer evidence already available in the system (satellite FRP/brightness, and a new LLM-derived news semantic signal) to a new V3 feature schema and dataset (`training_v3.csv`), and added a leave-one-archetype-out generalization check and a feature-group ablation. See [fire_detection_feature_representation_v3.md](fire_detection_feature_representation_v3.md) - the richer representation measurably reduces (but does not solve) the twin-family collision and improves grouped-CV stability. `training_v1.csv`/`training_v2.csv` remain unchanged and reproducible.

**Fourth update:** the V3 Logistic Regression model has since been integrated into the real runtime `FireDetectionAgent` (Task 5), as a subordinate signal - the deterministic calculator remains authoritative by default. See [fire_detection_runtime_ml.md](fire_detection_runtime_ml.md) for the runtime architecture, decision modes (`RULE_ONLY`/`SHADOW`/`HYBRID`), and why `SHADOW` is the default.
