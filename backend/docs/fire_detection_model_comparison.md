# Fire Detection ML - Model Comparison (V1 dataset)

## Status

This is an offline analysis and comparison exercise. It does not select a runtime classifier and does not change `FireDetectionAgent`, `FireDetectionCalculator`, or any production path. See [fire_detection_ml.md](fire_detection_ml.md) for the ML foundation this builds on, and [fire_detection.md](fire_detection.md) for the unchanged rule-based system.

## Why compare Logistic Regression and Random Forest

Logistic Regression provides a simple probabilistic classification, interpretability (coefficients), and a linear decision boundary. Random Forest provides nonlinear decision boundaries, automatic handling of feature interactions, no need for feature scaling, and general robustness on structured/tabular datasets. Neither is assumed better going in - this document reports what was empirically measured on the V1 dataset.

## Reused artifacts

`training_v1.csv` was **not** regenerated. Both classifiers use the exact same 2000-row dataset (1000 positive / 1000 negative), the same `FIRE_DETECTION_FEATURE_NAMES` feature columns, the same 80/20 stratified held-out split (`test_size=0.2, random_state=42`), and the same 5-fold stratified cross-validation folds (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). The existing `fire_detection_logistic_v1.joblib`/metadata from Task 1 were not overwritten; Logistic Regression is refit in-memory inside `scripts/compare_fire_detection_models.py` purely so it can be measured on identical folds/split to Random Forest.

## Dataset statistics by class

Full per-feature count/mean/median/std/min/max for both classes is produced by `python -m scripts.analyze_fire_detection_dataset`. Headline means:

| Feature | label=0 mean | label=1 mean |
| --- | --- | --- |
| `satellite_count` | 1.302 | 1.813 |
| `news_count` | 0.794 | 1.128 |
| `satellite_low_count` | 0.645 | 0.000 |
| `satellite_nominal_count` | 0.513 | 0.947 |
| `satellite_high_count` | 0.144 | 0.866 |
| `time_span_minutes` | 10.27 | 15.93 |
| `max_pairwise_distance_km` | 0.964 | 1.589 |

`satellite_low_count` is exactly `0.0` (mean, std, min, max) for every label=1 row: none of the six positive scenario families generate low-confidence satellite evidence, by construction. This is a strong, but not perfectly deterministic, negative signal - many negative families (`high_confidence_satellite_false_positive`, `nominal_satellite_false_heat_source`, `news_rumor_no_fire`, `multiple_news_same_false_alarm`) also have `satellite_low_count == 0`, so "low count is 0" does not by itself imply label=1.

## Inspecting the "unintuitive" Logistic Regression coefficients

| Coefficient | Sign | Marginal (per-feature) pattern in the data |
| --- | --- | --- |
| `max_pairwise_distance_km` | `+0.871` | label=1 mean (1.589) > label=0 mean (0.964) - **consistent** with the coefficient sign |
| `time_span_minutes` | `+0.206` | label=1 mean (15.93) > label=0 mean (10.27) - **consistent** with the coefficient sign |
| `satellite_count` | `-0.684` | label=1 mean (1.813) > label=0 mean (1.302) - **opposite** to the coefficient sign |

`max_pairwise_distance_km` and `time_span_minutes` are not actually surprising once the marginal statistics are inspected: positive (fire) examples do have larger spatial spread and longer time span on average in this synthetic dataset, and the Logistic Regression coefficient sign agrees with that.

`satellite_count` is the genuinely counter-intuitive one: univariately, higher `satellite_count` is *more* associated with label=1, yet its multivariate coefficient is negative. The correlation matrix (below) shows why: `satellite_count` is highly correlated with `satellite_nominal_count` (r=0.611), `satellite_high_count` (r=0.428), and `satellite_low_count` (r=0.353) - and, structurally, `satellite_count` **exactly equals** their sum for every row (verified: `satellite_count_identity_holds` is `True` on the full dataset). Logistic Regression fits one linear combination of all seven (correlated, partially redundant) features simultaneously; when `satellite_nominal_count` and `satellite_high_count` already carry strong positive signal for label=1, the model can assign `satellite_count` a compensating negative weight without changing what the *combination* predicts for any real row (since a real row's `satellite_count` cannot vary independently of its three confidence sub-counts). This is a standard multicollinearity effect, not a sign that the model or dataset is broken - but it does mean **`satellite_count`'s coefficient must not be read on its own** ("more satellite evidence reduces predicted fire probability") as a causal or even a reliable directional statement. See "Coefficient interpretation" below.

## Feature correlation matrix

Pearson correlation between the seven ML features (full 2000-row dataset):

| | satellite_count | news_count | low | nominal | high | time_span | max_dist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **satellite_count** | 1.000 | -0.271 | 0.353 | 0.611 | 0.428 | 0.538 | 0.670 |
| **news_count** | -0.271 | 1.000 | -0.270 | -0.165 | 0.047 | 0.388 | 0.263 |
| **satellite_low_count** | 0.353 | -0.270 | 1.000 | -0.147 | -0.257 | 0.168 | 0.235 |
| **satellite_nominal_count** | 0.611 | -0.165 | -0.147 | 1.000 | -0.116 | 0.373 | 0.397 |
| **satellite_high_count** | 0.428 | 0.047 | -0.257 | -0.116 | 1.000 | 0.200 | 0.302 |
| **time_span_minutes** | 0.538 | 0.388 | 0.168 | 0.373 | 0.200 | 1.000 | 0.659 |
| **max_pairwise_distance_km** | 0.670 | 0.263 | 0.235 | 0.397 | 0.302 | 0.659 | 1.000 |

`satellite_count = satellite_low_count + satellite_nominal_count + satellite_high_count` holds exactly for all 2000 rows (structural redundancy, confirmed programmatically, not merely the ~0.35-0.61 pairwise correlations shown above, which understate it because they are pairwise rather than the full three-way sum relationship). This redundancy is the primary source of Logistic Regression multicollinearity in this feature set. No feature was removed in this task - feature-set redesign, if warranted, is a decision for a later task.

`scenario_family` (metadata, never a feature) perfectly determines `label` within this synthetic dataset, since each family was designed to represent either a positive or a negative scenario. This is expected and safe precisely because `scenario_family` is excluded from `FIRE_DETECTION_FEATURE_NAMES` and never reaches either model.

## Logistic Regression

Held-out test metrics (400 rows, same split as Random Forest below):

| Metric | Value |
| --- | --- |
| Accuracy | 0.7150 |
| Precision | 0.7067 |
| Recall | 0.7350 |
| F1 | 0.7206 |
| ROC-AUC | 0.8514 |
| FP / FN | 61 / 53 |

5-fold cross-validation (mean ± std, full 2000-row dataset):

| Metric | Value |
| --- | --- |
| Accuracy | 0.743 ± 0.004 |
| Precision | 0.740 ± 0.016 |
| Recall | 0.750 ± 0.029 |
| F1 | 0.744 ± 0.008 |
| ROC-AUC | 0.865 ± 0.017 |

CV results are close to, and slightly better than, the single held-out split - consistent (no sign the held-out split was an unusually hard or easy draw).

### Coefficient interpretation

Logistic Regression coefficients are fit over `StandardScaler`-standardized features, so they are not directly "raw units" statements. More importantly, because the features are correlated - and `satellite_count` is *structurally* dependent on the three confidence-specific counts - individual coefficient signs must not be read as independent causal rules. Do **not** conclude "more satellite evidence reduces fire probability" from the negative `satellite_count` coefficient; the model's actual prediction for any given row is the joint linear combination of all seven correlated features together, and only that combination (not any single coefficient in isolation) is meaningful.

## Random Forest

Configuration: `RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None)` - sklearn defaults otherwise (`max_depth=None`, `min_samples_leaf=1`). No scaling applied (tree-based, not needed). No hyperparameter search performed; no tuning against the held-out test set.

Held-out test metrics (same 400-row split as Logistic Regression):

| Metric | Value |
| --- | --- |
| Accuracy | 0.8350 |
| Precision | 0.8895 |
| Recall | 0.7650 |
| F1 | 0.8226 |
| ROC-AUC | 0.9223 |
| FP / FN | 19 / 47 |

Confusion matrix: TN=181, FP=19, FN=47, TP=153 (derived from Accuracy/FP/FN above and 400 total test rows: TN=400-19-47-153=181, TP=200-47=153).

5-fold cross-validation (mean ± std, same folds as Logistic Regression):

| Metric | Value |
| --- | --- |
| Accuracy | 0.825 ± 0.020 |
| Precision | 0.872 ± 0.019 |
| Recall | 0.763 ± 0.034 |
| F1 | 0.814 ± 0.024 |
| ROC-AUC | 0.921 ± 0.017 |

### Overfitting check

| | Accuracy | F1 |
| --- | --- | --- |
| Train (1600 rows) | 0.9156 | 0.9078 |
| Test (400 rows) | 0.8350 | 0.8226 |

There is a real gap (~8 points on both metrics) between train and held-out test performance. This is a **moderate overfitting signal**, not the "near-perfect train, much lower test" pattern that would indicate the model has essentially memorized the training rows - train accuracy is 0.92, not ~1.0 - but it is real and reported plainly rather than tuned away. No hyperparameter changes (e.g. `max_depth`, `min_samples_leaf`) were applied to close this gap, per this task's instructions not to tune against held-out results.

### Feature importances

Impurity-based (`feature_importances_`, descending):

| Feature | Importance |
| --- | --- |
| `satellite_low_count` | 0.2229 |
| `time_span_minutes` | 0.2052 |
| `max_pairwise_distance_km` | 0.2031 |
| `satellite_high_count` | 0.1691 |
| `news_count` | 0.0839 |
| `satellite_count` | 0.0671 |
| `satellite_nominal_count` | 0.0489 |

Permutation importance (held-out test set, `scoring="f1"`, `n_repeats=10`, descending):

| Feature | Importance |
| --- | --- |
| `satellite_low_count` | +0.1294 |
| `time_span_minutes` | +0.0656 |
| `satellite_high_count` | +0.0593 |
| `max_pairwise_distance_km` | +0.0570 |
| `news_count` | +0.0400 |
| `satellite_nominal_count` | +0.0182 |
| `satellite_count` | +0.0104 |

The two rankings agree closely (same top feature, same bottom two). Impurity-based importance is biased toward high-cardinality/correlated features and is not a causal measure; it is reported alongside the held-out permutation importance specifically because this feature set has known correlation (see above). Both agree that `satellite_count` - the structurally redundant feature - contributes the least once the three confidence-specific counts are already available, which is consistent with the Logistic Regression multicollinearity finding above (drawn independently, from a different model family, which is reassuring rather than coincidental).

## Direct comparison

### Held-Out Test Metrics

| Metric | Logistic Regression | Random Forest |
| --- | --- | --- |
| Accuracy | 0.7150 | 0.8350 |
| Precision | 0.7067 | 0.8895 |
| Recall | 0.7350 | 0.7650 |
| F1 | 0.7206 | 0.8226 |
| ROC-AUC | 0.8514 | 0.9223 |
| False Positives | 61 | 19 |
| False Negatives | 53 | 47 |

### 5-Fold Cross-Validation (mean ± std)

| Metric | Logistic Regression | Random Forest |
| --- | --- | --- |
| Accuracy | 0.743 ± 0.004 | 0.825 ± 0.020 |
| Precision | 0.740 ± 0.016 | 0.872 ± 0.019 |
| Recall | 0.750 ± 0.029 | 0.763 ± 0.034 |
| F1 | 0.744 ± 0.008 | 0.814 ± 0.024 |
| ROC-AUC | 0.865 ± 0.017 | 0.921 ± 0.017 |

Random Forest scores higher on every metric except Recall, where Logistic Regression is slightly higher (0.735/0.750 vs. 0.765/0.763). Random Forest has noticeably higher variance across CV folds on Accuracy/F1 (std ~0.02-0.024 vs. LR's ~0.004-0.008), i.e. it is more sensitive to which rows land in which fold.

### False positives / false negatives (fire-detection-specific reading)

A **false negative** here means a synthetic active-fire candidate the classifier labeled as no-fire - the operationally dangerous error for a wildfire detection system. A **false positive** means a synthetic no-fire candidate labeled as fire - an operational nuisance (unnecessary dispatch/alert), not a safety risk.

- Logistic Regression: 61 FP, 53 FN (held-out test, 400 rows)
- Random Forest: 19 FP, 47 FN (held-out test, 400 rows)

Random Forest has fewer false positives (19 vs 61) and slightly fewer false negatives (47 vs 53) - i.e. it dominates Logistic Regression on both error types on this held-out split, not merely on aggregate accuracy. Recall (fraction of real fires caught) is close between the two: LR 0.735 vs RF 0.765 (held-out) / 0.750 vs 0.763 (CV mean) - RF is marginally better on average but the CV standard deviations overlap, so this is not a decisive gap. Neither model was optimized past its standard 0.5 decision threshold in this task; threshold tuning (e.g. lowering the classification threshold to further reduce false negatives at some false-positive cost) is left to a future task once a runtime classifier is selected.

## Optional: rule-based `FireDetectionCalculator` on the same held-out rows

As an additional, low-effort analysis (not required, and no code change to the calculator), the unmodified `FireDetectionCalculator` was evaluated on the exact same 400 held-out rows, using evidence regenerated by `FireDetectionTrainingDataGenerator(seed=42)` after confirming the regeneration reproduces `training_v1.csv` exactly (row-for-row, feature-for-feature). `NO_EVENT` was mapped to predicted label 0; `SUSPECTED`/`CONFIRMED` to predicted label 1.

| Metric | Rule-Based Calculator |
| --- | --- |
| Accuracy | 0.5700 |
| Precision | 0.5376 |
| Recall | 1.0000 |
| F1 | 0.6993 |
| ROC-AUC (using `decision.confidence` as score) | 0.7683 |
| TN / FP / FN / TP | 28 / 172 / 0 / 200 |

The rule-based calculator never misses a synthetic active fire on this dataset (Recall = 1.0, FN = 0) but raises far more false alarms than either ML model (FP = 172, vs. 61 for Logistic Regression and 19 for Random Forest). This is expected, not a defect: `FireDetectionCalculator`'s thresholds (news alone -> `SUSPECTED`; any nominal/high satellite evidence -> at least `SUSPECTED`) were manually tuned for the production correlation pipeline, not for this synthetic V1 dataset's deliberately adversarial false-positive scenario families (a false alarm with real news coverage, or a non-fire nominal-confidence heat source, is by design meant to look plausible). The two ML models, by contrast, were trained directly on these same synthetic false positives and learned to discriminate them - something the fixed rule-based thresholds cannot do without being explicitly re-tuned. `decision.confidence`'s ROC-AUC (0.768) is included as a courtesy comparison point only; it comes from a rule-based heuristic score, not a trained/calibrated probability, and should be read with that caveat, not as directly comparable to the LR/RF ROC-AUC numbers above.

## Important scientific limitation

**The dataset is synthetic.** All metrics in this document measure relative performance on the synthetic V1 distribution only. They do not represent real-world wildfire-detection operational accuracy, and a model performing better here is not automatically better for real-world deployment. This comparison is useful for understanding the feature representation, the classifiers' relative behavior, and dataset quality - not as a substitute for real-world validation.

## No runtime selection

This document reports factual trade-offs. It does not declare a winner, and no runtime classifier is selected here. `FireDetectionAgent` and `FireDetectionCalculator` remain unchanged; a future task is expected to make the runtime-selection decision (and to consider threshold tuning, real-world validation needs, and possibly additional classifiers) using this comparison as input.
