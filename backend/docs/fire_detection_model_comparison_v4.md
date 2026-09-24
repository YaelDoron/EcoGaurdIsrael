# Fire Detection ML - V4 Model Comparison

> **All numbers below are measured on the synthetic V4 benchmark (`training_v4.csv`).**
> They describe how models generalise across *synthetic* scenario families and evidence
> shapes. They are **not** real-world wildfire detection accuracy.

## Verdict

**No V4 model passes the (a-priori) generalisation gate, so no runtime-ready artifact was
created and V4 should not become the primary scorer yet.** V4 ML does improve on the
hand-written rule in useful ways (ranking quality, far fewer false alarms), but not by
enough - and not robustly enough on unseen families - to hand it the decision. Notably,
**Fire Danger (FFWI) did not improve generalisation; it made every model slightly worse.**

Nothing in runtime changed: `FireDetectionAgent`, `FireDetectionHybridPolicy`,
`FIRE_DETECTION_DECISION_MODE`, `.env` and the V3 classifier/artifacts are untouched, and
no FireEvent is influenced by V4.

Full machine-readable results: `data/fire_detection/fire_detection_model_comparison_v4.json`
(reproducible: two independent full runs produced byte-identical reports).
Reproduce with `python -m scripts.compare_fire_detection_models_v4` (about 7-9 minutes).

## Data, environment, reproducibility

| | |
| --- | --- |
| Dataset | `data/fire_detection/training_v4.csv`, frozen, 9,000 rows, 20 scenario families, 19 features |
| Dataset SHA-256 | `6e14edc5ff536f3e55e680dfe5e89862b9e712ea1e5a6639be2a1e10767d812f` |
| Python / scikit-learn / numpy / joblib | 3.9.6 / 1.6.1 / 2.0.2 / 1.5.3 |
| Random seed | 42 everywhere; models are single-threaded (`n_jobs=1`) |

The feature schema is validated against `validate_feature_names_v4()` before any training:
a reordered, dropped or leakage-named feature column is refused. The design matrix is built
from the validated `FireDetectionFeaturesV4` contract, so metadata and the label can never
enter the model input.

## Models and preprocessing

All three families share one preprocessing design, inside a single sklearn `Pipeline`
(nothing is imputed or scaled outside it, and fitting uses training rows only):

1. **`SimpleImputer(strategy="median")` only on the two nullable Fire Danger numerics**
   (`fire_danger_score`, `fire_danger_age_minutes`).
2. **`fire_danger_available` is never imputed** - it is a real binary feature and is passed
   through, so the model can always tell an imputed score from an observed one. All other
   features pass through unchanged.
3. `StandardScaler` for Logistic Regression only.
4. The classifier.

| Model | Pipeline | Hyperparameters (fixed a priori, not tuned on the evaluation folds) |
| --- | --- | --- |
| **Logistic Regression V4** | impute -> scale -> `LogisticRegression` | L2, C=1.0, max_iter=2000 |
| **Random Forest V4** | impute -> `RandomForestClassifier` | 300 trees, min_samples_leaf=10, max_features=sqrt |
| **Histogram Gradient Boosting V4** | impute -> `HistGradientBoostingClassifier` | depth 3, lr 0.05, 200 iterations, min_samples_leaf=40, L2=1.0 |

Hyperparameter sensitivity (grouped ROC-AUC): LR is flat across C in {0.1, 1, 10}
(0.733-0.736); Random Forest 0.659 / 0.676 / 0.689 for min_samples_leaf 2 / 10 / 30;
gradient boosting 0.721 / 0.702 / 0.686 for depth 2 / 3 / 5. No setting approaches the gate's
0.80, so the conclusion does not depend on the chosen values.

## Evaluation design

* **Primary (model selection): grouped by scenario family** - 5 folds, whole families held out
  (2 fire + 2 no-fire per fold), from the frozen `grouped_family_folds` helper. Thresholds,
  calibration and importances use only these out-of-fold predictions.
* **Secondary: leave-one-archetype-out** (4 folds).
* **Diagnostic only: random 80/20 split** and random 5-fold CV. Never used to choose a model.
* **Baseline: the current rule-based `FireDetectionCalculator`**, evaluated against the same
  ground-truth labels (SUSPECTED or CONFIRMED = fire). It is never used as a label.
* Fold-metric operating point 0.5; std is the population std over folds.

## Random split - diagnostic only

| Model | Accuracy | F1 | ROC-AUC | 5-fold CV ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.760 | 0.760 | 0.849 | 0.839 +/- 0.008 |
| Random Forest | 0.797 | 0.799 | 0.889 | 0.879 +/- 0.003 |
| Gradient Boosting | 0.790 | 0.790 | 0.879 | 0.871 +/- 0.005 |

These look much better than the grouped numbers below - exactly the V3 pattern. They measure
"a new sample from a family the model already saw", so they are not used for selection.

## Grouped-family results (primary)

Held-out families per fold: 0 = N10, N5, P10, P5; 1 = N1, N6, P1, P6; 2 = N2, N7, P2, P7;
3 = N3, N8, P3, P8; 4 = N4, N9, P4, P9.

ROC-AUC per fold (F1 in brackets):

| Model | f0 | f1 | f2 | f3 | f4 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.827 (0.689) | 0.877 (0.777) | 0.534 (0.336) | 0.599 (0.627) | 0.841 (0.778) |
| Random Forest | 0.817 (0.719) | 0.787 (0.750) | 0.486 (0.399) | 0.463 (0.527) | 0.826 (0.768) |
| Gradient Boosting | 0.823 (0.697) | 0.818 (0.749) | 0.507 (0.308) | 0.534 (0.563) | 0.827 (0.770) |

Summary, mean +/- std [min, max]:

| | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.663 +/- 0.107 | 0.655 +/- 0.115 | 0.653 +/- 0.221 | 0.641 +/- 0.163 [0.336, 0.778] | **0.736 +/- 0.140** [0.534, 0.877] |
| Random Forest | 0.631 +/- 0.124 | 0.622 +/- 0.112 | 0.664 +/- 0.193 | 0.633 +/- 0.145 [0.399, 0.768] | 0.676 +/- 0.165 [0.463, 0.826] |
| Gradient Boosting | 0.632 +/- 0.118 | 0.617 +/- 0.124 | 0.642 +/- 0.224 | 0.617 +/- 0.171 [0.308, 0.770] | 0.702 +/- 0.148 [0.507, 0.827] |
| Rule baseline | 0.520 +/- 0.078 | 0.513 +/- 0.046 | 0.964 +/- 0.056 | 0.669 +/- 0.048 [0.598, 0.747] | 0.597 +/- 0.185 [0.272, 0.782] |

Take-aways: Logistic Regression is the most robust model (tree models overfit families more,
as in V3); results swing widely by fold (AUC 0.53-0.88), so no model is reliable across
families; the rule's AUC uses its own confidence as the ranking score.

**Fold 2 collapses for every model.** It holds out *both* satellite-only fire families (P2
and P7), so the training set contains no satellite-only fire evidence. That is a property of
the frozen fold layout (families dealt round-robin by name), not a dataset bug. It is not what
limits performance overall: a supplementary leave-one-family-out check (sibling families kept
in training) does not score better - pooled ROC-AUC 0.672 (LR) / 0.617 (RF) / 0.634 (HGB)
versus 0.726 / 0.670 / 0.694 pooled over the 5 grouped folds - although that design has its
own bias (single-label groups; predictions pooled from 20 different models). Weaknesses that
persist under both layouts are per-family (below).

## Leave-one-archetype-out

ROC-AUC (F1) when a whole evidence shape is held out:

| Model | multi_satellite | news_led | satellite_news | satellite_only | mean AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.879 (0.811) | 0.749 (0.518) | 0.781 (0.785) | 0.703 (0.489) | 0.778 +/- 0.065 |
| Random Forest | 0.860 (0.803) | 0.786 (0.525) | 0.738 (0.771) | 0.748 (0.699) | 0.783 +/- 0.048 |
| Gradient Boosting | 0.870 (0.786) | 0.790 (0.518) | 0.730 (0.772) | 0.723 (0.672) | 0.778 +/- 0.059 |

The ranking signal partly transfers to unseen evidence shapes (AUC 0.70-0.88), but recall on
`news_led` and `satellite_only` is low for Logistic Regression (0.41 / 0.37 at threshold 0.5).

## Rule baseline comparison (pooled, out-of-fold for the models, threshold 0.5)

| | Precision | Recall | F1 | False positives | False negatives |
| --- | --- | --- | --- | --- | --- |
| Rule (SUSPECTED or CONFIRMED) | 0.511 | 0.964 | 0.667 | 4,157 | 164 |
| Rule (CONFIRMED only) | 0.520 | 0.497 | 0.508 | 2,068 | 2,263 |
| Logistic Regression V4 | 0.666 | 0.653 | 0.660 | 1,471 | 1,562 |
| Random Forest V4 | 0.623 | 0.664 | 0.643 | 1,805 | 1,514 |
| Gradient Boosting V4 | 0.629 | 0.642 | 0.636 | 1,700 | 1,612 |

The rule declares almost everything a fire (92% of no-fire rows). ML removes ~65% of the false
positives but gives up ~1,400 fires; overall F1 is about the same (the models are slightly
*below* the rule's F1). At the SUSPECTED operating point below (85% recall) Logistic Regression
reaches precision 0.586 versus the rule's 0.511 - a real but modest gain. The models rank
candidates better than the rule's confidence (grouped AUC 0.68-0.74 vs 0.60).

## Hard negatives

Hard negatives = no-fire candidates with fire-looking evidence (3,347 rows, 74% of negatives).
False-positive rate on them: rule **1.000**; Logistic Regression 0.405, Random Forest 0.458,
Gradient Boosting 0.453 at threshold 0.5 (0.668 / 0.722 / 0.721 at the 85%-recall SUSPECTED
threshold). Predicted-fire rate per no-fire family at threshold 0.5:

| Family | Rule | LR | RF | GB |
| --- | --- | --- | --- | --- |
| N1 extreme danger | 0.90 | 0.62 | 0.60 | 0.62 |
| N2 high danger + rumours | 1.00 | 0.11 | 0.13 | 0.12 |
| N3 isolated low-conf hotspot | 0.34 | 0.24 | 0.69 | 0.52 |
| N4 weak/false news | 1.00 | 0.13 | 0.08 | 0.08 |
| N5 satellite + unrelated news | 1.00 | 0.16 | 0.20 | 0.18 |
| N6 multiple weak pieces | 1.00 | 0.11 | 0.27 | 0.27 |
| N7 no-fire, missing Fire Danger | 1.00 | 0.48 | 0.49 | 0.48 |
| **N8 strong-looking false positive** | 1.00 | **0.72** | 0.70 | 0.69 |
| **N9 persistent heat source** | 1.00 | **0.56** | 0.61 | 0.60 |
| N10 repeated false reports | 1.00 | 0.14 | 0.25 | 0.21 |

The models handle rumours and weak-news alarms well, but **N8, N9 and N1 are still flagged as
fire more than half the time** - these are the cases the dataset deliberately made
indistinguishable from real fires by their evidence.

## Weak positives

Weak positives = real fires with thin evidence (1,062 rows). Recall: rule 0.846; Logistic
Regression 0.372, Random Forest 0.419, Gradient Boosting 0.373 at threshold 0.5 (0.669 / 0.590
/ 0.643 at the SUSPECTED threshold). Per fire family (predicted-fire rate = recall):

| Family | Rule | LR | RF | GB |
| --- | --- | --- | --- | --- |
| P1 strong evidence | 1.00 | 0.99 | 0.96 | 0.94 |
| P2 satellite only | 0.98 | 0.52 | 0.63 | 0.46 |
| P3 news first | 1.00 | 0.55 | 0.46 | 0.48 |
| P4 moderate danger | 0.98 | 0.80 | 0.74 | 0.75 |
| P5 low FFWI | 0.98 | 0.74 | 0.72 | 0.65 |
| P6 missing Fire Danger | 0.98 | 0.74 | 0.76 | 0.78 |
| **P7 weak satellite measurements** | 0.73 | **0.01** | 0.02 | 0.01 |
| P8 spread over time | 0.99 | 0.80 | 0.75 | 0.78 |
| P9 persistent multi-satellite | 1.00 | 0.92 | 0.94 | 0.93 |
| P10 news-led multiple reports | 1.00 | 0.47 | 0.65 | 0.63 |

**P7 is essentially never detected**: its evidence (one or two weak, often unmeasured
hotspots) is statistically almost the same as N3's noise, so at threshold 0.5 the model calls
it noise. The ML models do not achieve their precision by "requiring multi-source evidence"
everywhere, but a whole class of genuine early/weak fires is lost.

## Fire Danger ablation

Grouped-family CV, full 19 features versus the first 16 (no Fire Danger), same folds, paired
differences (full minus reduced):

| Model | ROC-AUC full -> without FFWI | dAUC | folds improved | dF1 | dPrecision | dRecall | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.736 -> **0.764** | -0.028 | 1 of 5 | -0.020 | -0.025 | -0.020 | worse |
| Random Forest | 0.676 -> **0.699** | -0.023 | 1 of 5 | -0.019 | -0.017 | -0.020 | worse |
| Gradient Boosting | 0.702 -> **0.731** | -0.029 | 0 of 5 | -0.023 | -0.022 | -0.025 | worse |

**Fire Danger did not improve generalisation; removing it improved every model.** Families
hurt by Fire Danger in at least two of the three models: N1, N2, N3, N7 (no-fire, several under high/extreme danger) and P2, P5, P6.
The likely reason is the dataset itself: FFWI enters the benchmark as a *weak, family-level
prior* (fire families lean hotter, but N1/N2 are deliberately hot-and-no-fire and P5 is
cool-and-fire), so a model learns "hot => fire" from the families it sees and is wrong on the
held-out ones. The verdict rule was fixed in advance ("improved" needs +0.01 mean AUC and a gain
in 4 of 5 folds). Non-zero coefficients or importances are not evidence of value; here the
held-out permutation importance of the Fire Danger block is negative for all three models (about
-0.02 ROC-AUC): shuffling Fire Danger *improves* held-out AUC, i.e. the models' use of it hurts.

Other feature-group ablations (grouped ROC-AUC / F1, LR):

| Removed | ROC-AUC | F1 |
| --- | --- | --- |
| nothing (19) | 0.736 | 0.641 |
| Fire Danger (3) | 0.764 | 0.662 |
| geometry + time (2) | 0.756 | 0.653 |
| satellite FRP + brightness (6) | 0.711 | 0.612 |
| **news (5)** | **0.611** | 0.548 |

News is the most important family (the other two tree models show the same order); geometry/time
is dead weight - removing it improves the score slightly.

## Threshold analysis (out-of-fold grouped predictions)

Logistic Regression:

| Threshold | Precision | Recall | F1 | False-positive rate | False-negative rate |
| --- | --- | --- | --- | --- | --- |
| 0.40 | 0.636 | 0.735 | 0.682 | 0.420 | 0.265 |
| 0.50 | 0.666 | 0.653 | 0.660 | 0.327 | 0.347 |
| 0.60 | 0.703 | 0.562 | 0.625 | 0.238 | 0.438 |
| 0.65 | 0.719 | 0.508 | 0.596 | 0.198 | 0.492 |
| 0.70 | 0.740 | 0.451 | 0.560 | 0.159 | 0.549 |
| 0.75 | 0.757 | 0.380 | 0.506 | 0.122 | 0.620 |
| 0.80 | 0.774 | 0.310 | 0.443 | 0.090 | 0.690 |

(The report contains the same grid for all three models.)

### Recommended SUSPECTED / CONFIRMED thresholds (not implemented)

Targets fixed in advance: **SUSPECTED = the highest threshold that still reaches 85% recall**
(catch most real fires with as few alarms as possible); **CONFIRMED = the lowest threshold
above it that reaches 90% precision** (V3's target).

| Model | Suspect threshold | Precision / recall / FPR there | Confirm threshold |
| --- | --- | --- | --- |
| Logistic Regression | **0.25** | 0.586 / 0.850 / 0.601 | **not attainable** |
| Random Forest | 0.29 | 0.568 / 0.852 / 0.648 | not attainable |
| Gradient Boosting | 0.25 | 0.568 / 0.856 / 0.651 | not attainable |

**No threshold on any model reaches 90% precision out of fold.** The highest precision reachable
while keeping recall >= 10% is 0.82 (LR, threshold 0.91, recall 0.13), 0.75 (RF) and 0.85 (GB,
recall 0.12). The trade-off is stark: a SUSPECTED threshold that catches 85% of fires flags 60%
of no-fire candidates, and a CONFIRMED threshold that is even close to reliable confirms only
a small fraction of real fires. So **a CONFIRMED level with "significantly stronger evidence"
cannot be defined from V4 model probabilities**, and these numbers must not be wired in as
runtime thresholds. Keeping the rule's evidence-based CONFIRMED (satellite + news
corroboration) remains the safer definition.

## Calibration (Logistic Regression, out of fold)

Brier 0.222 versus 0.250 for always predicting the base rate (skill 0.11); expected calibration
error 0.097. Reliability is poor at the extremes: predictions of 0.94 correspond to 81%
observed fire; predictions of 0.05 to 13.5%. (Random Forest: Brier 0.244, ECE 0.102; Gradient
Boosting: 0.237, 0.120.) The probabilities are over-confident on families the model has not
seen, so an "AI Confidence 0.85" would really mean about 74%.

`CalibratedClassifierCV` (nested, grouped, sigmoid and isotonic) was evaluated: it lowers ECE
(LR 0.097 -> 0.040-0.043) but does **not** improve Brier (0.222 -> 0.230-0.232) and costs about
0.026 grouped ROC-AUC, so by the rule fixed in advance (Brier gain >= 0.005 and AUC drop <=
0.005) **calibration is not recommended** and was not applied.

## Feature importance

Logistic Regression standardized coefficients (full fit; coefficients of correlated pairs
must be read together):

| Feature | Coef | Feature | Coef |
| --- | --- | --- | --- |
| satellite_frp_max | +3.080 | satellite_low_count | -0.655 |
| satellite_frp_mean | -2.367 | news_weak_count | -0.492 |
| satellite_brightness_max | +1.344 | news_strong_count | +0.361 |
| satellite_brightness_mean | -1.303 | fire_danger_score | +0.083 |
| news_none_count | -1.187 | fire_danger_available / age | ~0 |

Max/mean FRP and brightness carry large opposite-sign coefficients (collinearity) - the
individual values are unstable, only their combination is meaningful. Held-out permutation
importance (drop in grouped ROC-AUC): LR - `satellite_frp_max` 0.274 (43% of total), then
`news_none_count`, `satellite_brightness_max`; Random Forest / Gradient Boosting - `news_none_count`
47-61% of total. **Shortcut flags:** the models lean heavily on one feature each (FRP max for LR,
`news_none_count` for the trees), which fails the gate's 40% single-feature limit; this mirrors how
the synthetic families were parameterised (FRP ranges and news-signal mixes differ by family).
Geometry/time features (`max_pairwise_distance_km`, `time_span_minutes`) and the Fire Danger block
carry no positive held-out importance (shuffling them slightly improves AUC). Random-Forest impurity importances are in the report but are biased
towards continuous features and are not used for conclusions.

## Generalisation gate (fixed before any model was evaluated; not tuned)

| Criterion | Threshold | Logistic Regression |
| --- | --- | --- |
| Grouped mean ROC-AUC | >= 0.80 | 0.736 - **fail** |
| Worst grouped-fold ROC-AUC | >= 0.70 | 0.534 - **fail** |
| Grouped mean F1 minus rule's | >= +0.03 | -0.027 - **fail** |
| Recall at SUSPECTED threshold | >= 0.85 | 0.850 - pass |
| Precision at SUSPECTED minus rule's | >= +0.08 | +0.075 - **fail** (narrow) |
| Hard-negative FPR at SUSPECTED threshold | <= 0.60 | 0.668 - **fail** |
| Weak-positive recall at SUSPECTED threshold | >= 0.60 | 0.669 - pass |
| Archetype mean ROC-AUC | >= 0.75 | 0.778 - pass |
| Worst archetype ROC-AUC | >= 0.65 | 0.703 - pass |
| Calibration ECE | <= 0.10 | 0.097 - pass (narrow) |
| Largest single-feature importance share | <= 0.40 | 0.426 - **fail** (narrow) |
| Geometry/time importance share | <= 0.25 | 0.000 - pass |

Random Forest fails 8 criteria and Gradient Boosting 7 (see the JSON). Selection rule if any
model had passed: highest grouped ROC-AUC among passing models, with near-ties (within 0.01 AUC and
0.02 F1) going to the simplest (LR < RF < GB).

## Selected model

**None.** No model passes the gate; the code refuses to save an artifact
(`save_model_artifact_v4` and `scripts/train_fire_detection_model_v4.py` both refuse when the
report selected nothing; tests exercise saving only in temp directories). Logistic Regression V4
is the best of the three and the one to build on, but it is not runtime-ready.

## Limitations

* Synthetic data with deliberately overlapping families (e.g. N8 vs P1, N3 vs P7). Part of the
  remaining error may be irreducible for this benchmark; measured in-distribution (random-split)
  AUC is only 0.85-0.89.
* The FFWI-label association in the benchmark is a designed weak prior; the finding that Fire Danger
  does not help says the *benchmark's* Fire Danger signal does not transfer across families, not that
  Fire Danger is useless in the real world.
* High fold-to-fold variance (5 folds, 4 families each); the frozen fold layout can hold out every
  family of one evidence shape.
* Hyperparameters were fixed a priori; only a small sensitivity check was run.
* No real labelled wildfire data was used; none of these metrics validate operational performance.

## Suggested next steps (not done here)

Do not proceed to a V4 primary-scorer integration on these results. Options for the team to
weigh: keep V3/shadow while gathering real labelled evidence; treat V4-without-Fire-Danger as the
candidate to re-evaluate; revisit the synthetic benchmark's design (an explicit decision, since the
dataset is frozen for this task) - for example whether N8/P1 and N3/P7 should be separable at all;
and keep the rule's evidence-based CONFIRMED definition. Re-running the gate after any such change
is a single command.
