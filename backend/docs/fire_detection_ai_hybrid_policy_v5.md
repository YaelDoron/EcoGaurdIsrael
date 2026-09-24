# Fire Detection — AI Hybrid Policy: Confirmatory Evaluation (Task 8)

> **All training and confirmatory evaluation data is SYNTHETIC. These results describe the AI Hybrid policy on the V5 benchmark generator and are NOT real-world wildfire detection accuracy.**

## Verdict: **AI HYBRID POLICY PASSED**

Task 7 binary-primary V5 gate: **FAILED** (unchanged, historical). Task 8 evaluates a different architecture: a calibrated HGB likelihood estimator + an explicit uncertainty state (SUSPECTED) + one deterministic corroboration guardrail. It is not presented as if Task 7 had passed.

## Locked model and policy (fixed before evaluation)

- Model: **HistGradientBoostingClassifier** `{'learning_rate': 0.1, 'max_depth': 3}`, the exact ordered V5 feature contract (25 features); preprocessing: none (HistGradientBoostingClassifier handles NaN natively). Trained once on `training_v5.csv` (10000 rows); confirmatory rows used in training: 0.
- Policy `ai_hybrid_policy_v5.0`: suspect threshold 0.4, confirm threshold 0.8, corroboration: satellite_low_count + satellite_nominal_count + satellite_high_count >= 2 (hotspots of the current candidate).
  - **CONFIRMED**: P(fire) >= 0.8 AND current satellite pixel count >= 2
  - **NO_EVENT**: P(fire) < 0.4
  - **SUSPECTED**: P(fire) >= 0.4 and the CONFIRMED condition is not satisfied

## Confirmatory dataset

| item | value |
|---|---|
| seed (training seed 42) | 8202609 |
| rows | 10000 |
| sha256 | `7c24405e83fdceb1c4f74801da97d80aae84ea46161654c282df30ea190d8ead` |
| environments / pairs | 250 / 2500 |
| shared environment ids / sample ids / pair ids with training | 0 / 0 / 0 |
| exact feature-vector overlap with training (diagnostic) | 811 rows (8.1%); by regime {'news_led': 535, 'satellite_only_single_pass': 51, 'sparse_early_evidence': 225} |

## Policy acceptance gate (fixed before evaluation)

| criterion | value | requirement | passed |
|---|---|---|---|
| brier_score | 0.2088 | < 0.25 | yes |
| confirmed_precision | 0.9367 | >= 0.9 | yes |
| confirmed_recall_non_sparse | 0.1611 | >= 0.1 | yes |
| expected_calibration_error | 0.0174 | <= 0.1 | yes |
| false_confirmation_rate | 0.0098 | <= 0.1 | yes |
| fire_retained_recall_non_sparse | 0.8176 | >= 0.8 | yes |
| non_sparse_no_fire_alert_rate_vs_rules | 0.5016 | <= 0.75 x the rule detector's 0.794 = 0.596 | yes |
| persistent_thermal_roc_auc | 0.7791 | >= 0.7 | yes |
| sparse_confirmed_rate | 0.0000 | <= 0.05 | yes |

Failed criteria: none.

### How robust are the margins? (descriptive; the verdict uses the point estimates)

Environment-clustered bootstrap (whole environments resampled). DESCRIPTIVE only: the verdict comes from the point estimates against the predefined gate; this shows how robust each margin is on one synthetic draw.

| metric | point estimate | 95% interval | gate limit | whole interval clears limit |
|---|---|---|---|---|
| brier_score | 0.2088 | [0.2058, 0.2120] | < 0.25 | yes |
| confirmed_precision | 0.9367 | [0.9178, 0.9529] | >= 0.9 | yes |
| confirmed_recall_non_sparse | 0.1611 | [0.1499, 0.1718] | >= 0.1 | yes |
| expected_calibration_error | 0.0174 | [0.0124, 0.0270] | <= 0.1 | yes |
| false_confirmation_rate | 0.0098 | [0.0072, 0.0129] | <= 0.1 | yes |
| fire_retained_recall_non_sparse | 0.8176 | [0.8061, 0.8293] | >= 0.8 | yes |
| non_sparse_no_fire_alert_rate_ratio_vs_rules | n/a | [0.6105, 0.6526] | <= 0.75 | yes |
| persistent_thermal_roc_auc | 0.7791 | [0.7638, 0.7942] | >= 0.7 | yes |
| sparse_confirmed_rate | 0.0000 | [0.0000, 0.0000] | <= 0.05 | yes |

## 3-state outcome matrix

### All rows

| ground truth | rows | NO_EVENT | SUSPECTED | CONFIRMED |
|---|---|---|---|---|
| FIRE | 5000 | 939 (18.8%) | 3336 (66.7%) | 725 (14.5%) |
| NO_FIRE | 5000 | 2423 (48.5%) | 2528 (50.6%) | 49 (1.0%) |

### Non-sparse rows

| ground truth | rows | NO_EVENT | SUSPECTED | CONFIRMED |
|---|---|---|---|---|
| FIRE | 4500 | 821 (18.2%) | 2954 (65.6%) | 725 (16.1%) |
| NO_FIRE | 4500 | 2243 (49.8%) | 2208 (49.1%) | 49 (1.1%) |

A false SUSPECTED is operationally far milder than a false CONFIRMED.

## Per-regime outcomes

| regime | truth | rows | NO_EVENT | SUSPECTED | CONFIRMED |
|---|---|---|---|---|---|
| multi_pixel_single_pass | FIRE | 800 | 162 (20.2%) | 431 (53.9%) | 207 (25.9%) |
| multi_pixel_single_pass | NO_FIRE | 800 | 406 (50.7%) | 382 (47.8%) | 12 (1.5%) |
| news_led | FIRE | 700 | 140 (20.0%) | 499 (71.3%) | 61 (8.7%) |
| news_led | NO_FIRE | 700 | 352 (50.3%) | 346 (49.4%) | 2 (0.3%) |
| persistent_thermal | FIRE | 1300 | 218 (16.8%) | 811 (62.4%) | 271 (20.8%) |
| persistent_thermal | NO_FIRE | 1300 | 726 (55.8%) | 554 (42.6%) | 20 (1.5%) |
| satellite_news | FIRE | 900 | 159 (17.7%) | 555 (61.7%) | 186 (20.7%) |
| satellite_news | NO_FIRE | 900 | 465 (51.7%) | 420 (46.7%) | 15 (1.7%) |
| satellite_only_single_pass | FIRE | 800 | 142 (17.8%) | 658 (82.2%) | 0 (0.0%) |
| satellite_only_single_pass | NO_FIRE | 800 | 294 (36.8%) | 506 (63.2%) | 0 (0.0%) |
| sparse_early_evidence | FIRE | 500 | 118 (23.6%) | 382 (76.4%) | 0 (0.0%) |
| sparse_early_evidence | NO_FIRE | 500 | 180 (36.0%) | 320 (64.0%) | 0 (0.0%) |

## Per-latent-subtype outcomes

| latent subtype | truth | rows | NO_EVENT | SUSPECTED | CONFIRMED | mean P(fire) |
|---|---|---|---|---|---|---|
| controlled_or_agricultural_burn | NO_FIRE | 1020 | 321 (31.5%) | 678 (66.5%) | 21 (2.1%) | 0.491 |
| early_wildfire | FIRE | 2522 | 648 (25.7%) | 1847 (73.2%) | 27 (1.1%) | 0.499 |
| established_wildfire | FIRE | 1601 | 247 (15.4%) | 1158 (72.3%) | 196 (12.2%) | 0.591 |
| false_or_rumour_report | NO_FIRE | 814 | 411 (50.5%) | 399 (49.0%) | 4 (0.5%) | 0.406 |
| industrial_heat_source | NO_FIRE | 2029 | 1130 (55.7%) | 880 (43.4%) | 19 (0.9%) | 0.382 |
| large_wildfire | FIRE | 877 | 44 (5.0%) | 331 (37.7%) | 502 (57.2%) | 0.792 |
| sensor_noise | NO_FIRE | 1137 | 561 (49.3%) | 571 (50.2%) | 5 (0.4%) | 0.417 |

False-confirmation rate by no-fire subtype: controlled_or_agricultural_burn 2.1%, false_or_rumour_report 0.5%, industrial_heat_source 0.9%, sensor_noise 0.4%.

## Sparse early evidence

Brier 0.2465 (constant 0.5 = 0.25), ECE 0.0558, mean P 0.459; between 0.35–0.65: 83.1%; P<0.10: 0.0%; P>0.90: 0.0%; status rates NO_EVENT 29.8% / SUSPECTED 70.2% / CONFIRMED 0.0%.

Probability histogram (10 bins 0–1): [0, 3, 53, 242, 408, 86, 208, 0, 0, 0]

| ground truth | rows | NO_EVENT | SUSPECTED | CONFIRMED |
|---|---|---|---|---|
| FIRE | 500 | 118 (23.6%) | 382 (76.4%) | 0 (0.0%) |
| NO_FIRE | 500 | 180 (36.0%) | 320 (64.0%) | 0 (0.0%) |

## Persistent / history behaviour by satellite_pass_count

A STATIC dataset: each row is one independent observation, so this compares rows with different pass counts; it does NOT prove that one event's confidence evolves over time. Sequential runtime progression is a later test.

| pass count | rows | fire rows | mean P | median P | NO_EVENT | SUSPECTED | CONFIRMED | alert precision | alert recall | confirmed precision | confirmed recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (no satellite) | 750 | 379 | 0.507 | 0.488 | 22.1% | 77.9% | 0.0% | 0.551 | 0.850 | n/a | 0.000 |
| 1 | 6650 | 3321 | 0.499 | 0.452 | 33.9% | 58.9% | 7.3% | 0.604 | 0.800 | 0.940 | 0.137 |
| 2 | 1511 | 795 | 0.511 | 0.508 | 30.3% | 59.8% | 9.9% | 0.638 | 0.845 | 0.913 | 0.172 |
| >=3 | 1089 | 505 | 0.468 | 0.449 | 44.6% | 42.4% | 12.9% | 0.680 | 0.812 | 0.950 | 0.265 |

Inside `persistent_thermal` only (both labels present):

| pass count | rows | mean P (fire rows) | mean P (no-fire rows) | confirmed rate (fire) | confirmed rate (no-fire) | ROC-AUC |
|---|---|---|---|---|---|---|
| 2 | 1511 | 0.600 | 0.413 | 17.2% | 1.8% | 0.739 |
| >=3 | 1089 | 0.633 | 0.326 | 26.5% | 1.2% | 0.823 |

## Rule detector vs AI Hybrid policy (same confirmatory rows)

rule detector: SUSPECTED or CONFIRMED = fire (alert), CONFIRMED-only reported separately; AI Hybrid policy: status != NO_EVENT = alert. Same confirmatory rows; non-sparse rows for the slices.

### Non-sparse rows: alert = SUSPECTED or CONFIRMED

| system | precision | recall (fire retained) | F1 | false positives | false negatives | no-fire alert rate | hard-negative FPR |
|---|---|---|---|---|---|---|---|
| rule detector | 0.522 | 0.868 | 0.652 | 3575 | 593 | 0.794 | 1.000 |
| AI hybrid policy | 0.620 | 0.818 | 0.705 | 2257 | 821 | 0.502 | 0.582 |

### CONFIRMED only

| system | confirmed precision (all rows) | false-confirmed rows | false-confirmation rate | non-sparse confirmed recall |
|---|---|---|---|---|
| rule detector | 0.588 | 942 | 18.8% | 0.298 |
| AI hybrid policy | 0.937 | 49 | 1.0% | 0.161 |

Relative reduction of the non-sparse no-fire alert rate versus the rules: 36.9%.

### Per regime alert behaviour (rule vs policy)

| regime | rule recall | policy recall | rule no-fire alert rate | policy no-fire alert rate |
|---|---|---|---|---|
| multi_pixel_single_pass | 0.830 | 0.797 | 0.710 | 0.492 |
| news_led | 1.000 | 0.800 | 1.000 | 0.497 |
| persistent_thermal | 0.848 | 0.832 | 0.808 | 0.442 |
| satellite_news | 1.000 | 0.823 | 1.000 | 0.483 |
| satellite_only_single_pass | 0.675 | 0.823 | 0.446 | 0.632 |
| sparse_early_evidence | 0.518 | 0.764 | 0.392 | 0.640 |

## Calibration and ranking on the confirmatory data

ROC-AUC 0.728, PR-AUC 0.739, Brier 0.2088, ECE 0.0174, persistent_thermal ROC-AUC 0.779.

| bin | count | mean predicted | observed fire fraction |
|---|---|---|---|
| 0.0-0.1 | 173 | 0.068 | 0.040 |
| 0.1-0.2 | 432 | 0.156 | 0.171 |
| 0.2-0.3 | 866 | 0.252 | 0.263 |
| 0.3-0.4 | 1891 | 0.356 | 0.333 |
| 0.4-0.5 | 2070 | 0.444 | 0.470 |
| 0.5-0.6 | 1516 | 0.554 | 0.559 |
| 0.6-0.7 | 1627 | 0.644 | 0.626 |
| 0.7-0.8 | 570 | 0.745 | 0.758 |
| 0.8-0.9 | 371 | 0.848 | 0.871 |
| 0.9-1.0 | 484 | 0.956 | 0.965 |

## Limitations

- All training and confirmatory evaluation data is synthetic.
- Static data cannot prove sequential progression of one event.
- The policy is offline: not integrated into runtime; SUSPECTED currently triggers the full response pipeline (see the runtime-integration audit).
