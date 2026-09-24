# Fire Detection ML — V5 Model Comparison (Task 7)

> **All training and evaluation data is SYNTHETIC (backend/docs/fire_detection_dataset_v5.md). These metrics describe behaviour on the frozen V5 benchmark and are NOT real-world wildfire detection accuracy.**

## Verdict

**Selected model: NONE.** no model passed the predefined acceptance gate; no runtime artifact is saved

| model | gate | failed criteria |
|---|---|---|
| hist_gradient_boosting | FAIL | grouped_environment_mean_roc_auc, hard_negative_fpr |
| logistic_regression | FAIL | grouped_environment_mean_roc_auc, hard_negative_fpr |
| random_forest | FAIL | grouped_environment_mean_roc_auc, hard_negative_fpr |

Dataset `training_v5` sha256 `37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035` (10000 rows, 25 canonical features), seed 42; Python 3.9.6, scikit-learn 1.6.1, numpy 2.0.2, joblib 1.5.3.

## Key findings

- **hist_gradient_boosting**: grouped-environment ROC-AUC 0.728 (worst fold 0.715); gate FAILED (grouped_environment_mean_roc_auc, hard_negative_fpr).
- **logistic_regression**: grouped-environment ROC-AUC 0.693 (worst fold 0.676); gate FAILED (grouped_environment_mean_roc_auc, hard_negative_fpr).
- **random_forest**: grouped-environment ROC-AUC 0.721 (worst fold 0.708); gate FAILED (grouped_environment_mean_roc_auc, hard_negative_fpr).
- Best-ranked model by grouped ROC-AUC: **hist_gradient_boosting**.
- hist_gradient_boosting: random-split ROC-AUC 0.706 vs grouped 0.728 (gap -0.022; DIAGNOSTIC ONLY).
- logistic_regression: random-split ROC-AUC 0.684 vs grouped 0.693 (gap -0.009; DIAGNOSTIC ONLY).
- random_forest: random-split ROC-AUC 0.694 vs grouped 0.721 (gap -0.027; DIAGNOSTIC ONLY).
- History block, hist_gradient_boosting: improved persistent_thermal ROC-AUC (full minus without-history +0.0626, environment-bootstrap 95% CI [+0.0497, +0.0755]); overall grouped difference +0.0189.
- History block, logistic_regression: improved persistent_thermal ROC-AUC (full minus without-history +0.0494, environment-bootstrap 95% CI [+0.0360, +0.0629]); overall grouped difference +0.0135.
- History block, random_forest: improved persistent_thermal ROC-AUC (full minus without-history +0.0483, environment-bootstrap 95% CI [+0.0360, +0.0596]); overall grouped difference +0.0138.
- Versus rules (hist_gradient_boosting, non-sparse rows, at its SUSPECTED threshold 0.40): hard-negative FPR 0.583 vs rules 1.000 (41.7% reduction); recall 0.815 vs 0.878 (-0.063); precision 0.614 vs 0.525; F1 0.700 vs 0.657.
- Versus rules (logistic_regression, non-sparse rows, at its SUSPECTED threshold 0.38): hard-negative FPR 0.683 vs rules 1.000 (31.7% reduction); recall 0.831 vs 0.878 (-0.047); precision 0.574 vs 0.525; F1 0.679 vs 0.657.
- Versus rules (random_forest, non-sparse rows, at its SUSPECTED threshold 0.41): hard-negative FPR 0.626 vs rules 1.000 (37.4% reduction); recall 0.818 vs 0.878 (-0.060); precision 0.606 vs 0.525; F1 0.696 vs 0.657.
- hist_gradient_boosting: probability-only CONFIRMED (precision >= 0.90 at recall >= 0.10) is attainable.
- logistic_regression: probability-only CONFIRMED (precision >= 0.90 at recall >= 0.10) is attainable.
- random_forest: probability-only CONFIRMED (precision >= 0.90 at recall >= 0.10) is attainable.
- Selected model: **NONE** - no artifact saved.

## Acceptance gate (fixed before evaluation)

| criterion | value |
|---|---|
| brier_score_max_exclusive | 0.25 |
| expected_calibration_error_max | 0.1 |
| grouped_environment_mean_roc_auc_min | 0.75 |
| grouped_environment_worst_fold_roc_auc_min | 0.65 |
| hard_negative_fpr_max | 0.45 |
| non_sparse_positive_recall_min | 0.8 |
| persistent_thermal_roc_auc_min | 0.7 |
| top_feature_importance_share_max | 0.4 |
| top_feature_importance_share_tolerance | 0.02 |
| vs_rules_f1_min_delta | 0.0 |
| vs_rules_hard_negative_fpr_max_ratio | 0.75 |
| vs_rules_recall_max_drop | 0.15 |

Primary evaluation: grouped by environment_id (5 folds; whole environments and all pair members held out together).

## Gate results per model

### hist_gradient_boosting

| criterion | value | requirement | passed |
|---|---|---|---|
| beats_rules_f1_not_worse | 0.7000 | >= rule F1 (0.657) + 0.0 | yes |
| beats_rules_hard_negative_fpr | 0.5826 | <= 0.75 x rule hard-negative FPR (1.000) = 0.750 | yes |
| beats_rules_recall_not_collapsed | 0.8147 | >= rule recall (0.878) - 0.15 = 0.728 | yes |
| brier_score | 0.2093 | < 0.25 | yes |
| expected_calibration_error | 0.0073 | <= 0.1 | yes |
| grouped_environment_mean_roc_auc | 0.7284 | >= 0.75 | NO |
| grouped_environment_worst_fold_roc_auc | 0.7151 | >= 0.65 | yes |
| hard_negative_fpr | 0.5826 | <= 0.45 at the SUSPECTED operating point (sparse excluded) | NO |
| non_sparse_positive_recall | 0.8147 | >= 0.8 at the SUSPECTED operating point | yes |
| persistent_thermal_roc_auc | 0.7713 | >= 0.7 | yes |
| top_feature_importance_share | 0.2234 | <= 0.4 (+0.02 tolerance) | yes |

### logistic_regression

| criterion | value | requirement | passed |
|---|---|---|---|
| beats_rules_f1_not_worse | 0.6791 | >= rule F1 (0.657) + 0.0 | yes |
| beats_rules_hard_negative_fpr | 0.6831 | <= 0.75 x rule hard-negative FPR (1.000) = 0.750 | yes |
| beats_rules_recall_not_collapsed | 0.8307 | >= rule recall (0.878) - 0.15 = 0.728 | yes |
| brier_score | 0.2200 | < 0.25 | yes |
| expected_calibration_error | 0.0052 | <= 0.1 | yes |
| grouped_environment_mean_roc_auc | 0.6927 | >= 0.75 | NO |
| grouped_environment_worst_fold_roc_auc | 0.6764 | >= 0.65 | yes |
| hard_negative_fpr | 0.6831 | <= 0.45 at the SUSPECTED operating point (sparse excluded) | NO |
| non_sparse_positive_recall | 0.8307 | >= 0.8 at the SUSPECTED operating point | yes |
| persistent_thermal_roc_auc | 0.7185 | >= 0.7 | yes |
| top_feature_importance_share | 0.2812 | <= 0.4 (+0.02 tolerance) | yes |

### random_forest

| criterion | value | requirement | passed |
|---|---|---|---|
| beats_rules_f1_not_worse | 0.6963 | >= rule F1 (0.657) + 0.0 | yes |
| beats_rules_hard_negative_fpr | 0.6264 | <= 0.75 x rule hard-negative FPR (1.000) = 0.750 | yes |
| beats_rules_recall_not_collapsed | 0.8180 | >= rule recall (0.878) - 0.15 = 0.728 | yes |
| brier_score | 0.2117 | < 0.25 | yes |
| expected_calibration_error | 0.0051 | <= 0.1 | yes |
| grouped_environment_mean_roc_auc | 0.7214 | >= 0.75 | NO |
| grouped_environment_worst_fold_roc_auc | 0.7080 | >= 0.65 | yes |
| hard_negative_fpr | 0.6264 | <= 0.45 at the SUSPECTED operating point (sparse excluded) | NO |
| non_sparse_positive_recall | 0.8180 | >= 0.8 at the SUSPECTED operating point | yes |
| persistent_thermal_roc_auc | 0.7522 | >= 0.7 | yes |
| top_feature_importance_share | 0.2844 | <= 0.4 (+0.02 tolerance) | yes |

## Preprocessing and hyperparameter grid

**hist_gradient_boosting** — none (HistGradientBoostingClassifier handles NaN natively); scaling: none; missing indicators: none. Fixed: `{'early_stopping': False, 'l2_regularization': 1.0, 'max_iter': 200, 'min_samples_leaf': 40}`. Selected by highest grouped-environment mean ROC-AUC; ties keep the earlier (simpler) setting; no stress test used: `{'learning_rate': 0.1, 'max_depth': 3}`.

| params | grouped ROC-AUC mean | worst fold | Brier |
|---|---|---|---|
| {'learning_rate': 0.05, 'max_depth': 2} | 0.7168 | 0.7010 | 0.2135 |
| {'learning_rate': 0.05, 'max_depth': 3} | 0.7260 | 0.7115 | 0.2102 |
| {'learning_rate': 0.1, 'max_depth': 3} | 0.7284 | 0.7151 | 0.2093 |
| {'learning_rate': 0.05, 'max_depth': 4} | 0.7276 | 0.7133 | 0.2095 |

**logistic_regression** — median (fit on training folds only) for the nullable features; scaling: StandardScaler after imputation; missing indicators: none added. Fixed: `{'max_iter': 2000, 'penalty': 'l2'}`. Selected by highest grouped-environment mean ROC-AUC; ties keep the earlier (simpler) setting; no stress test used: `{'C': 0.1}`.

| params | grouped ROC-AUC mean | worst fold | Brier |
|---|---|---|---|
| {'C': 0.01} | 0.6913 | 0.6748 | 0.2207 |
| {'C': 0.1} | 0.6924 | 0.6753 | 0.2205 |
| {'C': 1.0} | 0.6923 | 0.6752 | 0.2205 |
| {'C': 10.0} | 0.6920 | 0.6748 | 0.2205 |

**random_forest** — median (fit on training folds only) for the nullable features; scaling: none (tree model); missing indicators: none added. Fixed: `{'max_features': 'sqrt', 'n_estimators': 300, 'n_jobs': 1}`. Selected by highest grouped-environment mean ROC-AUC; ties keep the earlier (simpler) setting; no stress test used: `{'max_depth': 12, 'min_samples_leaf': 10}`.

| params | grouped ROC-AUC mean | worst fold | Brier |
|---|---|---|---|
| {'max_depth': 4, 'min_samples_leaf': 20} | 0.6992 | 0.6832 | 0.2237 |
| {'max_depth': 6, 'min_samples_leaf': 20} | 0.7125 | 0.6958 | 0.2183 |
| {'max_depth': 8, 'min_samples_leaf': 10} | 0.7188 | 0.7037 | 0.2149 |
| {'max_depth': 12, 'min_samples_leaf': 10} | 0.7209 | 0.7086 | 0.2128 |

## Grouped-environment results (PRIMARY)

### hist_gradient_boosting (threshold 0.50)

| fold | ROC-AUC | PR-AUC | accuracy | precision | recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| environment_fold_0 | 0.735 | 0.740 | 0.671 | 0.701 | 0.607 | 0.650 | 0.207 |
| environment_fold_1 | 0.715 | 0.728 | 0.657 | 0.658 | 0.637 | 0.648 | 0.213 |
| environment_fold_2 | 0.745 | 0.754 | 0.670 | 0.677 | 0.648 | 0.663 | 0.204 |
| environment_fold_3 | 0.718 | 0.723 | 0.659 | 0.681 | 0.601 | 0.638 | 0.213 |
| environment_fold_4 | 0.729 | 0.741 | 0.666 | 0.673 | 0.647 | 0.660 | 0.210 |
| mean | 0.728 | 0.737 | 0.665 | 0.678 | 0.628 | 0.652 | 0.209 |
| std | 0.011 | 0.011 | 0.006 | 0.014 | 0.020 | 0.009 | 0.003 |
| min | 0.715 | 0.723 | 0.657 | 0.658 | 0.601 | 0.638 | 0.204 |
| max | 0.745 | 0.754 | 0.671 | 0.701 | 0.648 | 0.663 | 0.213 |

### logistic_regression (threshold 0.50)

| fold | ROC-AUC | PR-AUC | accuracy | precision | recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| environment_fold_0 | 0.702 | 0.712 | 0.651 | 0.667 | 0.614 | 0.640 | 0.217 |
| environment_fold_1 | 0.676 | 0.681 | 0.626 | 0.625 | 0.608 | 0.616 | 0.225 |
| environment_fold_2 | 0.698 | 0.704 | 0.644 | 0.647 | 0.633 | 0.640 | 0.218 |
| environment_fold_3 | 0.692 | 0.706 | 0.643 | 0.663 | 0.587 | 0.622 | 0.220 |
| environment_fold_4 | 0.696 | 0.702 | 0.645 | 0.641 | 0.657 | 0.649 | 0.220 |
| mean | 0.693 | 0.701 | 0.642 | 0.648 | 0.620 | 0.633 | 0.220 |
| std | 0.009 | 0.011 | 0.008 | 0.015 | 0.024 | 0.012 | 0.003 |
| min | 0.676 | 0.681 | 0.626 | 0.625 | 0.587 | 0.616 | 0.217 |
| max | 0.702 | 0.712 | 0.651 | 0.667 | 0.657 | 0.649 | 0.225 |

### random_forest (threshold 0.50)

| fold | ROC-AUC | PR-AUC | accuracy | precision | recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| environment_fold_0 | 0.728 | 0.733 | 0.666 | 0.690 | 0.611 | 0.648 | 0.210 |
| environment_fold_1 | 0.708 | 0.712 | 0.644 | 0.640 | 0.642 | 0.641 | 0.216 |
| environment_fold_2 | 0.738 | 0.745 | 0.662 | 0.672 | 0.635 | 0.653 | 0.206 |
| environment_fold_3 | 0.711 | 0.716 | 0.655 | 0.673 | 0.606 | 0.638 | 0.215 |
| environment_fold_4 | 0.722 | 0.737 | 0.655 | 0.667 | 0.619 | 0.642 | 0.211 |
| mean | 0.721 | 0.728 | 0.657 | 0.668 | 0.623 | 0.644 | 0.212 |
| std | 0.011 | 0.013 | 0.008 | 0.016 | 0.014 | 0.005 | 0.004 |
| min | 0.708 | 0.712 | 0.644 | 0.640 | 0.606 | 0.638 | 0.206 |
| max | 0.738 | 0.745 | 0.666 | 0.690 | 0.642 | 0.653 | 0.216 |

## Random split — DIAGNOSTIC ONLY

| model | random ROC-AUC | grouped-env ROC-AUC | gap | environments in both sides |
|---|---|---|---|---|
| hist_gradient_boosting | 0.706 | 0.728 | -0.0224 | 250 |
| logistic_regression | 0.684 | 0.693 | -0.0086 | 250 |
| random_forest | 0.694 | 0.721 | -0.0271 | 250 |

Never used to choose a model.

## Leave-one-regime-out (stress test)

`persistent_thermal` is the only regime with history: holding it out leaves the training set with no multi-pass examples. That row is a severe out-of-distribution stress test, not a hard criterion.

### hist_gradient_boosting

| held-out regime | ROC-AUC | precision@0.5 | recall@0.5 | F1@0.5 | Brier |
|---|---|---|---|---|---|
| multi_pixel_single_pass | 0.719 | 0.691 | 0.578 | 0.629 | 0.209 |
| news_led | 0.701 | 0.604 | 0.727 | 0.660 | 0.221 |
| persistent_thermal | 0.699 | 0.645 | 0.624 | 0.634 | 0.218 |
| satellite_news | 0.723 | 0.667 | 0.622 | 0.644 | 0.211 |
| satellite_only_single_pass | 0.663 | 0.644 | 0.641 | 0.642 | 0.231 |
| sparse_early_evidence | 0.597 | 0.634 | 0.360 | 0.459 | 0.248 |

### logistic_regression

| held-out regime | ROC-AUC | precision@0.5 | recall@0.5 | F1@0.5 | Brier |
|---|---|---|---|---|---|
| multi_pixel_single_pass | 0.699 | 0.632 | 0.642 | 0.637 | 0.218 |
| news_led | 0.698 | 0.623 | 0.694 | 0.657 | 0.222 |
| persistent_thermal | 0.661 | 0.623 | 0.598 | 0.611 | 0.228 |
| satellite_news | 0.719 | 0.643 | 0.694 | 0.668 | 0.214 |
| satellite_only_single_pass | 0.642 | 0.659 | 0.490 | 0.562 | 0.240 |
| sparse_early_evidence | 0.585 | 0.625 | 0.150 | 0.242 | 0.252 |

### random_forest

| held-out regime | ROC-AUC | precision@0.5 | recall@0.5 | F1@0.5 | Brier |
|---|---|---|---|---|---|
| multi_pixel_single_pass | 0.723 | 0.709 | 0.573 | 0.633 | 0.209 |
| news_led | 0.672 | 0.598 | 0.743 | 0.663 | 0.225 |
| persistent_thermal | 0.689 | 0.654 | 0.610 | 0.631 | 0.221 |
| satellite_news | 0.729 | 0.678 | 0.610 | 0.642 | 0.209 |
| satellite_only_single_pass | 0.661 | 0.637 | 0.655 | 0.646 | 0.230 |
| sparse_early_evidence | 0.591 | 0.630 | 0.330 | 0.433 | 0.258 |

## Leave-one-no-fire-subtype-out

### hist_gradient_boosting

| held-out subtype | rows | mean P(fire) | FPR@0.40 | FPR@0.50 | high-conf FPR (P>=0.8) |
|---|---|---|---|---|---|
| controlled_or_agricultural_burn | 962 | 0.560 | 0.806 | 0.581 | 0.109 |
| false_or_rumour_report | 786 | 0.627 | 0.836 | 0.682 | 0.370 |
| industrial_heat_source | 2159 | 0.730 | 0.969 | 0.915 | 0.378 |
| sensor_noise | 1093 | 0.624 | 0.898 | 0.830 | 0.035 |

### logistic_regression

| held-out subtype | rows | mean P(fire) | FPR@0.38 | FPR@0.50 | high-conf FPR (P>=0.8) |
|---|---|---|---|---|---|
| controlled_or_agricultural_burn | 962 | 0.540 | 0.864 | 0.571 | 0.062 |
| false_or_rumour_report | 786 | 0.604 | 0.898 | 0.686 | 0.177 |
| industrial_heat_source | 2159 | 0.704 | 0.960 | 0.800 | 0.369 |
| sensor_noise | 1093 | 0.567 | 0.928 | 0.711 | 0.026 |

### random_forest

| held-out subtype | rows | mean P(fire) | FPR@0.41 | FPR@0.50 | high-conf FPR (P>=0.8) |
|---|---|---|---|---|---|
| controlled_or_agricultural_burn | 962 | 0.561 | 0.799 | 0.606 | 0.071 |
| false_or_rumour_report | 786 | 0.645 | 0.861 | 0.705 | 0.358 |
| industrial_heat_source | 2159 | 0.731 | 0.980 | 0.940 | 0.340 |
| sensor_noise | 1093 | 0.648 | 0.907 | 0.844 | 0.063 |

## Paired cases

### hist_gradient_boosting — overall pairwise ranking accuracy 0.740, mean margin 0.1762

| pair construction | pairs | ranking accuracy | mean P(fire member) - P(no-fire member) |
|---|---|---|---|
| multi_pixel_fire_vs_industrial_cluster | 400 | 0.748 | 0.1899 |
| news_led_fire_vs_repeated_false_reports | 350 | 0.744 | 0.1687 |
| persistent_wildfire_vs_persistent_industrial_heat | 650 | 0.795 | 0.2527 |
| satellite_and_real_news_fire_vs_thermal_source_and_false_news | 450 | 0.764 | 0.1935 |
| sparse_early_fire_vs_sparse_false_signal | 250 | 0.572 | 0.0415 |
| weak_early_fire_vs_weak_satellite_noise | 400 | 0.718 | 0.1093 |

### logistic_regression — overall pairwise ranking accuracy 0.712, mean margin 0.1303

| pair construction | pairs | ranking accuracy | mean P(fire member) - P(no-fire member) |
|---|---|---|---|
| multi_pixel_fire_vs_industrial_cluster | 400 | 0.721 | 0.1348 |
| news_led_fire_vs_repeated_false_reports | 350 | 0.724 | 0.1533 |
| persistent_wildfire_vs_persistent_industrial_heat | 650 | 0.758 | 0.1725 |
| satellite_and_real_news_fire_vs_thermal_source_and_false_news | 450 | 0.708 | 0.1536 |
| sparse_early_fire_vs_sparse_false_signal | 250 | 0.576 | 0.0228 |
| weak_early_fire_vs_weak_satellite_noise | 400 | 0.708 | 0.0779 |

### random_forest — overall pairwise ranking accuracy 0.735, mean margin 0.1623

| pair construction | pairs | ranking accuracy | mean P(fire member) - P(no-fire member) |
|---|---|---|---|
| multi_pixel_fire_vs_industrial_cluster | 400 | 0.765 | 0.1880 |
| news_led_fire_vs_repeated_false_reports | 350 | 0.743 | 0.1644 |
| persistent_wildfire_vs_persistent_industrial_heat | 650 | 0.758 | 0.1989 |
| satellite_and_real_news_fire_vs_thermal_source_and_false_news | 450 | 0.764 | 0.1736 |
| sparse_early_fire_vs_sparse_false_signal | 250 | 0.596 | 0.0490 |
| weak_early_fire_vs_weak_satellite_noise | 400 | 0.716 | 0.1335 |

## Sparse early evidence (uncertainty, not accuracy)

| model | Brier | ECE | mean P | 0.35-0.65 | P<0.10 | P>0.90 | fire share when P>0.90 |
|---|---|---|---|---|---|---|---|
| hist_gradient_boosting | 0.243 | 0.044 | 0.458 | 82.8% | 0.0% | 0.0% | n/a |
| logistic_regression | 0.247 | 0.061 | 0.439 | 99.8% | 0.0% | 0.0% | n/a |
| random_forest | 0.245 | 0.058 | 0.453 | 64.6% | 0.8% | 0.0% | n/a |

A constant P = 0.5 has Brier 0.25 on this balanced regime.

## Rule baseline (same rows)

### SUSPECTED or CONFIRMED = fire

| slice | rows | precision | recall | F1 | FP | FN | FPR | hard-neg FPR |
|---|---|---|---|---|---|---|---|---|
| all rows | 10000 | 0.528 | 0.845 | 0.650 | 3773 | 777 | 0.755 | 1.000 |
| non-sparse rows | 9000 | 0.525 | 0.878 | 0.657 | 3570 | 551 | 0.793 | 1.000 |
| multi_pixel_single_pass | 1600 | 0.541 | 0.841 | 0.659 | 570 | 127 | 0.713 | 1.000 |
| news_led | 1400 | 0.500 | 1.000 | 0.667 | 700 | 0 | 1.000 | 1.000 |
| persistent_thermal | 2600 | 0.513 | 0.858 | 0.642 | 1061 | 184 | 0.816 | 1.000 |
| satellite_news | 1800 | 0.500 | 1.000 | 0.667 | 900 | 0 | 1.000 | 1.000 |
| satellite_only_single_pass | 1600 | 0.623 | 0.700 | 0.659 | 339 | 240 | 0.424 | 1.000 |
| sparse_early_evidence | 1000 | 0.574 | 0.548 | 0.561 | 203 | 226 | 0.406 | 1.000 |

### CONFIRMED only = fire

| slice | rows | precision | recall | F1 | FP | FN | FPR | hard-neg FPR |
|---|---|---|---|---|---|---|---|---|
| all rows | 10000 | 0.573 | 0.270 | 0.367 | 1006 | 3650 | 0.201 | 0.314 |
| non-sparse rows | 9000 | 0.573 | 0.300 | 0.394 | 1006 | 3150 | 0.224 | 0.324 |
| multi_pixel_single_pass | 1600 | 0.000 | 0.000 | 0.000 | 0 | 800 | 0.000 | 0.000 |
| news_led | 1400 | 0.614 | 0.439 | 0.512 | 193 | 393 | 0.276 | 0.400 |
| persistent_thermal | 2600 | 0.528 | 0.265 | 0.352 | 308 | 956 | 0.237 | 0.308 |
| satellite_news | 1800 | 0.581 | 0.777 | 0.664 | 505 | 201 | 0.561 | 0.707 |
| satellite_only_single_pass | 1600 | 0.000 | 0.000 | 0.000 | 0 | 800 | 0.000 | 0.000 |
| sparse_early_evidence | 1000 | 0.000 | 0.000 | 0.000 | 0 | 500 | 0.000 | 0.000 |

### Model vs rules (non-sparse rows, model at its SUSPECTED threshold)

| model | threshold | precision | recall | F1 | hard-neg FPR | FPR reduction vs rules | recall change |
|---|---|---|---|---|---|---|---|
| hist_gradient_boosting | 0.40 | 0.614 | 0.815 | 0.700 | 0.583 | 41.7% | -0.063 |
| logistic_regression | 0.38 | 0.574 | 0.831 | 0.679 | 0.683 | 31.7% | -0.047 |
| random_forest | 0.41 | 0.606 | 0.818 | 0.696 | 0.626 | 37.4% | -0.060 |

## History ablation (does multi-overpass history help persistent cases?)

### hist_gradient_boosting

| feature set | features | grouped ROC-AUC | persistent ROC-AUC | persistent P@0.5 | persistent R@0.5 | persistent F1@0.5 | persistent FPR@0.5 | persistent FPR @ 80% recall |
|---|---|---|---|---|---|---|---|---|
| full_v5 | 25 | 0.728 | 0.771 | 0.693 | 0.689 | 0.691 | 0.305 | 0.433 |
| retained_14_baseline | 14 | 0.671 | 0.641 | 0.614 | 0.596 | 0.605 | 0.375 | 0.650 |
| without_current_spatial_temporal | 22 | 0.706 | 0.749 | 0.675 | 0.682 | 0.679 | 0.328 | 0.452 |
| without_frp_brightness | 14 | 0.707 | 0.736 | 0.676 | 0.651 | 0.663 | 0.312 | 0.515 |
| without_history | 20 | 0.709 | 0.709 | 0.663 | 0.615 | 0.638 | 0.313 | 0.564 |
| without_news | 19 | 0.706 | 0.761 | 0.677 | 0.682 | 0.679 | 0.326 | 0.460 |

Clustered-bootstrap AUC differences (environment resampling):

| comparison | slice | AUC difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| current_additions_effect_without_history_vs_retained_14 | non_persistent_rows | 0.0287 | [0.0215, 0.0360] | yes |
| current_additions_effect_without_history_vs_retained_14 | overall_grouped | 0.0387 | [0.0319, 0.0456] | yes |
| current_additions_effect_without_history_vs_retained_14 | persistent_thermal | 0.0676 | [0.0510, 0.0813] | yes |
| history_effect_full_vs_without_history | non_persistent_rows | 0.0019 | [-0.0010, 0.0046] | no |
| history_effect_full_vs_without_history | overall_grouped | 0.0189 | [0.0146, 0.0233] | yes |
| history_effect_full_vs_without_history | persistent_thermal | 0.0626 | [0.0497, 0.0755] | yes |
| total_effect_full_vs_retained_14 | non_persistent_rows | 0.0306 | [0.0241, 0.0369] | yes |
| total_effect_full_vs_retained_14 | overall_grouped | 0.0576 | [0.0502, 0.0650] | yes |
| total_effect_full_vs_retained_14 | persistent_thermal | 0.1302 | [0.1099, 0.1482] | yes |

### logistic_regression

| feature set | features | grouped ROC-AUC | persistent ROC-AUC | persistent P@0.5 | persistent R@0.5 | persistent F1@0.5 | persistent FPR@0.5 | persistent FPR @ 80% recall |
|---|---|---|---|---|---|---|---|---|
| full_v5 | 25 | 0.693 | 0.718 | 0.677 | 0.618 | 0.647 | 0.295 | 0.558 |
| retained_14_baseline | 14 | 0.661 | 0.628 | 0.593 | 0.609 | 0.601 | 0.418 | 0.720 |
| without_current_spatial_temporal | 22 | 0.683 | 0.704 | 0.670 | 0.609 | 0.638 | 0.300 | 0.580 |
| without_frp_brightness | 14 | 0.690 | 0.710 | 0.672 | 0.592 | 0.629 | 0.288 | 0.549 |
| without_history | 20 | 0.680 | 0.669 | 0.612 | 0.625 | 0.618 | 0.396 | 0.681 |
| without_news | 19 | 0.667 | 0.706 | 0.668 | 0.612 | 0.639 | 0.305 | 0.602 |

Clustered-bootstrap AUC differences (environment resampling):

| comparison | slice | AUC difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| current_additions_effect_without_history_vs_retained_14 | non_persistent_rows | 0.0109 | [0.0047, 0.0180] | yes |
| current_additions_effect_without_history_vs_retained_14 | overall_grouped | 0.0186 | [0.0124, 0.0250] | yes |
| current_additions_effect_without_history_vs_retained_14 | persistent_thermal | 0.0408 | [0.0272, 0.0514] | yes |
| history_effect_full_vs_without_history | non_persistent_rows | 0.0006 | [-0.0011, 0.0024] | no |
| history_effect_full_vs_without_history | overall_grouped | 0.0135 | [0.0091, 0.0179] | yes |
| history_effect_full_vs_without_history | persistent_thermal | 0.0494 | [0.0360, 0.0629] | yes |
| total_effect_full_vs_retained_14 | non_persistent_rows | 0.0115 | [0.0058, 0.0184] | yes |
| total_effect_full_vs_retained_14 | overall_grouped | 0.0321 | [0.0247, 0.0390] | yes |
| total_effect_full_vs_retained_14 | persistent_thermal | 0.0902 | [0.0716, 0.1081] | yes |

### random_forest

| feature set | features | grouped ROC-AUC | persistent ROC-AUC | persistent P@0.5 | persistent R@0.5 | persistent F1@0.5 | persistent FPR@0.5 | persistent FPR @ 80% recall |
|---|---|---|---|---|---|---|---|---|
| full_v5 | 25 | 0.721 | 0.752 | 0.677 | 0.658 | 0.667 | 0.314 | 0.473 |
| retained_14_baseline | 14 | 0.670 | 0.629 | 0.594 | 0.603 | 0.599 | 0.412 | 0.665 |
| without_current_spatial_temporal | 22 | 0.698 | 0.729 | 0.660 | 0.646 | 0.653 | 0.333 | 0.505 |
| without_frp_brightness | 14 | 0.706 | 0.729 | 0.671 | 0.623 | 0.646 | 0.306 | 0.508 |
| without_history | 20 | 0.707 | 0.704 | 0.658 | 0.609 | 0.633 | 0.317 | 0.574 |
| without_news | 19 | 0.702 | 0.746 | 0.675 | 0.650 | 0.662 | 0.313 | 0.508 |

Clustered-bootstrap AUC differences (environment resampling):

| comparison | slice | AUC difference | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| current_additions_effect_without_history_vs_retained_14 | non_persistent_rows | 0.0238 | [0.0176, 0.0303] | yes |
| current_additions_effect_without_history_vs_retained_14 | overall_grouped | 0.0372 | [0.0310, 0.0431] | yes |
| current_additions_effect_without_history_vs_retained_14 | persistent_thermal | 0.0751 | [0.0605, 0.0871] | yes |
| history_effect_full_vs_without_history | non_persistent_rows | 0.0009 | [-0.0014, 0.0033] | no |
| history_effect_full_vs_without_history | overall_grouped | 0.0138 | [0.0102, 0.0175] | yes |
| history_effect_full_vs_without_history | persistent_thermal | 0.0483 | [0.0360, 0.0596] | yes |
| total_effect_full_vs_retained_14 | non_persistent_rows | 0.0247 | [0.0193, 0.0304] | yes |
| total_effect_full_vs_retained_14 | overall_grouped | 0.0510 | [0.0443, 0.0570] | yes |
| total_effect_full_vs_retained_14 | persistent_thermal | 0.1234 | [0.1048, 0.1400] | yes |

## Calibration

| model | variant | Brier | ECE | grouped ROC-AUC | adopted |
|---|---|---|---|---|---|
| hist_gradient_boosting | isotonic | 0.2100 | 0.0073 | 0.7268 |  |
| hist_gradient_boosting | none | 0.2093 | 0.0073 | 0.7284 | yes |
| hist_gradient_boosting | sigmoid | 0.2094 | 0.0122 | 0.7284 |  |
| logistic_regression | isotonic | 0.2200 | 0.0052 | 0.6927 | yes |
| logistic_regression | none | 0.2205 | 0.0200 | 0.6924 |  |
| logistic_regression | sigmoid | 0.2205 | 0.0210 | 0.6922 |  |
| random_forest | isotonic | 0.2117 | 0.0051 | 0.7214 | yes |
| random_forest | none | 0.2128 | 0.0263 | 0.7209 |  |
| random_forest | sigmoid | 0.2122 | 0.0164 | 0.7213 |  |

- hist_gradient_boosting: no calibration variant improved ECE enough without costing ranking; left uncalibrated

- logistic_regression: isotonic improves ECE by 0.0148 with a grouped ROC-AUC change of +0.0004

- random_forest: isotonic improves ECE by 0.0212 with a grouped ROC-AUC change of +0.0005

## Threshold analysis

### hist_gradient_boosting

SUSPECTED candidate threshold **0.40** (highest with >= 80% non-sparse positive recall): non-sparse positive recall 81.5%, non-sparse precision 0.614, non-sparse FPR 51.3%, hard-negative FPR 58.3%, all-rows precision 0.607 / recall 0.811.

| regime | rows | precision | recall | FPR |
|---|---|---|---|---|
| multi_pixel_single_pass | 1600 | 0.612 | 0.794 | 0.502 |
| news_led | 1400 | 0.609 | 0.791 | 0.507 |
| persistent_thermal | 2600 | 0.641 | 0.832 | 0.466 |
| satellite_news | 1800 | 0.632 | 0.808 | 0.471 |
| satellite_only_single_pass | 1600 | 0.562 | 0.836 | 0.651 |
| sparse_early_evidence | 1000 | 0.550 | 0.776 | 0.634 |

CONFIRMED (probability only): precision >= 0.90 at recall >= 0.10 is **attainable** (threshold 0.82: precision 0.903, recall 0.143).

### logistic_regression

SUSPECTED candidate threshold **0.38** (highest with >= 80% non-sparse positive recall): non-sparse positive recall 83.1%, non-sparse precision 0.574, non-sparse FPR 61.6%, hard-negative FPR 68.3%, all-rows precision 0.567 / recall 0.832.

| regime | rows | precision | recall | FPR |
|---|---|---|---|---|
| multi_pixel_single_pass | 1600 | 0.566 | 0.819 | 0.629 |
| news_led | 1400 | 0.588 | 0.836 | 0.586 |
| persistent_thermal | 2600 | 0.604 | 0.777 | 0.509 |
| satellite_news | 1800 | 0.596 | 0.833 | 0.566 |
| satellite_only_single_pass | 1600 | 0.518 | 0.922 | 0.859 |
| sparse_early_evidence | 1000 | 0.513 | 0.844 | 0.800 |

CONFIRMED (probability only): precision >= 0.90 at recall >= 0.10 is **attainable** (threshold 0.82: precision 0.909, recall 0.112).

### random_forest

SUSPECTED candidate threshold **0.41** (highest with >= 80% non-sparse positive recall): non-sparse positive recall 81.8%, non-sparse precision 0.606, non-sparse FPR 53.2%, hard-negative FPR 62.6%, all-rows precision 0.603 / recall 0.809.

| regime | rows | precision | recall | FPR |
|---|---|---|---|---|
| multi_pixel_single_pass | 1600 | 0.618 | 0.812 | 0.502 |
| news_led | 1400 | 0.603 | 0.823 | 0.541 |
| persistent_thermal | 2600 | 0.622 | 0.835 | 0.508 |
| satellite_news | 1800 | 0.612 | 0.822 | 0.521 |
| satellite_only_single_pass | 1600 | 0.567 | 0.786 | 0.601 |
| sparse_early_evidence | 1000 | 0.572 | 0.724 | 0.542 |

CONFIRMED (probability only): precision >= 0.90 at recall >= 0.10 is **attainable** (threshold 0.78: precision 0.907, recall 0.131).

## Corroboration analysis (guardrail candidates for a future policy; nothing implemented)

### hist_gradient_boosting

| guardrail alone | rows | precision | recall |
|---|---|---|---|
| and_multi_pixel | 4706 | 0.508 | 0.478 |
| and_multiple_passes | 2600 | 0.500 | 0.260 |
| and_multiple_passes_and_satellite_news | 884 | 0.502 | 0.089 |
| and_satellite_and_news | 3494 | 0.501 | 0.350 |
| and_strong_news | 1581 | 0.650 | 0.206 |
| probability_only | 10000 | 0.500 | 1.000 |

P(fire) >= 0.40:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 2967 | 0.653 | 0.387 | 1031 |
| and_multiple_passes | 1687 | 0.641 | 0.216 | 606 |
| and_multiple_passes_and_satellite_news | 565 | 0.646 | 0.073 | 200 |
| and_satellite_and_news | 2164 | 0.640 | 0.277 | 779 |
| and_strong_news | 1391 | 0.692 | 0.192 | 429 |
| probability_only | 6679 | 0.607 | 0.811 | 2625 |

P(fire) >= 0.60:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 1552 | 0.801 | 0.249 | 309 |
| and_multiple_passes | 878 | 0.771 | 0.135 | 201 |
| and_multiple_passes_and_satellite_news | 308 | 0.763 | 0.047 | 73 |
| and_satellite_and_news | 1122 | 0.770 | 0.173 | 258 |
| and_strong_news | 935 | 0.770 | 0.144 | 215 |
| probability_only | 3075 | 0.742 | 0.456 | 793 |

P(fire) >= 0.70:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 1120 | 0.863 | 0.193 | 153 |
| and_multiple_passes | 574 | 0.819 | 0.094 | 104 |
| and_multiple_passes_and_satellite_news | 211 | 0.791 | 0.033 | 44 |
| and_satellite_and_news | 729 | 0.830 | 0.121 | 124 |
| and_strong_news | 536 | 0.836 | 0.090 | 88 |
| probability_only | 1453 | 0.833 | 0.242 | 242 |

P(fire) >= 0.80:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 791 | 0.906 | 0.143 | 74 |
| and_multiple_passes | 335 | 0.890 | 0.060 | 37 |
| and_multiple_passes_and_satellite_news | 119 | 0.882 | 0.021 | 14 |
| and_satellite_and_news | 442 | 0.894 | 0.079 | 47 |
| and_strong_news | 331 | 0.885 | 0.059 | 38 |
| probability_only | 885 | 0.892 | 0.158 | 96 |

P(fire) >= 0.82:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 721 | 0.915 | 0.132 | 61 |
| and_multiple_passes | 300 | 0.897 | 0.054 | 31 |
| and_multiple_passes_and_satellite_news | 109 | 0.899 | 0.020 | 11 |
| and_satellite_and_news | 402 | 0.905 | 0.073 | 38 |
| and_strong_news | 303 | 0.901 | 0.055 | 30 |
| probability_only | 794 | 0.903 | 0.143 | 77 |

### logistic_regression

| guardrail alone | rows | precision | recall |
|---|---|---|---|
| and_multi_pixel | 4706 | 0.508 | 0.478 |
| and_multiple_passes | 2600 | 0.500 | 0.260 |
| and_multiple_passes_and_satellite_news | 884 | 0.502 | 0.089 |
| and_satellite_and_news | 3494 | 0.501 | 0.350 |
| and_strong_news | 1581 | 0.650 | 0.206 |
| probability_only | 10000 | 0.500 | 1.000 |

P(fire) >= 0.38:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 3279 | 0.601 | 0.394 | 1308 |
| and_multiple_passes | 1672 | 0.604 | 0.202 | 662 |
| and_multiple_passes_and_satellite_news | 559 | 0.612 | 0.068 | 217 |
| and_satellite_and_news | 2317 | 0.606 | 0.281 | 914 |
| and_strong_news | 1430 | 0.678 | 0.194 | 460 |
| probability_only | 7331 | 0.567 | 0.832 | 3171 |

P(fire) >= 0.60:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 1833 | 0.738 | 0.270 | 481 |
| and_multiple_passes | 788 | 0.741 | 0.117 | 204 |
| and_multiple_passes_and_satellite_news | 292 | 0.733 | 0.043 | 78 |
| and_satellite_and_news | 1169 | 0.741 | 0.173 | 303 |
| and_strong_news | 966 | 0.755 | 0.146 | 237 |
| probability_only | 2696 | 0.724 | 0.391 | 743 |

P(fire) >= 0.70:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 964 | 0.865 | 0.167 | 130 |
| and_multiple_passes | 394 | 0.832 | 0.066 | 66 |
| and_multiple_passes_and_satellite_news | 163 | 0.816 | 0.027 | 30 |
| and_satellite_and_news | 565 | 0.850 | 0.096 | 85 |
| and_strong_news | 445 | 0.843 | 0.075 | 70 |
| probability_only | 1098 | 0.842 | 0.185 | 173 |

P(fire) >= 0.80:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 655 | 0.907 | 0.119 | 61 |
| and_multiple_passes | 268 | 0.892 | 0.048 | 29 |
| and_multiple_passes_and_satellite_news | 111 | 0.883 | 0.020 | 13 |
| and_satellite_and_news | 380 | 0.900 | 0.068 | 38 |
| and_strong_news | 297 | 0.889 | 0.053 | 33 |
| probability_only | 705 | 0.894 | 0.126 | 75 |

P(fire) >= 0.82:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 577 | 0.919 | 0.106 | 47 |
| and_multiple_passes | 238 | 0.908 | 0.043 | 22 |
| and_multiple_passes_and_satellite_news | 94 | 0.904 | 0.017 | 9 |
| and_satellite_and_news | 331 | 0.912 | 0.060 | 29 |
| and_strong_news | 262 | 0.905 | 0.047 | 25 |
| probability_only | 615 | 0.909 | 0.112 | 56 |

### random_forest

| guardrail alone | rows | precision | recall |
|---|---|---|---|
| and_multi_pixel | 4706 | 0.508 | 0.478 |
| and_multiple_passes | 2600 | 0.500 | 0.260 |
| and_multiple_passes_and_satellite_news | 884 | 0.502 | 0.089 |
| and_satellite_and_news | 3494 | 0.501 | 0.350 |
| and_strong_news | 1581 | 0.650 | 0.206 |
| probability_only | 10000 | 0.500 | 1.000 |

P(fire) >= 0.41:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 3119 | 0.637 | 0.397 | 1133 |
| and_multiple_passes | 1747 | 0.622 | 0.217 | 661 |
| and_multiple_passes_and_satellite_news | 608 | 0.610 | 0.074 | 237 |
| and_satellite_and_news | 2308 | 0.617 | 0.285 | 883 |
| and_strong_news | 1427 | 0.683 | 0.195 | 453 |
| probability_only | 6706 | 0.603 | 0.809 | 2663 |

P(fire) >= 0.60:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 1357 | 0.815 | 0.221 | 251 |
| and_multiple_passes | 705 | 0.782 | 0.110 | 154 |
| and_multiple_passes_and_satellite_news | 238 | 0.782 | 0.037 | 52 |
| and_satellite_and_news | 1031 | 0.780 | 0.161 | 227 |
| and_strong_news | 879 | 0.777 | 0.137 | 196 |
| probability_only | 2977 | 0.736 | 0.438 | 785 |

P(fire) >= 0.70:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 911 | 0.885 | 0.161 | 105 |
| and_multiple_passes | 367 | 0.869 | 0.064 | 48 |
| and_multiple_passes_and_satellite_news | 133 | 0.857 | 0.023 | 19 |
| and_satellite_and_news | 599 | 0.855 | 0.102 | 87 |
| and_strong_news | 468 | 0.850 | 0.080 | 70 |
| probability_only | 1331 | 0.830 | 0.221 | 226 |

P(fire) >= 0.78:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 670 | 0.919 | 0.123 | 54 |
| and_multiple_passes | 226 | 0.920 | 0.042 | 18 |
| and_multiple_passes_and_satellite_news | 78 | 0.949 | 0.015 | 4 |
| and_satellite_and_news | 347 | 0.914 | 0.063 | 30 |
| and_strong_news | 246 | 0.911 | 0.045 | 22 |
| probability_only | 723 | 0.907 | 0.131 | 67 |

P(fire) >= 0.80:

| guardrail | flagged | precision | recall | false positives |
|---|---|---|---|---|
| and_multi_pixel | 618 | 0.930 | 0.115 | 43 |
| and_multiple_passes | 202 | 0.931 | 0.038 | 14 |
| and_multiple_passes_and_satellite_news | 68 | 0.956 | 0.013 | 3 |
| and_satellite_and_news | 305 | 0.931 | 0.057 | 21 |
| and_strong_news | 210 | 0.933 | 0.039 | 14 |
| probability_only | 644 | 0.921 | 0.119 | 51 |

## Feature importance and shortcut audit

### hist_gradient_boosting — strongest single feature: `satellite_cluster_radius_km` with 22.3% of the total positive permutation importance

| feature | mean AUC drop | share of positive total |
|---|---|---|
| satellite_cluster_radius_km | 0.0592 | 22.3% |
| satellite_low_count | 0.0581 | 22.0% |
| satellite_high_count | 0.0244 | 9.2% |
| satellite_centroid_stability_km | 0.0228 | 8.6% |
| news_none_count | 0.0138 | 5.2% |
| news_strong_count | 0.0134 | 5.1% |
| satellite_frp_std | 0.0106 | 4.0% |
| satellite_brightness_mean | 0.0088 | 3.3% |
| news_weak_count | 0.0083 | 3.1% |
| satellite_frp_trend_per_hour | 0.0083 | 3.1% |

Blocks: current_additions_block 0.0654, frp_brightness_block 0.0533, history_block 0.0375, news_block 0.0403, satellite_confidence_counts 0.1010

### logistic_regression — strongest single feature: `satellite_low_count` with 28.1% of the total positive permutation importance

| feature | mean AUC drop | share of positive total |
|---|---|---|
| satellite_low_count | 0.0677 | 28.1% |
| satellite_cluster_radius_km | 0.0481 | 20.0% |
| satellite_centroid_stability_km | 0.0217 | 9.0% |
| news_none_count | 0.0192 | 8.0% |
| satellite_pass_count | 0.0165 | 6.9% |
| news_strong_count | 0.0119 | 5.0% |
| news_weak_count | 0.0109 | 4.5% |
| satellite_high_count | 0.0106 | 4.4% |
| satellite_frp_std | 0.0047 | 1.9% |
| news_satellite_lag_minutes | 0.0042 | 1.8% |

Blocks: current_additions_block 0.0509, frp_brightness_block 0.0108, history_block 0.0280, news_block 0.0434, satellite_confidence_counts 0.0853

Standardized coefficients (top 8): `satellite_low_count` -0.474, `satellite_cluster_radius_km` +0.439, `satellite_centroid_stability_km` +0.275, `news_none_count` -0.261, `satellite_high_count` +0.229, `satellite_pass_count` -0.227, `news_strong_count` +0.194, `news_weak_count` -0.154

### random_forest — strongest single feature: `satellite_low_count` with 28.4% of the total positive permutation importance

| feature | mean AUC drop | share of positive total |
|---|---|---|
| satellite_low_count | 0.0579 | 28.4% |
| satellite_cluster_radius_km | 0.0517 | 25.4% |
| satellite_centroid_stability_km | 0.0167 | 8.2% |
| satellite_high_count | 0.0149 | 7.3% |
| news_strong_count | 0.0126 | 6.2% |
| news_none_count | 0.0103 | 5.1% |
| news_weak_count | 0.0054 | 2.6% |
| satellite_nominal_count | 0.0046 | 2.3% |
| satellite_brightness_std | 0.0045 | 2.2% |
| satellite_brightness_mean | 0.0041 | 2.0% |

Blocks: current_additions_block 0.0560, frp_brightness_block 0.0409, history_block 0.0246, news_block 0.0315, satellite_confidence_counts 0.0955

Impurity importances (top 8): `satellite_cluster_radius_km` 0.151, `satellite_low_count` 0.090, `satellite_centroid_stability_km` 0.070, `satellite_frp_sum` 0.068, `satellite_brightness_mean` 0.065, `satellite_brightness_max` 0.062, `satellite_frp_max` 0.061, `satellite_frp_mean` 0.060

## Missingness audit

| feature | missing rate | fire share when missing | when present | indicator AUC | within-regime indicator AUC |
|---|---|---|---|---|---|
| news_satellite_lag_minutes | 65.1% | 50.0% | 50.1% | 0.499 | news_led: 0.499, persistent_thermal: 0.498 |
| satellite_brightness_trend_per_hour | 89.4% | 50.3% | 47.8% | 0.505 | persistent_thermal: 0.518 |
| satellite_centroid_stability_km | 74.0% | 50.0% | 50.0% | 0.500 | - |
| satellite_cluster_radius_km | 8.1% | 50.1% | 50.0% | 0.500 | news_led: 0.499, sparse_early_evidence: 0.503 |
| satellite_frp_trend_per_hour | 89.8% | 50.4% | 46.9% | 0.506 | persistent_thermal: 0.524 |
| satellite_history_span_minutes | 8.1% | 50.1% | 50.0% | 0.500 | news_led: 0.499, sparse_early_evidence: 0.503 |
| satellite_night_fraction | 10.6% | 50.8% | 49.9% | 0.502 | news_led: 0.503, persistent_thermal: 0.501, sparse_early_evidence: 0.508 |

Model use (held-out AUC drop when the values vs. the missingness pattern are shuffled):

- **hist_gradient_boosting**: `news_satellite_lag_minutes` values +0.0031 / missingness +0.0049; `satellite_brightness_trend_per_hour` values +0.0047 / missingness +0.0054; `satellite_centroid_stability_km` values +0.0219 / missingness +0.0226; `satellite_cluster_radius_km` values +0.0584 / missingness +0.0579; `satellite_frp_trend_per_hour` values +0.0058 / missingness +0.0062; `satellite_history_span_minutes` values +0.0013 / missingness +0.0013; `satellite_night_fraction` values +0.0007 / missingness +0.0006
- **logistic_regression**: `news_satellite_lag_minutes` values +0.0035 / missingness +0.0037; `satellite_brightness_trend_per_hour` values +0.0018 / missingness +0.0008; `satellite_centroid_stability_km` values +0.0216 / missingness +0.0229; `satellite_cluster_radius_km` values +0.0421 / missingness +0.0440; `satellite_frp_trend_per_hour` values -0.0001 / missingness -0.0000; `satellite_history_span_minutes` values +0.0012 / missingness +0.0014; `satellite_night_fraction` values +0.0009 / missingness +0.0012
- **random_forest**: `news_satellite_lag_minutes` values +0.0023 / missingness +0.0031; `satellite_brightness_trend_per_hour` values +0.0010 / missingness +0.0005; `satellite_centroid_stability_km` values +0.0180 / missingness +0.0165; `satellite_cluster_radius_km` values +0.0488 / missingness +0.0509; `satellite_frp_trend_per_hour` values +0.0020 / missingness +0.0020; `satellite_history_span_minutes` values +0.0009 / missingness +0.0004; `satellite_night_fraction` values +0.0005 / missingness +0.0006

## Dataset oracle

No generator / Bayes oracle probability exists: the frozen V5 generator (Task 6) persists only the latent subtype and label, not a posterior P(fire | features). It was not modified to create one, so no oracle ceiling is available; the predefined acceptance gate and the rule baseline are used instead.

## Limitations

- All training and evaluation data is synthetic; nothing here is real-world wildfire detection accuracy.
- Sparse early evidence is intentionally ambiguous; history features exist only in `persistent_thermal`.
- No site recurrence, multi-platform diversity or news-source-count signal.
- The SUSPECTED threshold is chosen and evaluated on the same grouped out-of-fold probabilities (a single scalar; per-fold behaviour at that threshold is in the JSON).
- V5 is not integrated into runtime.
