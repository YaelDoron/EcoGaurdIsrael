"""Task 7 configuration, written BEFORE any V5 model was evaluated: the acceptance gate, the
hyperparameter grid, operating-point targets and ablation definitions.

Nothing in this module may be changed after seeing results. The comparison report embeds a copy of it
(`report["acceptance_gate"]`) so a later edit is visible against the JSON that was produced.

All training and evaluation data is SYNTHETIC. The gate measures whether a model is fit to be a runtime
candidate on this benchmark - it is not a claim about real-world wildfire detection accuracy.
"""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_features_v5 import (
    CURRENT_EVIDENCE_FEATURE_NAMES_V5,
    FIRE_DETECTION_FEATURE_NAMES_V5,
    HISTORY_FEATURE_NAMES_V5,
    RETAINED_FEATURE_NAMES_V5,
)

DATASET_VERSION_V5 = "training_v5"
MODEL_VERSION_V5 = "5.0"
FEATURE_SCHEMA_VERSION_V5 = "v5"
RANDOM_SEED_V5 = 42
EXPECTED_TRAINING_CSV_SHA256_V5 = "37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035"
EXPECTED_ROWS_V5 = 10000

N_ENVIRONMENT_FOLDS = 5
SPARSE_REGIME = "sparse_early_evidence"
PERSISTENT_REGIME = "persistent_thermal"

# --- the acceptance gate (all criteria are critical: a model passes only if every one holds) -----------------

GATE_V5: dict = {
    # ranking / generalisation, on grouped-environment out-of-fold predictions
    "grouped_environment_mean_roc_auc_min": 0.75,
    "grouped_environment_worst_fold_roc_auc_min": 0.65,
    # operating point: the SUSPECTED threshold is the HIGHEST threshold whose recall on NON-SPARSE positives is >= 0.80
    "non_sparse_positive_recall_min": 0.80,
    # hard negatives = no-fire rows with fire-looking evidence (see `is_hard_negative_v5`), sparse rows excluded
    "hard_negative_fpr_max": 0.45,
    # history was added to solve persistence
    "persistent_thermal_roc_auc_min": 0.70,
    # a runtime-ready probability must be a usable probability (grouped out-of-fold)
    "brier_score_max_exclusive": 0.25,
    "expected_calibration_error_max": 0.10,
    # no single feature may carry more than ~40 % of the total positive held-out permutation importance
    "top_feature_importance_share_max": 0.40,
    "top_feature_importance_share_tolerance": 0.02,  # "approximately 40 %": numeric slack on that one criterion
    # must beat the rule engine meaningfully at the SAME rows (non-sparse rows, SUSPECTED-or-CONFIRMED = fire)
    "vs_rules_hard_negative_fpr_max_ratio": 0.75,  # >= 25 % relative reduction of the rule's hard-negative FPR
    "vs_rules_recall_max_drop": 0.15,  # recall may not collapse: model recall >= rule recall - 0.15
    "vs_rules_f1_min_delta": 0.0,  # and the model's F1 may not be worse than the rules'
}

# --- operating-point targets --------------------------------------------------------------------------------

SUSPECT_RECALL_TARGET = GATE_V5["non_sparse_positive_recall_min"]
CONFIRM_PRECISION_TARGET = 0.90
CONFIRM_MIN_RECALL = 0.10  # "meaningful recall" for a probability-only CONFIRMED threshold
THRESHOLD_GRID = tuple(round(0.01 * step, 2) for step in range(1, 100))  # 0.01 ... 0.99
DEFAULT_THRESHOLD = 0.50
HIGH_CONFIDENCE_FALSE_POSITIVE_THRESHOLD = 0.80

# sparse-evidence uncertainty bands
SPARSE_UNCERTAIN_BAND = (0.35, 0.65)
SPARSE_LOW_EXTREME = 0.10
SPARSE_HIGH_EXTREME = 0.90

# --- selection ---------------------------------------------------------------------------------------------

SIMPLICITY_MARGIN_ROC_AUC = 0.02  # a simpler passing model within this grouped ROC-AUC of the best passer wins
CALIBRATION_ADOPT_MIN_ECE_IMPROVEMENT = 0.01  # calibration is adopted only if it improves ECE by this much ...
CALIBRATION_MAX_ROC_AUC_LOSS = 0.005  # ... without losing more than this much grouped ROC-AUC

# --- hyperparameter grid: small and fixed a priori; chosen per family by grouped-environment mean ROC-AUC only --

MODEL_FAMILIES_V5: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting")
SIMPLICITY_RANK_V5: dict[str, int] = {"logistic_regression": 0, "random_forest": 1, "hist_gradient_boosting": 2}

CANDIDATE_GRID_V5: dict[str, tuple[dict, ...]] = {
    "logistic_regression": tuple({"C": c} for c in (0.01, 0.1, 1.0, 10.0)),
    "random_forest": tuple(
        {"max_depth": depth, "min_samples_leaf": leaf} for depth, leaf in ((4, 20), (6, 20), (8, 10), (12, 10))
    ),
    "hist_gradient_boosting": tuple(
        {"max_depth": depth, "learning_rate": rate} for depth, rate in ((2, 0.05), (3, 0.05), (3, 0.10), (4, 0.05))
    ),
}
FIXED_HYPERPARAMETERS_V5: dict[str, dict] = {
    "logistic_regression": {"penalty": "l2", "max_iter": 2000},
    "random_forest": {"n_estimators": 300, "max_features": "sqrt", "n_jobs": 1},
    "hist_gradient_boosting": {"max_iter": 200, "min_samples_leaf": 40, "l2_regularization": 1.0, "early_stopping": False},
}

# --- ablations (feature groups derived from the canonical schema, never re-typed) -------------------------------

_NEWS_FEATURES = tuple(name for name in RETAINED_FEATURE_NAMES_V5 if name.startswith("news_")) + ("news_satellite_lag_minutes",)
_FRP_BRIGHTNESS_FEATURES = tuple(
    name for name in FIRE_DETECTION_FEATURE_NAMES_V5 if "frp" in name or "brightness" in name
)
_CURRENT_SPATIAL_TEMPORAL = ("satellite_night_fraction", "satellite_cluster_radius_km", "news_satellite_lag_minutes")

FEATURE_SETS_V5: dict[str, tuple[str, ...]] = {
    "full_v5": FIRE_DETECTION_FEATURE_NAMES_V5,
    "without_history": tuple(n for n in FIRE_DETECTION_FEATURE_NAMES_V5 if n not in HISTORY_FEATURE_NAMES_V5),
    "retained_14_baseline": RETAINED_FEATURE_NAMES_V5,
    "without_news": tuple(n for n in FIRE_DETECTION_FEATURE_NAMES_V5 if n not in _NEWS_FEATURES),
    "without_frp_brightness": tuple(n for n in FIRE_DETECTION_FEATURE_NAMES_V5 if n not in _FRP_BRIGHTNESS_FEATURES),
    "without_current_spatial_temporal": tuple(
        n for n in FIRE_DETECTION_FEATURE_NAMES_V5 if n not in _CURRENT_SPATIAL_TEMPORAL
    ),
}
# The 14 retained -> + current-evidence additions -> + history: the incremental architecture ladder.
ABLATION_LADDER_V5: tuple[str, ...] = ("retained_14_baseline", "without_history", "full_v5")

PERMUTATION_BLOCKS_V5: dict[str, tuple[str, ...]] = {
    "history_block": HISTORY_FEATURE_NAMES_V5,
    "news_block": _NEWS_FEATURES,
    "frp_brightness_block": _FRP_BRIGHTNESS_FEATURES,
    "satellite_confidence_counts": ("satellite_low_count", "satellite_nominal_count", "satellite_high_count"),
    "current_additions_block": CURRENT_EVIDENCE_FEATURE_NAMES_V5,
}

PERMUTATION_REPEATS = 5
BOOTSTRAP_REPEATS = 500
