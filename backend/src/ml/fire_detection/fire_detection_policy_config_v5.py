"""Task 8 configuration, written BEFORE the confirmatory dataset was generated or any policy was evaluated:
the locked model, the locked 3-state policy, the confirmatory-dataset identity and the policy acceptance gate.

Nothing in this module may be changed after confirmatory evaluation begins. If the policy fails its gate, any
redesign is a NEW, explicitly versioned policy evaluated on ANOTHER untouched dataset - never an edit here.

Task 7 (binary-primary V5 model gate) FAILED and that verdict stays documented. Task 8 asks a different question:
can the HGB likelihood estimator, together with an explicit uncertainty state (SUSPECTED) and one deterministic
corroboration guardrail, make a defensible NO_EVENT / SUSPECTED / CONFIRMED decision?

All training and confirmatory evaluation data is SYNTHETIC. This is offline: nothing here is wired to runtime.
"""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5

TASK_7_BINARY_PRIMARY_GATE_VERDICT = "FAILED"  # historical, immutable
POLICY_VERSION = "ai_hybrid_policy_v5.0"

# --- locked model (Task 7's strongest V5 candidate; NOT re-tuned, no other family evaluated) -----------------

MODEL_FAMILY = "hist_gradient_boosting"
MODEL_CLASS_NAME = "HistGradientBoostingClassifier"
MODEL_PARAMS: dict = {"max_depth": 3, "learning_rate": 0.1}
MODEL_FEATURE_NAMES = FIRE_DETECTION_FEATURE_NAMES_V5  # the exact ordered V5 contract

# --- locked 3-state policy ----------------------------------------------------------------------------------
#
#   P(fire) <  SUSPECT_THRESHOLD                                     -> NO_EVENT
#   P(fire) >= SUSPECT_THRESHOLD, CONFIRMED condition not satisfied  -> SUSPECTED
#   P(fire) >= CONFIRM_THRESHOLD AND current satellite pixels >= 2   -> CONFIRMED
#
# The multi-pixel condition is a deterministic CORROBORATION GUARDRAIL, not the detector: the model stays the
# primary likelihood estimator. There is exactly ONE confirmation branch.

SUSPECT_THRESHOLD = 0.40
CONFIRM_THRESHOLD = 0.80
MULTI_PIXEL_MIN_SATELLITE_PIXELS = 2
# current satellite pixel count = low + nominal + high confidence hotspots of the CURRENT candidate
MULTI_PIXEL_COUNT_FEATURES = ("satellite_low_count", "satellite_nominal_count", "satellite_high_count")
MULTI_PIXEL_DEFINITION = (
    "satellite_low_count + satellite_nominal_count + satellite_high_count >= 2 (hotspots of the current candidate)"
)

STATUS_NO_EVENT = "NO_EVENT"
STATUS_SUSPECTED = "SUSPECTED"
STATUS_CONFIRMED = "CONFIRMED"
STATUSES = (STATUS_NO_EVENT, STATUS_SUSPECTED, STATUS_CONFIRMED)

# --- untouched confirmatory dataset ---------------------------------------------------------------------------
# Produced by the FROZEN V5 generator (unchanged distributions) with a NEW seed. The generator numbers
# environments / samples / pairs from 1 for every seed, so the rows are namespaced by these offsets purely to
# make disjointness checkable; the environment PARAMETERS are new because they are drawn from (seed, id).

CONFIRMATORY_SEED = 8202609
CONFIRMATORY_ROWS = 10000
CONFIRMATORY_FILE_NAME = "fire_detection_v5_policy_validation.csv"
SAMPLE_ID_OFFSET = 1_000_000
ENVIRONMENT_ID_OFFSET = 10_000
PAIR_ID_OFFSET = 1_000_000

TRAINING_CSV_SHA256 = "37e056793fb4dda72f517a20153b21094b284e1579a065140483b8a6e1d78035"
# Fingerprint of the deterministic confirmatory file, recorded right after generation and BEFORE any evaluation.
CONFIRMATORY_CSV_SHA256 = "7c24405e83fdceb1c4f74801da97d80aae84ea46161654c282df30ea190d8ead"  # recorded right after generation

# --- policy acceptance gate (every criterion is critical) ----------------------------------------------------

POLICY_GATE_V5: dict = {
    # CONFIRMED must be trustworthy ...
    "confirmed_precision_min": 0.90,
    # ... but must not confirm almost nothing (recall on NON-SPARSE positive rows)
    "confirmed_recall_min_non_sparse": 0.10,
    # NO_FIRE rows classified CONFIRMED / all NO_FIRE rows
    "false_confirmation_rate_max": 0.10,
    # SUSPECTED or CONFIRMED = "fire retained for monitoring"; non-sparse positive rows
    "fire_retained_recall_min_non_sparse": 0.80,
    # non-sparse no-fire rows with status != NO_EVENT must be >= 25 % relatively lower than the rule detector's
    "non_sparse_no_fire_alert_rate_max_ratio_vs_rules": 0.75,
    # sparse early evidence is ambiguous: it must mostly express uncertainty, not confirm
    "sparse_confirmed_rate_max": 0.05,
    # calibration of P(fire) on the confirmatory data
    "brier_score_max_exclusive": 0.25,
    "expected_calibration_error_max": 0.10,
    # history was added to solve persistence
    "persistent_thermal_roc_auc_min": 0.70,
}

SPARSE_REGIME = "sparse_early_evidence"
PERSISTENT_REGIME = "persistent_thermal"
PASS_COUNT_GROUPS = ("1", "2", ">=3")

MODEL_ARTIFACT_STEM = "fire_detection_hgb_v5"
