"""Task 8: confirmatory evaluation of the locked AI Hybrid policy on the untouched confirmatory dataset.

    train the locked HGB ONCE on training_v5.csv  ->  P(fire) on the confirmatory rows  ->  policy  ->  gate

Nothing here selects a model, a feature, a threshold or a calibration: the policy (0.40 / 0.80 / multi-pixel >= 2) and
the gate are read from fire_detection_policy_config_v5 (locked before evaluation) and are never adjusted. The
existing rule detector is evaluated on exactly the same rows. All data is synthetic.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ml.fire_detection.fire_detection_ai_hybrid_policy_v5 import STATUS_CODES, decide_statuses
from src.ml.fire_detection.fire_detection_evaluation_v5 import (
    calibration_report,
    hard_negative_mask,
    non_sparse_mask,
    rule_baseline_outputs,
    sparse_evidence_report,
    to_builtin,
    verify_samples_match_matrix,
)
from src.ml.fire_detection.fire_detection_features_v5 import FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection.fire_detection_model_v5 import (
    TrainingMatrixV5,
    assert_feature_schema_v5,
    build_pipeline_v5,
    dataset_sha256,
    describe_preprocessing_v5,
)
from src.ml.fire_detection.fire_detection_policy_config_v5 import (
    CONFIRM_THRESHOLD,
    CONFIRMATORY_CSV_SHA256,
    CONFIRMATORY_ROWS,
    CONFIRMATORY_SEED,
    MODEL_ARTIFACT_STEM,
    MODEL_CLASS_NAME,
    MODEL_FAMILY,
    MODEL_PARAMS,
    MULTI_PIXEL_DEFINITION,
    PERSISTENT_REGIME,
    POLICY_GATE_V5,
    POLICY_VERSION,
    SPARSE_REGIME,
    STATUSES,
    SUSPECT_THRESHOLD,
    TASK_7_BINARY_PRIMARY_GATE_VERDICT,
    TRAINING_CSV_SHA256,
)

SYNTHETIC_NOTICE = (
    "All training and confirmatory evaluation data is SYNTHETIC. These results describe the AI Hybrid policy on the "
    "V5 benchmark generator and are NOT real-world wildfire detection accuracy."
)
POLICY_PASSED = "AI HYBRID POLICY PASSED"
POLICY_FAILED = "AI HYBRID POLICY FAILED"
_NO_EVENT, _SUSPECTED, _CONFIRMED = (STATUS_CODES[name] for name in STATUSES)


class ArtifactRefusedError(RuntimeError):
    """Raised when saving the Task 8 artifact is refused."""


# ---------------------------------------------------------------------------------------------------------
# training (exactly once) and scoring
# ---------------------------------------------------------------------------------------------------------


def train_locked_model(training_matrix: TrainingMatrixV5):
    """Fit the locked HGB (max_depth 3, learning_rate 0.1, exact V5 features) on the TRAINING rows only."""
    names = assert_feature_schema_v5(training_matrix.feature_names)
    return build_pipeline_v5(MODEL_FAMILY, MODEL_PARAMS, names).fit(training_matrix.columns(names), training_matrix.y)


def satellite_pixel_counts(matrix: TrainingMatrixV5) -> np.ndarray:
    return sum(matrix.feature(name) for name in ("satellite_low_count", "satellite_nominal_count", "satellite_high_count"))


# ---------------------------------------------------------------------------------------------------------
# outcome tables
# ---------------------------------------------------------------------------------------------------------


def outcome_counts(y: np.ndarray, codes: np.ndarray) -> dict:
    """Ground truth FIRE / NO_FIRE  x  NO_EVENT / SUSPECTED / CONFIRMED, with row rates."""
    table = {}
    for label, name in ((1, "fire"), (0, "no_fire")):
        mask = y == label
        total = int(mask.sum())
        counts = {status: int(np.sum(codes[mask] == STATUS_CODES[status])) for status in STATUSES}
        table[name] = {
            "rows": total,
            **counts,
            "rates": {status: (counts[status] / total if total else None) for status in STATUSES},
        }
    return table


def status_breakdown(matrix: TrainingMatrixV5, codes: np.ndarray, groups: np.ndarray) -> dict:
    result = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        entry = outcome_counts(matrix.y[mask], codes[mask])
        entry["rows"] = int(mask.sum())
        result[group] = entry
    return result


def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float | None:
    den = int(mask_den.sum())
    return float(mask_num[mask_den].mean()) if den else None


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _binary_slice(y: np.ndarray, alert: np.ndarray, mask: np.ndarray, hard: np.ndarray) -> dict:
    yy, aa = y[mask], alert[mask].astype(bool)
    tp, fp = int(np.sum(aa & (yy == 1))), int(np.sum(aa & (yy == 0)))
    fn = int(np.sum(~aa & (yy == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    negatives = yy == 0
    return {
        "rows": int(mask.sum()),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": float(aa[negatives].mean()) if negatives.any() else None,
        "hard_negative_fpr": float(alert[hard & mask].mean()) if (hard & mask).any() else None,
    }


# ---------------------------------------------------------------------------------------------------------
# the policy gate
# ---------------------------------------------------------------------------------------------------------


def evaluate_policy_gate(metrics: dict, gate: dict = POLICY_GATE_V5) -> dict:
    """Apply the predefined gate to measured values. Every criterion is critical; an uncomputable one fails."""
    criteria = {}

    def check(name, value, passed, requirement):
        criteria[name] = {"value": value, "requirement": requirement, "passed": None if value is None else bool(passed)}

    def ge(v, m):
        return v is not None and v >= m

    def le(v, m):
        return v is not None and v <= m

    check("confirmed_precision", metrics["confirmed_precision"], ge(metrics["confirmed_precision"], gate["confirmed_precision_min"]),
          f">= {gate['confirmed_precision_min']}")
    check("confirmed_recall_non_sparse", metrics["confirmed_recall_non_sparse"],
          ge(metrics["confirmed_recall_non_sparse"], gate["confirmed_recall_min_non_sparse"]), f">= {gate['confirmed_recall_min_non_sparse']}")
    check("false_confirmation_rate", metrics["false_confirmation_rate"],
          le(metrics["false_confirmation_rate"], gate["false_confirmation_rate_max"]), f"<= {gate['false_confirmation_rate_max']}")
    check("fire_retained_recall_non_sparse", metrics["fire_retained_recall_non_sparse"],
          ge(metrics["fire_retained_recall_non_sparse"], gate["fire_retained_recall_min_non_sparse"]),
          f">= {gate['fire_retained_recall_min_non_sparse']}")
    rule_rate = metrics["rule_non_sparse_no_fire_alert_rate"]
    ratio_limit = None if rule_rate is None else gate["non_sparse_no_fire_alert_rate_max_ratio_vs_rules"] * rule_rate
    check("non_sparse_no_fire_alert_rate_vs_rules", metrics["non_sparse_no_fire_alert_rate"],
          ratio_limit is not None and le(metrics["non_sparse_no_fire_alert_rate"], ratio_limit),
          f"<= {gate['non_sparse_no_fire_alert_rate_max_ratio_vs_rules']} x the rule detector's {_fmt(rule_rate)} = {_fmt(ratio_limit)}")
    check("sparse_confirmed_rate", metrics["sparse_confirmed_rate"], le(metrics["sparse_confirmed_rate"], gate["sparse_confirmed_rate_max"]),
          f"<= {gate['sparse_confirmed_rate_max']}")
    check("brier_score", metrics["brier_score"], metrics["brier_score"] is not None and metrics["brier_score"] < gate["brier_score_max_exclusive"],
          f"< {gate['brier_score_max_exclusive']}")
    check("expected_calibration_error", metrics["expected_calibration_error"],
          le(metrics["expected_calibration_error"], gate["expected_calibration_error_max"]), f"<= {gate['expected_calibration_error_max']}")
    check("persistent_thermal_roc_auc", metrics["persistent_thermal_roc_auc"],
          ge(metrics["persistent_thermal_roc_auc"], gate["persistent_thermal_roc_auc_min"]), f">= {gate['persistent_thermal_roc_auc_min']}")
    failed = [name for name, c in criteria.items() if c["passed"] is not True]
    return {"criteria": criteria, "passed": not failed, "failed_criteria": failed}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# ---------------------------------------------------------------------------------------------------------
# uncertainty of the gate metrics (DESCRIPTIVE: not part of the gate, never changes the verdict)
# ---------------------------------------------------------------------------------------------------------

BOOTSTRAP_REPEATS = 500
BOOTSTRAP_SEED = 20260925


def gate_metric_uncertainty(matrix: TrainingMatrixV5, p: np.ndarray, codes: np.ndarray, rule_alert: np.ndarray,
                            repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Environment-clustered bootstrap 95 % intervals of the gate metrics (whole environments are resampled)."""
    y, envs = matrix.y, matrix.environments
    ns, sparse, persistent = non_sparse_mask(matrix), matrix.regimes == SPARSE_REGIME, matrix.regimes == PERSISTENT_REGIME
    confirmed, retained = codes == _CONFIRMED, codes >= _SUSPECTED
    unique = np.unique(envs)
    index_by_env = {env: np.flatnonzero(envs == env) for env in unique}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in (
        "confirmed_precision", "confirmed_recall_non_sparse", "false_confirmation_rate", "fire_retained_recall_non_sparse",
        "non_sparse_no_fire_alert_rate_ratio_vs_rules", "sparse_confirmed_rate", "brier_score", "expected_calibration_error",
        "persistent_thermal_roc_auc")}
    for _ in range(repeats):
        idx = np.concatenate([index_by_env[e] for e in rng.choice(unique, size=len(unique), replace=True)])
        yb, pb, cb, rb, nb = y[idx], p[idx], confirmed[idx], retained[idx], ns[idx]
        if not cb.any() or not (nb & (yb == 1)).any() or not (nb & (yb == 0)).any():
            continue
        neg_ns = nb & (yb == 0)
        rule_rate = rule_alert[idx][neg_ns].mean()
        cal = calibration_report(yb, pb)
        pers = persistent[idx]
        draws["confirmed_precision"].append(yb[cb].mean())
        draws["confirmed_recall_non_sparse"].append(cb[nb & (yb == 1)].mean())
        draws["false_confirmation_rate"].append(cb[yb == 0].mean())
        draws["fire_retained_recall_non_sparse"].append(rb[nb & (yb == 1)].mean())
        draws["non_sparse_no_fire_alert_rate_ratio_vs_rules"].append(rb[neg_ns].mean() / rule_rate if rule_rate else np.nan)
        draws["sparse_confirmed_rate"].append(cb[sparse[idx]].mean())
        draws["brier_score"].append(cal["brier_score"])
        draws["expected_calibration_error"].append(cal["expected_calibration_error"])
        draws["persistent_thermal_roc_auc"].append(roc_auc_score(yb[pers], pb[pers]))
    limits = {
        "confirmed_precision": (">=", POLICY_GATE_V5["confirmed_precision_min"]),
        "confirmed_recall_non_sparse": (">=", POLICY_GATE_V5["confirmed_recall_min_non_sparse"]),
        "false_confirmation_rate": ("<=", POLICY_GATE_V5["false_confirmation_rate_max"]),
        "fire_retained_recall_non_sparse": (">=", POLICY_GATE_V5["fire_retained_recall_min_non_sparse"]),
        "non_sparse_no_fire_alert_rate_ratio_vs_rules": ("<=", POLICY_GATE_V5["non_sparse_no_fire_alert_rate_max_ratio_vs_rules"]),
        "sparse_confirmed_rate": ("<=", POLICY_GATE_V5["sparse_confirmed_rate_max"]),
        "brier_score": ("<", POLICY_GATE_V5["brier_score_max_exclusive"]),
        "expected_calibration_error": ("<=", POLICY_GATE_V5["expected_calibration_error_max"]),
        "persistent_thermal_roc_auc": (">=", POLICY_GATE_V5["persistent_thermal_roc_auc_min"]),
    }
    result = {}
    for name, values in draws.items():
        values = np.array([v for v in values if not np.isnan(v)])
        low, high = np.percentile(values, [2.5, 97.5])
        direction, limit = limits[name]
        clears = low >= limit if direction in (">=",) else high <= limit if direction in ("<=", "<") else None
        result[name] = {"ci_95": [float(low), float(high)], "gate_limit": f"{direction} {limit}", "whole_interval_clears_limit": bool(clears)}
    return {
        "note": "Environment-clustered bootstrap (whole environments resampled). DESCRIPTIVE only: the verdict comes from the "
        "point estimates against the predefined gate; this shows how robust each margin is on one synthetic draw.",
        "repeats": repeats,
        "metrics": result,
    }


# ---------------------------------------------------------------------------------------------------------
# the confirmatory evaluation
# ---------------------------------------------------------------------------------------------------------


def evaluate_policy_v5(
    model,
    matrix: TrainingMatrixV5,
    samples,
    *,
    training_csv_sha256: str,
    confirmatory_csv_sha256: str,
    training_rows: int,
    overlap: dict,
) -> dict:
    """Score the confirmatory rows once, apply the locked policy, run the gate and every required breakdown."""
    assert_feature_schema_v5(matrix.feature_names)
    verify_samples_match_matrix(samples, matrix)  # regenerated evidence == the confirmatory CSV rows
    y = matrix.y
    p = model.predict_proba(matrix.X)[:, 1]
    pixels = satellite_pixel_counts(matrix)
    codes = decide_statuses(p, pixels)
    rule = rule_baseline_outputs(samples)

    ns = non_sparse_mask(matrix)
    hard = hard_negative_mask(matrix)
    positives_ns, negatives_ns = ns & (y == 1), ns & (y == 0)
    sparse = matrix.regimes == SPARSE_REGIME
    persistent = matrix.regimes == PERSISTENT_REGIME
    confirmed, retained = codes == _CONFIRMED, codes >= _SUSPECTED
    calibration = calibration_report(y, p)

    confirmed_count = int(confirmed.sum())
    rule_alert, rule_confirmed = rule.predicted_fire.astype(bool), rule.predicted_confirmed.astype(bool)
    metrics = {
        "confirmed_rows": confirmed_count,
        "confirmed_precision": float(y[confirmed].mean()) if confirmed_count else 0.0,
        "confirmed_precision_non_sparse": float(y[confirmed & ns].mean()) if (confirmed & ns).any() else None,
        "confirmed_recall_non_sparse": _rate(confirmed, positives_ns),
        "confirmed_recall_all_positives": _rate(confirmed, y == 1),
        "false_confirmation_rate": _rate(confirmed, y == 0),
        "false_confirmed_rows": int((confirmed & (y == 0)).sum()),
        "fire_retained_recall_non_sparse": _rate(retained, positives_ns),
        "non_sparse_no_fire_alert_rate": _rate(retained, negatives_ns),
        "rule_non_sparse_no_fire_alert_rate": _rate(rule_alert, negatives_ns),
        "sparse_confirmed_rate": float(confirmed[sparse].mean()),
        "brier_score": calibration["brier_score"],
        "expected_calibration_error": calibration["expected_calibration_error"],
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "persistent_thermal_roc_auc": float(roc_auc_score(y[persistent], p[persistent])),
    }
    gate = evaluate_policy_gate(metrics)

    by_subtype = {}
    for subtype in sorted(set(matrix.subtypes.tolist())):
        mask = matrix.subtypes == subtype
        entry = outcome_counts(y[mask], codes[mask])
        label = int(y[mask][0])
        by_subtype[subtype] = {
            "label": label,
            "rows": int(mask.sum()),
            "statuses": entry["fire" if label else "no_fire"],
            "confirmed_rate": float(confirmed[mask].mean()),
            "alert_rate": float(retained[mask].mean()),
            "mean_p_fire": float(p[mask].mean()),
        }

    sparse_report = sparse_evidence_report(matrix, p)
    sparse_report.update(
        {
            "suspected_rate": float(np.mean(codes[sparse] == _SUSPECTED)),
            "confirmed_rate": float(confirmed[sparse].mean()),
            "no_event_rate": float(np.mean(codes[sparse] == _NO_EVENT)),
            "status_by_truth": outcome_counts(y[sparse], codes[sparse]),
        }
    )

    pass_counts = matrix.feature("satellite_pass_count")
    groups = {"1": pass_counts == 1, "2": pass_counts == 2, ">=3": pass_counts >= 3, "0 (no satellite)": pass_counts == 0}
    progression = {}
    for name, mask in groups.items():
        if not mask.any():
            continue
        fire_mask = mask & (y == 1)
        entry = {
            "rows": int(mask.sum()),
            "fire_rows": int(fire_mask.sum()),
            "mean_p_fire": float(p[mask].mean()),
            "median_p_fire": float(np.median(p[mask])),
            "no_event_rate": float(np.mean(codes[mask] == _NO_EVENT)),
            "suspected_rate": float(np.mean(codes[mask] == _SUSPECTED)),
            "confirmed_rate": float(confirmed[mask].mean()),
            "alert_precision": float(y[mask & retained].mean()) if (mask & retained).any() else None,
            "alert_recall": _rate(retained, fire_mask) if fire_mask.any() else None,
            "confirmed_precision": float(y[mask & confirmed].mean()) if (mask & confirmed).any() else None,
            "confirmed_recall": _rate(confirmed, fire_mask) if fire_mask.any() else None,
        }
        if fire_mask.any():
            entry["fire_rows_mean_p"] = float(p[fire_mask].mean())
            entry["fire_rows_median_p"] = float(np.median(p[fire_mask]))
        progression[name] = entry
    progression_persistent = {}
    for name in ("1", "2", ">=3"):
        mask = groups[name] & persistent
        if mask.any() and (y[mask] == 1).any() and (y[mask] == 0).any():
            progression_persistent[name] = {
                "rows": int(mask.sum()),
                "fire_rows_mean_p": float(p[mask & (y == 1)].mean()),
                "no_fire_rows_mean_p": float(p[mask & (y == 0)].mean()),
                "confirmed_rate_fire": float(confirmed[mask & (y == 1)].mean()),
                "confirmed_rate_no_fire": float(confirmed[mask & (y == 0)].mean()),
                "roc_auc": float(roc_auc_score(y[mask], p[mask])),
            }

    hard_ns = hard & ns
    rule_slice = _binary_slice(y, rule_alert, ns, hard)
    policy_slice = _binary_slice(y, retained, ns, hard)
    rule_confirmed_slice = _binary_slice(y, rule_confirmed, ns, hard)
    policy_confirmed_slice = _binary_slice(y, confirmed, ns, hard)
    comparison = {
        "definitions": "rule detector: SUSPECTED or CONFIRMED = fire (alert), CONFIRMED-only reported separately; "
        "AI Hybrid policy: status != NO_EVENT = alert. Same confirmatory rows; non-sparse rows for the slices.",
        "non_sparse_alert": {"rule_detector": rule_slice, "ai_hybrid_policy": policy_slice},
        "non_sparse_confirmed_only": {"rule_detector": rule_confirmed_slice, "ai_hybrid_policy": policy_confirmed_slice},
        "false_confirmed_rows": {
            "rule_detector": int((rule_confirmed & (y == 0)).sum()),
            "ai_hybrid_policy": int((confirmed & (y == 0)).sum()),
            "rule_detector_rate": _rate(rule_confirmed, y == 0),
            "ai_hybrid_policy_rate": _rate(confirmed, y == 0),
        },
        "confirmed_precision": {
            "rule_detector": float(y[rule_confirmed].mean()) if rule_confirmed.any() else None,
            "ai_hybrid_policy": metrics["confirmed_precision"],
        },
        "hard_negative_rows_non_sparse": int(hard_ns.sum()),
        "relative_reduction_in_non_sparse_no_fire_alert_rate": (
            1 - metrics["non_sparse_no_fire_alert_rate"] / metrics["rule_non_sparse_no_fire_alert_rate"]
            if metrics["rule_non_sparse_no_fire_alert_rate"] else None
        ),
        "rule_per_regime_alert": {
            r: _binary_slice(y, rule_alert, matrix.regimes == r, hard) for r in sorted(set(matrix.regimes.tolist()))
        },
        "policy_per_regime_alert": {
            r: _binary_slice(y, retained, matrix.regimes == r, hard) for r in sorted(set(matrix.regimes.tolist()))
        },
    }

    return to_builtin(
        {
            "synthetic_data_notice": SYNTHETIC_NOTICE,
            "task_7_binary_primary_gate": TASK_7_BINARY_PRIMARY_GATE_VERDICT,
            "verdict": POLICY_PASSED if gate["passed"] else POLICY_FAILED,
            "locked_model": {
                "family": MODEL_CLASS_NAME, "params": MODEL_PARAMS, "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V5),
                "preprocessing": describe_preprocessing_v5(MODEL_FAMILY),
                "trained_once_on": "training_v5.csv", "training_rows": training_rows,
                "confirmatory_rows_used_in_training": 0,
            },
            "locked_policy": {
                "version": POLICY_VERSION, "suspect_threshold": SUSPECT_THRESHOLD, "confirm_threshold": CONFIRM_THRESHOLD,
                "corroboration": MULTI_PIXEL_DEFINITION,
                "rules": {
                    "NO_EVENT": f"P(fire) < {SUSPECT_THRESHOLD}",
                    "SUSPECTED": f"P(fire) >= {SUSPECT_THRESHOLD} and the CONFIRMED condition is not satisfied",
                    "CONFIRMED": f"P(fire) >= {CONFIRM_THRESHOLD} AND current satellite pixel count >= 2",
                },
            },
            "policy_gate_definition": POLICY_GATE_V5,
            "confirmatory_dataset": {
                "seed": CONFIRMATORY_SEED, "rows": int(len(y)), "expected_rows": CONFIRMATORY_ROWS,
                "sha256": confirmatory_csv_sha256, "expected_sha256": CONFIRMATORY_CSV_SHA256,
                "environments": int(len(set(matrix.environments.tolist()))),
                "pairs": int(len(set(matrix.pair_ids[matrix.pair_ids >= 0].tolist()))),
                "training_sha256": training_csv_sha256, "expected_training_sha256": TRAINING_CSV_SHA256,
                "overlap_with_training": overlap,
            },
            "metrics": metrics,
            "policy_gate": gate,
            "gate_metric_uncertainty": gate_metric_uncertainty(matrix, p, codes, rule_alert),
            "outcome_matrix": {"all_rows": outcome_counts(y, codes), "non_sparse_rows": outcome_counts(y[ns], codes[ns])},
            "per_regime": status_breakdown(matrix, codes, matrix.regimes),
            "per_latent_subtype": by_subtype,
            "sparse_early_evidence": sparse_report,
            "history_progression_by_pass_count": progression,
            "history_progression_persistent_regime": progression_persistent,
            "history_progression_note": (
                "A STATIC dataset: each row is one independent observation, so this compares rows with different pass counts; "
                "it does NOT prove that one event's confidence evolves over time. Sequential runtime progression is a later test."
            ),
            "rule_comparison": comparison,
            "probability_histogram_10_bins": {
                "fire": np.histogram(p[y == 1], bins=10, range=(0, 1))[0].astype(int).tolist(),
                "no_fire": np.histogram(p[y == 0], bins=10, range=(0, 1))[0].astype(int).tolist(),
            },
            "calibration_bins": calibration["bins"],
            "meta": {
                "python_version": platform.python_version(), "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                "joblib_version": joblib.__version__,
            },
        }
    )


# ---------------------------------------------------------------------------------------------------------
# artifact: only if the Task 8 policy gate passed
# ---------------------------------------------------------------------------------------------------------


def build_artifact_metadata(report: dict, training_csv: Path, confirmatory_csv: Path, created_at: datetime) -> dict:
    return {
        "model_version": "5.0",
        "model_type": MODEL_CLASS_NAME,
        "model_name": MODEL_ARTIFACT_STEM,
        "task_7_binary_primary_model_gate": TASK_7_BINARY_PRIMARY_GATE_VERDICT,
        "task_8_ai_hybrid_policy_gate": "PASSED" if report["policy_gate"]["passed"] else "FAILED",
        "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V5),
        "feature_schema_version": "v5",
        "hyperparameters": {**MODEL_PARAMS, "max_iter": 200, "min_samples_leaf": 40, "l2_regularization": 1.0, "early_stopping": False},
        "preprocessing": describe_preprocessing_v5(MODEL_FAMILY),
        "training_dataset_sha256": dataset_sha256(training_csv),
        "confirmatory_dataset_sha256": dataset_sha256(confirmatory_csv),
        "policy_version": POLICY_VERSION,
        "suspect_threshold": SUSPECT_THRESHOLD,
        "confirm_threshold": CONFIRM_THRESHOLD,
        "multi_pixel_corroboration": MULTI_PIXEL_DEFINITION,
        "policy_gate": report["policy_gate"],
        "confirmatory_metrics": report["metrics"],
        "outcome_matrix": report["outcome_matrix"],
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "random_seed": 42,
        "synthetic_data_notice": SYNTHETIC_NOTICE,
        "known_limitations": [
            "trained and confirmed ONLY on synthetic data; not a measure of real-world wildfire detection accuracy",
            "the policy is offline: not integrated into FireDetectionAgent, the decision policy or any runtime path",
            "static confirmatory data cannot prove sequential temporal progression of one event",
            "SUSPECTED currently triggers the full response pipeline; runtime semantics must change before integration",
        ],
        "created_at": created_at.isoformat(),
    }


def save_policy_artifact(report: dict, model, training_csv: Path, confirmatory_csv: Path, model_dir: Path,
                         created_at: datetime | None = None) -> tuple[Path, Path]:
    """Save the trained HGB and its metadata - refuses unless the Task 8 policy gate PASSED and the data are the frozen files."""
    if not report["policy_gate"]["passed"] or report["verdict"] != POLICY_PASSED:
        raise ArtifactRefusedError("the AI hybrid policy did not pass its gate; refusing to save a runtime artifact.")
    if dataset_sha256(training_csv) != TRAINING_CSV_SHA256:
        raise ArtifactRefusedError("training CSV is not the frozen V5 benchmark.")
    if dataset_sha256(confirmatory_csv) != CONFIRMATORY_CSV_SHA256 or report["confirmatory_dataset"]["sha256"] != CONFIRMATORY_CSV_SHA256:
        raise ArtifactRefusedError("confirmatory CSV is not the locked confirmatory dataset.")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_ARTIFACT_STEM}.joblib"
    metadata_path = model_dir / f"{MODEL_ARTIFACT_STEM}_metadata.json"
    for path in (model_path, metadata_path):
        if not path.name.startswith("fire_detection_hgb_v5"):
            raise ArtifactRefusedError(f"refusing to write {path.name!r}")
        if path.exists():
            raise ArtifactRefusedError(f"{path.name} already exists; V5 artifacts are never silently overwritten.")
    metadata = build_artifact_metadata(report, training_csv, confirmatory_csv, created_at or datetime.now(timezone.utc))
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return model_path, metadata_path


def reload_parity_cases(matrix: TrainingMatrixV5) -> dict[str, np.ndarray]:
    """Row selections covering the evidence situations the runtime will see (indices into `matrix`)."""
    sat = satellite_pixel_counts(matrix)
    news = sum(matrix.feature(n) for n in ("news_none_count", "news_weak_count", "news_moderate_count", "news_strong_count", "news_unknown_count"))
    passes = matrix.feature("satellite_pass_count")
    cases = {
        "news_only": (sat == 0) & (news > 0),
        "one_hotspot": (sat == 1) & (news == 0) & (passes == 1),
        "multi_pixel_candidate": sat >= 2,
        "satellite_and_news": (sat > 0) & (news > 0),
        "one_pass_event": (passes == 1),
        "three_pass_event": passes >= 3,
        "sparse_ambiguous_evidence": matrix.regimes == SPARSE_REGIME,
        "nullable_history_features_nan": np.isnan(matrix.feature("satellite_centroid_stability_km")),
    }
    return {name: np.flatnonzero(mask)[:50] for name, mask in cases.items()}


def reload_parity(model, reloaded, matrix: TrainingMatrixV5) -> dict[str, float]:
    """Largest absolute probability difference per evidence situation between the in-memory model and its reload (must be 0)."""
    result = {}
    for name, indices in reload_parity_cases(matrix).items():
        if len(indices) == 0:
            raise ValueError(f"no rows available for the parity case {name!r}")
        X = matrix.X[indices]
        result[name] = float(np.max(np.abs(model.predict_proba(X)[:, 1] - reloaded.predict_proba(X)[:, 1])))
    return result
