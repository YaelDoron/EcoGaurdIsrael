"""Task 7: train, evaluate and (only if the predefined gate passes) select the V5 Fire Detection model.

    run_model_comparison_v5(matrix, samples) -> report (JSON-serialisable dict)

Order of decisions, all made on GROUPED-ENVIRONMENT evaluation only:

  1. per family, choose ONE hyperparameter setting from the small predefined grid by grouped-environment mean
     ROC-AUC (stress tests, ablations, thresholds and the rule baseline never influence this choice);
  2. per family, assess sigmoid / isotonic calibration with NESTED grouped out-of-fold predictions and adopt it
     only under the predefined rule (config module); the default is no calibration;
  3. evaluate everything else (thresholds, paired, sparse, stress tests, ablations, importance, audits) for that
     single fixed candidate per family;
  4. apply the predefined gate (config module) and select - or select NONE.

All data is synthetic. Nothing here integrates V5 into runtime.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from src.ml.fire_detection.fire_detection_evaluation_v5 import (
    CALIBRATION_VARIANTS,
    NULLABLE_AUDIT_FEATURES,
    NamedSplit,
    assert_folds_represent_regimes_and_labels,
    assert_pairs_never_split,
    assert_splits_are_grouped,
    binary_metrics,
    calibration_report,
    clustered_bootstrap_auc_difference,
    confirmed_probability_analysis,
    corroboration_analysis,
    environment_splits,
    evaluate_gate,
    grouped_permutation_importance,
    hard_negative_mask,
    leave_group_out_report,
    logistic_coefficient_report,
    make_fit_predict,
    missingness_association,
    missingness_permutation_audit,
    model_slice_at_threshold,
    no_fire_subtype_splits,
    non_sparse_mask,
    operating_point_report,
    out_of_fold_scores,
    paired_analysis,
    per_split_metrics,
    random_split_diagnostic,
    regime_report,
    regime_splits,
    rule_baseline_outputs,
    rule_baseline_report,
    select_model,
    slice_metrics_at_matched_recall,
    sparse_evidence_report,
    summarize_folds,
    suspected_threshold,
    to_builtin,
    tree_importance_report,
    verify_samples_match_matrix,
)
from src.ml.fire_detection.fire_detection_model_config_v5 import (
    CALIBRATION_ADOPT_MIN_ECE_IMPROVEMENT,
    CALIBRATION_MAX_ROC_AUC_LOSS,
    CANDIDATE_GRID_V5,
    DATASET_VERSION_V5,
    DEFAULT_THRESHOLD,
    EXPECTED_TRAINING_CSV_SHA256_V5,
    FEATURE_SETS_V5,
    FIXED_HYPERPARAMETERS_V5,
    GATE_V5,
    MODEL_FAMILIES_V5,
    PERMUTATION_REPEATS,
    BOOTSTRAP_REPEATS,
    PERSISTENT_REGIME,
    RANDOM_SEED_V5,
    SPARSE_REGIME,
)
from src.ml.fire_detection.fire_detection_model_v5 import (
    TrainingMatrixV5,
    assert_feature_schema_v5,
    build_pipeline_v5,
    describe_preprocessing_v5,
    transformed_feature_names,
)

SYNTHETIC_DATA_NOTICE = (
    "All training and evaluation data is SYNTHETIC (backend/docs/fire_detection_dataset_v5.md). These metrics "
    "describe behaviour on the frozen V5 benchmark and are NOT real-world wildfire detection accuracy."
)
DATASET_ORACLE_NOTE = (
    "No generator / Bayes oracle probability exists: the frozen V5 generator (Task 6) persists only the latent "
    "subtype and label, not a posterior P(fire | features). It was not modified to create one, so no oracle "
    "ceiling is available; the predefined acceptance gate and the rule baseline are used instead."
)


def _noop(_message: str) -> None:
    return None


def _grouped_summary(y: np.ndarray, scores: np.ndarray, splits, threshold: float = DEFAULT_THRESHOLD) -> dict:
    folds = per_split_metrics(y, scores, splits, threshold)
    return {"per_fold": folds, "summary": summarize_folds(folds)}


def _feature_names_for(set_name: str) -> tuple[str, ...]:
    return FEATURE_SETS_V5[set_name]


class _FamilyRunner:
    """Everything evaluated for ONE model family; keeps the fixed candidate (params + calibration variant)."""

    def __init__(self, matrix: TrainingMatrixV5, splits: tuple[NamedSplit, ...], family: str, log: Callable[[str], None]) -> None:
        self.matrix, self.splits, self.family, self.log = matrix, splits, family, log

    def fit_predict(self, params: dict, feature_set: str = "full_v5", variant: str = "none"):
        names = _feature_names_for(feature_set)
        columns = self.matrix.columns(names)
        return make_fit_predict(
            lambda: build_pipeline_v5(self.family, params, names), columns, self.matrix.y, self.matrix.environments, variant
        )


def _config_selection(runner: _FamilyRunner, grid: tuple[dict, ...]) -> tuple[list[dict], dict, np.ndarray, list]:
    matrix, splits = runner.matrix, runner.splits
    results, best = [], None
    for params in grid:
        scores, fitted = out_of_fold_scores(runner.fit_predict(params), len(matrix.y), splits)
        summary = summarize_folds(per_split_metrics(matrix.y, scores, splits))
        results.append({"params": params, "grouped_roc_auc_mean": summary["roc_auc"]["mean"], "grouped_roc_auc_min": summary["roc_auc"]["min"],
                        "grouped_brier_mean": summary["brier"]["mean"]})
        if best is None or summary["roc_auc"]["mean"] > best[0] + 1e-12:  # ties keep the earlier (simpler) setting
            best = (summary["roc_auc"]["mean"], params, scores, fitted)
        runner.log(f"  {runner.family} {params}: grouped ROC-AUC {summary['roc_auc']['mean']:.4f}")
    return results, best[1], best[2], best[3]


def _calibration_assessment(runner: _FamilyRunner, params: dict, base_scores: np.ndarray, base_fitted: list) -> dict:
    """Nested grouped out-of-fold assessment of calibration; adoption follows the predefined rule."""
    matrix, splits = runner.matrix, runner.splits
    variants = {"none": (base_scores, base_fitted)}
    for variant in CALIBRATION_VARIANTS[1:]:
        variants[variant] = out_of_fold_scores(runner.fit_predict(params, variant=variant), len(matrix.y), splits)
    assessed = {}
    for variant, (scores, _) in variants.items():
        report = calibration_report(matrix.y, scores)
        folds = summarize_folds(per_split_metrics(matrix.y, scores, splits))
        assessed[variant] = {
            "brier_score": report["brier_score"],
            "expected_calibration_error": report["expected_calibration_error"],
            "grouped_roc_auc_mean": folds["roc_auc"]["mean"],
            "reliability_bins": report["bins"],
        }
    baseline = assessed["none"]
    adopted, reason = "none", "no calibration variant improved ECE enough without costing ranking; left uncalibrated"
    best_gain = 0.0
    for variant in CALIBRATION_VARIANTS[1:]:
        gain = baseline["expected_calibration_error"] - assessed[variant]["expected_calibration_error"]
        loss = baseline["grouped_roc_auc_mean"] - assessed[variant]["grouped_roc_auc_mean"]
        if gain >= CALIBRATION_ADOPT_MIN_ECE_IMPROVEMENT and loss <= CALIBRATION_MAX_ROC_AUC_LOSS and gain > best_gain:
            adopted, best_gain = variant, gain
            reason = f"{variant} improves ECE by {gain:.4f} with a grouped ROC-AUC change of {-loss:+.4f}"
    return {
        "method": "CalibratedClassifierCV fit on TRAINING rows only (inner grouped-by-environment folds); "
        "the held-out fold is never used to calibrate",
        "variants": assessed,
        "adopted_variant": adopted,
        "adoption_rule": {
            "min_ece_improvement": CALIBRATION_ADOPT_MIN_ECE_IMPROVEMENT,
            "max_roc_auc_loss": CALIBRATION_MAX_ROC_AUC_LOSS,
        },
        "adoption_reason": reason,
    }, variants[adopted]


def _ablations(
    runner: _FamilyRunner, params: dict, variant: str, full_scores: np.ndarray, full_threshold: float | None,
    bootstrap_repeats: int, feature_sets: dict[str, tuple[str, ...]],
) -> dict:
    matrix, splits = runner.matrix, runner.splits
    y = matrix.y
    persistent = matrix.regimes == PERSISTENT_REGIME
    ns = non_sparse_mask(matrix)
    hard = hard_negative_mask(matrix) & ns
    scores_by_set = {"full_v5": full_scores}
    for name in feature_sets:
        if name != "full_v5":
            scores_by_set[name] = out_of_fold_scores(runner.fit_predict(params, name, variant), len(y), splits)[0]

    def describe(name: str) -> dict:
        scores = scores_by_set[name]
        folds = summarize_folds(per_split_metrics(y, scores, splits))
        threshold = suspected_threshold(y, scores, ns & (y == 1))
        entry = {
            "feature_count": len(feature_sets[name]),
            "grouped_environment": {k: folds[k] for k in ("roc_auc", "pr_auc", "brier", "f1")},
            "overall_brier_ece": {k: v for k, v in calibration_report(y, scores).items() if k in ("brier_score", "expected_calibration_error")},
            "suspected_threshold": threshold,
            "hard_negative_fpr_at_own_suspected_threshold": (
                float(np.mean(scores[hard] >= threshold)) if threshold is not None and hard.any() else None
            ),
        }
        p_scores, p_y = scores[persistent], y[persistent]
        entry["persistent_thermal"] = {
            "roc_auc": float(binary_metrics(p_y, p_scores)["roc_auc"]),
            "per_fold_roc_auc": [
                float(binary_metrics(y[s.validation][persistent[s.validation]], scores[s.validation][persistent[s.validation]])["roc_auc"])
                for s in splits
            ],
            "at_0_50": {k: binary_metrics(p_y, p_scores, DEFAULT_THRESHOLD)[k] for k in ("precision", "recall", "f1", "false_positive_rate")},
            "at_own_suspected_threshold": (
                None if threshold is None else
                {k: binary_metrics(p_y, p_scores, threshold)[k] for k in ("precision", "recall", "f1", "false_positive_rate")}
            ),
            "matched_80_percent_persistent_recall": slice_metrics_at_matched_recall(p_y, p_scores),
        }
        return entry

    described = {name: describe(name) for name in scores_by_set}
    comparisons = {}
    pairs = {
        "history_effect_full_vs_without_history": ("full_v5", "without_history"),
        "current_additions_effect_without_history_vs_retained_14": ("without_history", "retained_14_baseline"),
        "total_effect_full_vs_retained_14": ("full_v5", "retained_14_baseline"),
    }
    for label, (a, b) in pairs.items():
        if a not in scores_by_set or b not in scores_by_set:
            continue
        comparisons[label] = {
            "overall_grouped": clustered_bootstrap_auc_difference(y, scores_by_set[a], scores_by_set[b], matrix.environments, bootstrap_repeats),
            "persistent_thermal": clustered_bootstrap_auc_difference(
                y[persistent], scores_by_set[a][persistent], scores_by_set[b][persistent], matrix.environments[persistent], bootstrap_repeats
            ),
            "non_persistent_rows": clustered_bootstrap_auc_difference(
                y[~persistent], scores_by_set[a][~persistent], scores_by_set[b][~persistent], matrix.environments[~persistent], bootstrap_repeats
            ),
        }
    return {
        "feature_sets": {k: list(v) for k, v in feature_sets.items()},
        "ladder": "retained_14_baseline -> without_history (14 + 6 current additions) -> full_v5 (+ 5 history)",
        "results": described,
        "comparisons": comparisons,
    }


def evaluate_family(
    matrix: TrainingMatrixV5,
    splits: tuple[NamedSplit, ...],
    family: str,
    rule: dict | None,
    *,
    grid: tuple[dict, ...] | None = None,
    permutation_repeats: int = PERMUTATION_REPEATS,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
    run_stress_tests: bool = True,
    run_ablations: bool = True,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    log: Callable[[str], None] = _noop,
) -> dict:
    """Full evaluation of one family's single fixed candidate; returns its report section (with gate + selection inputs)."""
    runner = _FamilyRunner(matrix, splits, family, log)
    y, ns = matrix.y, non_sparse_mask(matrix)
    hard = hard_negative_mask(matrix) & ns
    persistent = matrix.regimes == PERSISTENT_REGIME

    config_results, params, base_scores, base_fitted = _config_selection(runner, grid or CANDIDATE_GRID_V5[family])
    log(f"  {family}: selected {params}")
    calibration, (scores, fitted) = _calibration_assessment(runner, params, base_scores, base_fitted)
    variant = calibration["adopted_variant"]
    fit_predict = runner.fit_predict(params, variant=variant)
    log(f"  {family}: calibration adopted = {variant}")

    grouped = _grouped_summary(y, scores, splits)
    threshold = suspected_threshold(y, scores, ns & (y == 1))
    operating = operating_point_report(matrix, scores, threshold) if threshold is not None else None
    fold_at_threshold = (
        [{"split": s.name, **binary_metrics(y[s.validation], scores[s.validation], threshold),
          "non_sparse_positive_recall": float(np.mean(scores[s.validation][(y[s.validation] == 1) & ns[s.validation]] >= threshold)),
          "hard_negative_fpr": (float(np.mean(scores[s.validation][hard[s.validation]] >= threshold)) if hard[s.validation].any() else None)}
         for s in splits]
        if threshold is not None else None
    )
    overall_calibration = calibration_report(y, scores)
    persistent_auc = float(binary_metrics(y[persistent], scores[persistent])["roc_auc"])

    thresholds_of_interest = [DEFAULT_THRESHOLD] + ([threshold] if threshold is not None else [])
    section: dict = {
        "model_key": family,
        "preprocessing": describe_preprocessing_v5(family),
        "fixed_hyperparameters": FIXED_HYPERPARAMETERS_V5[family],
        "hyperparameter_grid_results": config_results,
        "selected_params": params,
        "selection_criterion": "highest grouped-environment mean ROC-AUC; ties keep the earlier (simpler) setting; no stress test used",
        "calibration": {**calibration, "overall_out_of_fold": overall_calibration},
        "grouped_environment": {**grouped, "at_threshold": DEFAULT_THRESHOLD, "per_fold_at_suspected_threshold": fold_at_threshold},
        "regime_out_of_fold_at_0_50": regime_report(matrix, scores, DEFAULT_THRESHOLD),
        "suspected_operating_point": operating,
        "confirmed_probability_analysis": confirmed_probability_analysis(matrix, scores),
        "paired_cases": paired_analysis(matrix, scores),
        "sparse_early_evidence": sparse_evidence_report(matrix, scores),
    }

    confirm = section["confirmed_probability_analysis"]
    corroboration_thresholds = [t for t in ([threshold] if threshold else []) + [0.60, 0.70, 0.80]]
    if confirm["candidate"]:
        corroboration_thresholds.append(confirm["candidate"]["threshold"])
    section["corroboration"] = corroboration_analysis(matrix, scores, corroboration_thresholds)

    if run_stress_tests:
        log(f"  {family}: stress tests")
        section["random_split_diagnostic"] = random_split_diagnostic(matrix, fit_predict, grouped["summary"]["roc_auc"]["mean"])
        section["leave_one_regime_out"] = {
            "note": "persistent_thermal is the only regime with history: holding it out leaves the model NO multi-pass "
            "examples, a severe out-of-distribution stress test - interpreted, never a hard criterion.",
            **leave_group_out_report(matrix, regime_splits(matrix), fit_predict, thresholds_of_interest),
        }
        section["leave_one_no_fire_subtype_out"] = leave_group_out_report(
            matrix, no_fire_subtype_splits(matrix), fit_predict, thresholds_of_interest
        )

    log(f"  {family}: importance and audits")
    names = matrix.feature_names
    columns = matrix.X
    permutation = grouped_permutation_importance(fitted, columns, y, splits, names, permutation_repeats)
    pipelines = [m for m in fitted]
    representative = fitted[0].calibrated_classifiers_[0].estimator if hasattr(fitted[0], "calibrated_classifiers_") else fitted[0]
    transformed = transformed_feature_names(representative, names)
    section["feature_importance"] = {
        "grouped_permutation": permutation,
        "logistic_coefficients": logistic_coefficient_report(pipelines, transformed),
        "tree_importances": tree_importance_report(pipelines, transformed),
    }
    section["missingness_audit"] = {
        "univariate_association": missingness_association(matrix),
        "model_use": missingness_permutation_audit(fitted, columns, y, splits, names),
    }

    if run_ablations:
        log(f"  {family}: ablations")
        section["ablations"] = _ablations(
            runner, params, variant, scores, threshold, bootstrap_repeats, feature_sets or FEATURE_SETS_V5
        )

    non_sparse_slice = model_slice_at_threshold(matrix, scores, threshold, ns) if threshold is not None else None
    grouped_auc = grouped["summary"]["roc_auc"]
    candidate = {
        "model_key": family,
        "grouped_mean_roc_auc": grouped_auc["mean"],
        "grouped_worst_fold_roc_auc": grouped_auc["min"],
        "suspected_threshold": threshold,
        "non_sparse_positive_recall": operating["non_sparse_positive_recall"] if operating else None,
        "hard_negative_fpr": operating["hard_negative_fpr"] if operating else None,
        "persistent_roc_auc": persistent_auc,
        "brier_score": overall_calibration["brier_score"],
        "expected_calibration_error": overall_calibration["expected_calibration_error"],
        "top_feature_share": permutation["top_feature_share"],
        "non_sparse_slice_at_suspected": non_sparse_slice,
    }
    if rule is not None:
        candidate["gate"] = evaluate_gate(candidate, rule["suspected_or_confirmed_is_fire"])
        if non_sparse_slice is not None:
            section["versus_rules_non_sparse_rows"] = {
                "model_at_suspected_threshold": non_sparse_slice,
                "rules_suspected_or_confirmed": rule["suspected_or_confirmed_is_fire"]["non_sparse_rows"],
                "hard_negative_fpr_reduction": (
                    1 - non_sparse_slice["hard_negative_fpr"] / rule["suspected_or_confirmed_is_fire"]["non_sparse_rows"]["hard_negative_fpr"]
                    if rule["suspected_or_confirmed_is_fire"]["non_sparse_rows"]["hard_negative_fpr"] else None
                ),
                "recall_change": non_sparse_slice["recall"] - rule["suspected_or_confirmed_is_fire"]["non_sparse_rows"]["recall"],
            }
    else:
        candidate["gate"] = {"passed": False, "failed_criteria": ["rule baseline unavailable"], "criteria": {}}
    section["gate_inputs"] = {k: v for k, v in candidate.items() if k not in ("gate",)}
    section["acceptance_gate_result"] = candidate["gate"]
    section["_candidate"] = candidate
    return section


def run_model_comparison_v5(
    matrix: TrainingMatrixV5,
    samples=None,
    *,
    families: tuple[str, ...] = MODEL_FAMILIES_V5,
    grids: dict[str, tuple[dict, ...]] | None = None,
    permutation_repeats: int = PERMUTATION_REPEATS,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
    run_stress_tests: bool = True,
    run_ablations: bool = True,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    csv_sha256: str | None = None,
    log: Callable[[str], None] = _noop,
) -> dict:
    """The complete Task 7 comparison. `samples` (regenerated from the frozen generator) enable the rule baseline."""
    assert_feature_schema_v5(matrix.feature_names)
    splits = environment_splits(matrix)
    assert_splits_are_grouped(splits, matrix.environments)
    assert_pairs_never_split(matrix, splits)
    assert_folds_represent_regimes_and_labels(matrix, splits)

    rule = None
    if samples is not None:
        verify_samples_match_matrix(samples, matrix)
        rule = rule_baseline_report(matrix, rule_baseline_outputs(samples))
        log("rule baseline evaluated")

    sections = {}
    for family in families:
        log(f"evaluating {family}")
        sections[family] = evaluate_family(
            matrix, splits, family, rule, grid=(grids or {}).get(family), permutation_repeats=permutation_repeats,
            bootstrap_repeats=bootstrap_repeats, run_stress_tests=run_stress_tests, run_ablations=run_ablations,
            feature_sets=feature_sets, log=log,
        )
    candidates = [section.pop("_candidate") for section in sections.values()]
    selection = select_model(candidates)

    import platform
    import joblib
    import sklearn

    return to_builtin(
        {
            "synthetic_data_notice": SYNTHETIC_DATA_NOTICE,
            "meta": {
                "dataset_version": DATASET_VERSION_V5,
                "training_csv_sha256": csv_sha256,
                "expected_training_csv_sha256": EXPECTED_TRAINING_CSV_SHA256_V5,
                "rows": int(len(matrix.y)),
                "feature_names": list(matrix.feature_names),
                "seed": RANDOM_SEED_V5,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "joblib_version": joblib.__version__,
            },
            "acceptance_gate": GATE_V5,
            "candidate_grid": CANDIDATE_GRID_V5,
            "primary_evaluation": "grouped by environment_id (5 folds; whole environments and all pair members held out together)",
            "environment_fold_layout": [
                {"fold": s.name, "rows": int(len(s.validation)), "fire_rows": int(matrix.y[s.validation].sum()),
                 "environments": int(len(set(matrix.environments[s.validation].tolist())))}
                for s in splits
            ],
            "rule_baseline": rule,
            "models": sections,
            "dataset_oracle": DATASET_ORACLE_NOTE,
            "selection": selection,
        }
    )
