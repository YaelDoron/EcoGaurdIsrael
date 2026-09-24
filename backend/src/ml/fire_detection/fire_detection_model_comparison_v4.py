"""Assemble the Fire Detection ML V4 model comparison report (offline only).

Orchestrates fire_detection_evaluation_v4 over the frozen V4 dataset:
grouped-family CV (PRIMARY), leave-one-archetype-out, random split (diagnostic),
rule baseline, hard-negative / weak-positive analysis, Fire Danger and
feature-group ablations, thresholds, calibration, interpretability, the
generalisation gate and the model selection.

The report contains no timestamp, so re-running it reproduces it exactly
(fixed seeds, single-threaded models).
"""
from __future__ import annotations

from pathlib import Path
import platform
import sys
from typing import Sequence

import joblib
import numpy as np
import sklearn

from src.ml.fire_detection.fire_detection_evaluation_v4 import (
    CANDIDATE_THRESHOLDS,
    GATE_V4,
    SUSPECT_RECALL_TARGET,
    CONFIRM_PRECISION_TARGET,
    archetype_splits,
    assert_splits_are_grouped,
    binary_metrics,
    calibrated_cross_validated_scores,
    calibration_report,
    cross_validated_scores,
    evaluate_gate,
    family_splits,
    grouped_permutation_importance,
    importance_shortcut_summary,
    leave_one_family_out_splits,
    logistic_coefficient_report,
    paired_differences,
    per_family_accuracy,
    per_split_metrics,
    random_split_diagnostic,
    recommend_thresholds,
    rule_baseline_outputs,
    rule_baseline_report,
    select_model,
    subgroup_performance,
    summarize_folds,
    threshold_table,
    to_builtin,
    tree_importance_report,
    verify_samples_match_matrix,
)
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4
from src.ml.fire_detection.fire_detection_model_v4 import (
    DATASET_VERSION_V4,
    FEATURE_GROUPS_V4,
    MODEL_SPECS_V4,
    RANDOM_SEED_V4,
    SENSITIVITY_VARIANTS_V4,
    TrainingMatrixV4,
    build_pipeline_v4,
    dataset_sha256,
    describe_preprocessing_v4,
    feature_names_without,
)

# A Fire Danger verdict is a mechanical rule on the paired grouped-fold differences, stated up front.
FIRE_DANGER_MATERIAL_AUC_DELTA = 0.01
FIRE_DANGER_MIN_FOLDS_IMPROVED = 4  # of 5


def environment_info(csv_path: Path | None) -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "random_seed": RANDOM_SEED_V4,
        "training_dataset_sha256": dataset_sha256(csv_path) if csv_path else None,
    }


def ablation_subsets() -> dict[str, tuple[str, ...]]:
    return {
        "full_19": FIRE_DETECTION_FEATURE_NAMES_V4,
        "no_fire_danger": feature_names_without(FEATURE_GROUPS_V4["fire_danger"]),
        "no_news": feature_names_without(FEATURE_GROUPS_V4["news"]),
        "no_satellite_frp_brightness": feature_names_without(FEATURE_GROUPS_V4["satellite_frp_brightness"]),
        "no_geometry_time": feature_names_without(FEATURE_GROUPS_V4["geometry_time"]),
    }


def fire_danger_verdict(paired: dict, n_folds: int) -> dict:
    """Mechanical verdict on whether Fire Danger improved grouped generalisation (full minus no-Fire-Danger)."""
    auc = paired["roc_auc"]
    required = min(FIRE_DANGER_MIN_FOLDS_IMPROVED, n_folds)
    if auc["mean_delta"] >= FIRE_DANGER_MATERIAL_AUC_DELTA and auc["folds_improved"] >= required:
        verdict = "improved"
    elif auc["mean_delta"] <= -FIRE_DANGER_MATERIAL_AUC_DELTA:
        verdict = "worse"
    else:
        verdict = "no material effect"
    return {
        "verdict": verdict,
        "rule": (
            f"'improved' needs a mean ROC-AUC gain >= {FIRE_DANGER_MATERIAL_AUC_DELTA} AND a gain in >= {required} "
            f"of {n_folds} grouped folds; 'worse' is a mean loss >= {FIRE_DANGER_MATERIAL_AUC_DELTA}"
        ),
    }


def run_model_comparison_v4(
    matrix: TrainingMatrixV4,
    samples,
    csv_path: Path | None = None,
    model_keys: Sequence[str] = tuple(MODEL_SPECS_V4),
    n_family_folds: int = 5,
    permutation_repeats: int = 5,
    include_sensitivity: bool = True,
    include_calibration_variants: bool = True,
    include_leave_one_family_out: bool = True,
) -> dict:
    """Run the whole comparison and return the JSON-serialisable report."""
    verify_samples_match_matrix(samples, matrix)
    fsplits = family_splits(matrix, n_family_folds)
    asplits = archetype_splits(matrix)
    lofo_splits = leave_one_family_out_splits(matrix)
    assert_splits_are_grouped(matrix, fsplits, matrix.families)
    assert_splits_are_grouped(matrix, asplits, matrix.archetypes)
    y = matrix.y

    # ---- rule baseline ----
    rule_outputs = rule_baseline_outputs(samples)
    rule = rule_baseline_report(y, rule_outputs, fsplits)
    rule_fire = rule_outputs.predicted_fire
    rule_summary = {
        "grouped_mean_f1": rule["suspected_or_confirmed_is_fire"]["summary"]["f1"]["mean"],
        "overall_precision": rule["suspected_or_confirmed_is_fire"]["overall"]["precision"],
    }
    rule["subgroups"] = {
        "suspected_or_confirmed_is_fire": subgroup_performance(matrix, rule_fire),
        "confirmed_only_is_fire": subgroup_performance(matrix, rule_outputs.predicted_confirmed),
    }

    models_report: dict[str, dict] = {}
    gate_inputs: list[dict] = []
    for key in model_keys:
        spec = MODEL_SPECS_V4[key]
        oof_family, fitted = cross_validated_scores(key, matrix, fsplits, keep_models=True)
        oof_archetype, _ = cross_validated_scores(key, matrix, asplits)
        family_folds = per_split_metrics(y, oof_family, fsplits)
        archetype_folds = per_split_metrics(y, oof_archetype, asplits)
        family_summary = summarize_folds(family_folds)
        archetype_summary = summarize_folds(archetype_folds)

        supplementary = None
        if include_leave_one_family_out:
            oof_lofo, _ = cross_validated_scores(key, matrix, lofo_splits)
            supplementary = {
                "note": (
                    "SUPPLEMENTARY diagnostic, not used for the gate or selection: one family held out at a time "
                    "(its sibling families stay in training). Pooled out-of-fold metrics from 20 different models; "
                    "each held-out group has a single label, which biases this design against the held-out family "
                    "(removing it makes its evidence look more like the opposite label)."
                ),
                "pooled_out_of_fold": binary_metrics(y, oof_lofo, 0.5),
                "subgroups_at_threshold_0.5": subgroup_performance(matrix, (oof_lofo >= 0.5).astype(float)),
            }

        recommendation = recommend_thresholds(y, oof_family, fsplits)
        suspect_t = recommendation["suspect_threshold_candidate"]
        confirm_t = recommendation["confirm_threshold_candidate"]
        at_default = subgroup_performance(matrix, (oof_family >= 0.5).astype(float))
        at_suspect = subgroup_performance(matrix, (oof_family >= suspect_t).astype(float)) if suspect_t is not None else None
        at_confirm = subgroup_performance(matrix, (oof_family >= confirm_t).astype(float)) if confirm_t is not None else None

        calibration = calibration_report(y, oof_family)
        permutation = grouped_permutation_importance(fitted, matrix, fsplits, n_repeats=permutation_repeats)
        shortcuts = importance_shortcut_summary(permutation)

        interpretability = {"permutation_importance_grouped": permutation, "shortcut_summary": shortcuts}
        if key == "logistic_regression":
            interpretability["standardized_coefficients_grouped_folds"] = logistic_coefficient_report(fitted, matrix.feature_names)
        full_fit = build_pipeline_v4(key, matrix.feature_names).fit(matrix.X, y)
        tree_importance = tree_importance_report(full_fit, matrix.feature_names)
        if tree_importance:
            interpretability["impurity_importance_full_fit"] = tree_importance
        if key == "logistic_regression":
            interpretability["standardized_coefficients_full_fit"] = logistic_coefficient_report([full_fit], matrix.feature_names)

        models_report[key] = {
            "display_name": spec.display_name,
            "preprocessing": describe_preprocessing_v4(key),
            "random_split_diagnostic": random_split_diagnostic(key, matrix),
            "grouped_family_cv": {
                "primary_selection_metric": True,
                "threshold": 0.5,
                "per_fold": family_folds,
                "summary": family_summary,
                "pooled_out_of_fold": binary_metrics(y, oof_family, 0.5),
            },
            "leave_one_archetype_out": {"threshold": 0.5, "per_archetype": archetype_folds, "summary": archetype_summary},
            "leave_one_family_out_supplementary": supplementary,
            "subgroups_at_threshold_0.5": at_default,
            "thresholds": {
                "candidate_grid_out_of_fold_grouped": threshold_table(y, oof_family, CANDIDATE_THRESHOLDS),
                "recommendation": recommendation,
                "subgroups_at_suspect_threshold": at_suspect,
                "subgroups_at_confirm_threshold": at_confirm,
            },
            "calibration": {"uncalibrated_out_of_fold": calibration},
            "interpretability": interpretability,
        }

        gate_inputs.append(
            {
                "model_key": key,
                "grouped_mean_roc_auc": family_summary["roc_auc"]["mean"],
                "grouped_min_fold_roc_auc": family_summary["roc_auc"]["min"],
                "grouped_mean_f1": family_summary["f1"]["mean"],
                "suspect_operating_point": (recommendation["suspect_operating_point"] or {}).get("overall"),
                "hard_negative_fpr_at_suspect": at_suspect["hard_negative_false_positive_rate"] if at_suspect else None,
                "weak_positive_recall_at_suspect": at_suspect["weak_positive_recall"] if at_suspect else None,
                "archetype_mean_roc_auc": archetype_summary["roc_auc"]["mean"],
                "archetype_worst_roc_auc": archetype_summary["roc_auc"]["min"],
                "calibration_ece": calibration["expected_calibration_error"],
                "top_feature_share": shortcuts["top_feature_share"],
                "geometry_time_share": shortcuts["geometry_time_share"],
            }
        )
        models_report[key]["_oof_family"] = oof_family  # stripped below; kept for the calibration/ablation steps

    # ---- gate + selection ----
    for candidate in gate_inputs:
        gate = evaluate_gate(candidate, rule_summary)
        candidate["gate_passed"] = gate["passed"]
        models_report[candidate["model_key"]]["generalization_gate"] = gate
    selection = select_model(gate_inputs)

    # ---- Fire Danger and feature-group ablations (grouped-family CV on the same folds) ----
    subsets = ablation_subsets()
    ablation: dict[str, dict] = {}
    for key in model_keys:
        per_subset = {}
        full_folds = models_report[key]["grouped_family_cv"]["per_fold"]
        for subset_name, names in subsets.items():
            if subset_name == "full_19":
                per_subset[subset_name] = {"features": len(names), "summary": models_report[key]["grouped_family_cv"]["summary"]}
                continue
            oof, _ = cross_validated_scores(key, matrix, fsplits, feature_names=names)
            folds = per_split_metrics(y, oof, fsplits)
            entry = {
                "features": len(names),
                "removed": sorted(set(FIRE_DETECTION_FEATURE_NAMES_V4) - set(names)),
                "summary": summarize_folds(folds),
                "paired_difference_full_minus_reduced": paired_differences(full_folds, folds),
            }
            if subset_name == "no_fire_danger":
                full_accuracy = per_family_accuracy(matrix, models_report[key]["_oof_family"])
                reduced_accuracy = per_family_accuracy(matrix, oof)
                deltas = {family: full_accuracy[family] - reduced_accuracy[family] for family in full_accuracy}
                entry["per_family_accuracy_delta_full_minus_reduced"] = deltas
                entry["families_improved_by_fire_danger"] = sorted(f for f, d in deltas.items() if d > 0.01)
                entry["families_hurt_by_fire_danger"] = sorted(f for f, d in deltas.items() if d < -0.01)
                entry["verdict"] = fire_danger_verdict(entry["paired_difference_full_minus_reduced"], len(fsplits))
                oof_arch_reduced, _ = cross_validated_scores(key, matrix, asplits, feature_names=names)
                arch_full = models_report[key]["leave_one_archetype_out"]["per_archetype"]
                arch_reduced = per_split_metrics(y, oof_arch_reduced, asplits)
                entry["leave_one_archetype_out"] = {
                    "reduced_summary": summarize_folds(arch_reduced),
                    "full_summary": models_report[key]["leave_one_archetype_out"]["summary"],
                    "paired_difference_full_minus_reduced": paired_differences(arch_full, arch_reduced),
                }
            per_subset[subset_name] = entry
        ablation[key] = per_subset

    # ---- calibration variants (nested, grouped) for every model ----
    if include_calibration_variants:
        for key in model_keys:
            variants = {}
            for method in ("sigmoid", "isotonic"):
                oof = calibrated_cross_validated_scores(key, matrix, fsplits, method)
                folds = per_split_metrics(y, oof, fsplits)
                variants[method] = {
                    "calibration": calibration_report(y, oof),
                    "grouped_family_cv_summary": summarize_folds(folds),
                }
            base = models_report[key]["calibration"]["uncalibrated_out_of_fold"]
            best_method = min(variants, key=lambda m: variants[m]["calibration"]["brier_score"])
            base_auc = models_report[key]["grouped_family_cv"]["summary"]["roc_auc"]["mean"]
            best_auc = variants[best_method]["grouped_family_cv_summary"]["roc_auc"]["mean"]
            improves_brier = variants[best_method]["calibration"]["brier_score"] <= base["brier_score"] - 0.005
            keeps_auc = best_auc >= base_auc - 0.005
            variants["assessment"] = {
                "best_method_by_brier": best_method,
                "brier_uncalibrated": base["brier_score"],
                "brier_best_calibrated": variants[best_method]["calibration"]["brier_score"],
                "ece_uncalibrated": base["expected_calibration_error"],
                "ece_best_calibrated": variants[best_method]["calibration"]["expected_calibration_error"],
                "roc_auc_uncalibrated": base_auc,
                "roc_auc_best_calibrated": best_auc,
                "materially_improves_calibration": bool(improves_brier),
                "preserves_grouped_generalization": bool(keeps_auc),
                "recommend_calibration": bool(improves_brier and keeps_auc),
                "rule": "recommend only if Brier improves by >= 0.005 AND grouped ROC-AUC does not drop by more than 0.005",
            }
            models_report[key]["calibration"]["calibrated_variants_nested_grouped"] = variants

    # ---- hyperparameter sensitivity ----
    sensitivity = {}
    if include_sensitivity:
        for key in model_keys:
            sensitivity[key] = {}
            for label, factory in SENSITIVITY_VARIANTS_V4[key].items():
                oof, _ = cross_validated_scores(key, matrix, fsplits, classifier_factory=factory)
                summary = summarize_folds(per_split_metrics(y, oof, fsplits))
                sensitivity[key][label] = {metric: summary[metric]["mean"] for metric in ("roc_auc", "f1", "precision", "recall")}

    for entry in models_report.values():
        entry.pop("_oof_family", None)

    report = {
        "report": "fire_detection_model_comparison_v4",
        "synthetic_data_notice": (
            "All metrics are measured on the synthetic V4 benchmark (training_v4.csv). They describe generalisation "
            "across synthetic scenario families/archetypes, NOT real-world wildfire detection accuracy."
        ),
        "dataset": {
            "version": DATASET_VERSION_V4,
            "rows": int(len(y)),
            "fire_rows": int(y.sum()),
            "scenario_families": int(len(set(matrix.families.tolist()))),
            "archetypes": sorted(set(matrix.archetypes.tolist())),
            "feature_names": list(matrix.feature_names),
            "sha256": dataset_sha256(csv_path) if csv_path else None,
        },
        "environment": environment_info(csv_path),
        "evaluation_design": {
            "primary": "grouped_family_cv: 5 folds, whole scenario families held out (2 fire + 2 no-fire families per fold)",
            "secondary": "leave_one_archetype_out: 4 folds",
            "diagnostic_only": "stratified random 80/20 split and 5-fold CV",
            "family_fold_composition": {
                split.name: sorted(set(matrix.families[split.validation].tolist())) for split in fsplits
            },
            "family_fold_caveat": (
                "The frozen fold layout deals families round-robin by name, so a fold can hold out EVERY family of one "
                "evidence shape (e.g. both satellite-only fire families in family_fold_2, whose collapse it explains). "
                "That is a property of the evaluation layout, not a dataset bug. It is NOT what limits performance: the "
                "supplementary leave-one-family-out check (sibling families kept in training) does not score better "
                "overall (see leave_one_family_out_supplementary; it also has its own bias - single-label groups and "
                "pooled predictions from 20 different models). Weaknesses that persist under both layouts are "
                "per-family, see subgroups_at_threshold_0.5."
            ),
            "operating_point_for_fold_metrics": 0.5,
            "std": "population standard deviation across folds",
            "thresholds_calibration_importance": "computed from out-of-fold grouped predictions only",
            "hyperparameters": "fixed a priori with moderate regularisation; not tuned on evaluation splits",
        },
        "rule_baseline": rule,
        "models": models_report,
        "fire_danger_ablation_and_feature_groups": ablation,
        "hyperparameter_sensitivity_grouped_family_cv": sensitivity,
        "generalization_gate_definition": {
            **GATE_V4,
            "note": "Fixed before any model was evaluated. Not tuned to the results.",
        },
        "threshold_recommendation_targets": {
            "suspect_recall": SUSPECT_RECALL_TARGET,
            "confirm_precision": CONFIRM_PRECISION_TARGET,
        },
        "selection": selection,
    }
    return to_builtin(report)
