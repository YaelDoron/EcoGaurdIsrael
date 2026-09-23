"""Compare Logistic Regression vs Random Forest on training_v3.csv (enriched evidence).

Extends the V2 comparison (scripts.compare_fire_detection_models_v2, left
untouched) with:
  - the V3 feature schema (satellite FRP/brightness + news semantic signal,
    replacing satellite_count with the three confidence counts directly)
  - archetype holdout (leave-one-scenario_archetype-out) generalization check
  - a feature-group ablation isolating the contribution of satellite-physical
    vs news-semantic features

training_v1.csv/training_v2.csv, their models, and their comparison scripts
are left untouched. Offline comparison only - no runtime classifier is
selected and FireDetectionAgent/FireDetectionCalculator are not touched.

Example:
    python -m scripts.compare_fire_detection_models_v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from scripts.train_fire_detection_model import MAX_ITER, build_pipeline as build_logistic_pipeline
from src.ml.fire_detection.fire_detection_dataset_v3 import load_training_dataset_rows_v3
from src.ml.fire_detection.fire_detection_feature_groups_v3 import FEATURE_GROUPS, build_feature_matrix
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_grouped_evaluation import (
    choose_grouped_fold_count,
    run_grouped_cross_validation,
    run_leave_one_group_out_cross_validation,
    verify_grouped_folds,
)
from src.ml.fire_detection.fire_detection_model_comparison import (
    CV_RANDOM_STATE,
    CV_SPLITS,
    HELD_OUT_RANDOM_STATE,
    HELD_OUT_TEST_SIZE,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    build_random_forest,
    evaluate_cross_validation,
    evaluate_held_out,
    evaluate_predictions,
    make_cv_splitter,
    make_held_out_split,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_V3_DATASET_PATH = DATA_DIR / "training_v3.csv"

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "fire_detection"
LOGISTIC_V3_MODEL_PATH = MODEL_DIR / "fire_detection_logistic_v3.joblib"
LOGISTIC_V3_METADATA_PATH = MODEL_DIR / "fire_detection_logistic_v3_metadata.json"
RF_V3_MODEL_PATH = MODEL_DIR / "fire_detection_random_forest_v3.joblib"
RF_V3_METADATA_PATH = MODEL_DIR / "fire_detection_random_forest_v3_metadata.json"
COMPARISON_V3_JSON_PATH = MODEL_DIR / "fire_detection_model_comparison_v3.json"

MODEL_VERSION_V3 = "3.0"


def _dataset_summary(rows) -> dict:
    positive = sum(1 for row in rows if row.label == 1)
    return {"total_samples": len(rows), "positive_samples": positive, "negative_samples": len(rows) - positive}


def _fold_summaries(result) -> list[dict]:
    return [
        {"fold_index": fold.fold_index, "validation_groups": list(fold.validation_families), "metrics": fold.metrics}
        for fold in result.fold_results
    ]


def _run_ablation(features_full: list, labels: list, groups: list, cv, n_splits_grouped: int) -> dict:
    results: dict[str, dict] = {}
    for group_name in FEATURE_GROUPS:
        selected = build_feature_matrix(features_full, group_name)
        model_results: dict[str, dict] = {}
        for model_name, factory in (
            ("logistic_regression", build_logistic_pipeline),
            ("random_forest", build_random_forest),
        ):
            standard_cv = evaluate_cross_validation(factory(), selected, labels, cv)
            grouped = run_grouped_cross_validation(factory, selected, labels, groups, n_splits_grouped)
            model_results[model_name] = {
                "standard_cv_f1": standard_cv["f1"]["mean"],
                "standard_cv_f1_std": standard_cv["f1"]["std"],
                "standard_cv_roc_auc": standard_cv["roc_auc"]["mean"],
                "standard_cv_roc_auc_std": standard_cv["roc_auc"]["std"],
                "grouped_cv_f1": grouped.summary["f1"]["mean"],
                "grouped_cv_f1_std": grouped.summary["f1"]["std"],
                "grouped_cv_roc_auc": grouped.summary["roc_auc"]["mean"],
                "grouped_cv_roc_auc_std": grouped.summary["roc_auc"]["std"],
            }
        results[group_name] = {"feature_names": list(FEATURE_GROUPS[group_name]), "models": model_results}
    return results


def run_comparison(dataset_path: Path) -> dict:
    rows = load_training_dataset_rows_v3(dataset_path)
    features = [list(row.features) for row in rows]
    labels = [row.label for row in rows]
    families = [row.scenario_family for row in rows]
    archetypes = [row.scenario_archetype.value for row in rows]

    # --- Standard (random, stratified) held-out split + CV ---
    split = make_held_out_split(features, labels)
    cv = make_cv_splitter()

    logistic_pipeline = build_logistic_pipeline()
    logistic_pipeline.fit(split.x_train, split.y_train)
    logistic_held_out = evaluate_held_out(logistic_pipeline, split.x_test, split.y_test)
    logistic_cv = evaluate_cross_validation(build_logistic_pipeline(), features, labels, cv)

    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)
    rf_train_metrics = evaluate_predictions(random_forest, split.x_train, split.y_train)
    rf_held_out = evaluate_held_out(random_forest, split.x_test, split.y_test)
    rf_cv = evaluate_cross_validation(build_random_forest(), features, labels, cv)

    # --- Grouped (unseen scenario-family) CV ---
    n_splits_grouped = choose_grouped_fold_count(labels, families)
    verify_grouped_folds(labels, families, n_splits_grouped)
    logistic_grouped = run_grouped_cross_validation(build_logistic_pipeline, features, labels, families, n_splits_grouped)
    rf_grouped = run_grouped_cross_validation(build_random_forest, features, labels, families, n_splits_grouped)

    # --- Archetype holdout (leave-one-archetype-out) ---
    logistic_archetype = run_leave_one_group_out_cross_validation(build_logistic_pipeline, features, labels, archetypes)
    rf_archetype = run_leave_one_group_out_cross_validation(build_random_forest, features, labels, archetypes)

    # --- Feature-group ablation ---
    ablation = _run_ablation(features, labels, families, cv, n_splits_grouped)

    # --- Save V3 artifacts (never overwrites V1/V2 paths) ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_pipeline, LOGISTIC_V3_MODEL_PATH)
    joblib.dump(random_forest, RF_V3_MODEL_PATH)

    dataset_stats = _dataset_summary(rows)

    logistic_metadata = {
        "model_type": "LogisticRegression",
        "model_role": "interpretable_ml_baseline",
        "model_version": MODEL_VERSION_V3,
        "dataset": dataset_path.name,
        "random_seed": HELD_OUT_RANDOM_STATE,
        "test_size": HELD_OUT_TEST_SIZE,
        "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V3),
        **dataset_stats,
        "train_samples": len(split.y_train),
        "test_samples": len(split.y_test),
        "held_out_metrics": logistic_held_out,
        "standard_cross_validation": {"n_splits": CV_SPLITS, "random_state": CV_RANDOM_STATE, "metrics": logistic_cv},
        "grouped_cross_validation": {
            "n_splits": n_splits_grouped,
            "group": "scenario_family",
            "metrics": logistic_grouped.summary,
        },
        "archetype_holdout": {
            "n_splits": logistic_archetype.n_splits,
            "group": "scenario_archetype",
            "metrics": logistic_archetype.summary,
        },
    }
    LOGISTIC_V3_METADATA_PATH.write_text(json.dumps(logistic_metadata, indent=2) + "\n", encoding="utf-8")

    rf_metadata = {
        "model_type": "RandomForestClassifier",
        "model_role": "nonlinear_ml_comparison_candidate",
        "model_version": MODEL_VERSION_V3,
        "dataset": dataset_path.name,
        "random_seed": RF_RANDOM_STATE,
        "feature_names": list(FIRE_DETECTION_FEATURE_NAMES_V3),
        "training_configuration": {"n_estimators": RF_N_ESTIMATORS, "random_state": RF_RANDOM_STATE, "class_weight": None},
        **dataset_stats,
        "train_samples": len(split.y_train),
        "test_samples": len(split.y_test),
        "train_metrics": rf_train_metrics,
        "held_out_metrics": rf_held_out,
        "standard_cross_validation": {"n_splits": CV_SPLITS, "random_state": CV_RANDOM_STATE, "metrics": rf_cv},
        "grouped_cross_validation": {
            "n_splits": n_splits_grouped,
            "group": "scenario_family",
            "metrics": rf_grouped.summary,
        },
        "archetype_holdout": {
            "n_splits": rf_archetype.n_splits,
            "group": "scenario_archetype",
            "metrics": rf_archetype.summary,
        },
    }
    RF_V3_METADATA_PATH.write_text(json.dumps(rf_metadata, indent=2) + "\n", encoding="utf-8")

    comparison = {
        "dataset": dataset_path.name,
        **dataset_stats,
        "held_out_split": {"test_size": HELD_OUT_TEST_SIZE, "random_state": HELD_OUT_RANDOM_STATE},
        "standard_cross_validation": {"n_splits": CV_SPLITS, "random_state": CV_RANDOM_STATE},
        "grouped_cross_validation": {"n_splits": n_splits_grouped, "group": "scenario_family"},
        "archetype_holdout": {"n_splits": logistic_archetype.n_splits, "group": "scenario_archetype"},
        "logistic_regression": {
            "config": {"scaler": "StandardScaler", "max_iter": MAX_ITER, "random_state": HELD_OUT_RANDOM_STATE},
            "held_out_metrics": logistic_held_out,
            "standard_cv_metrics": logistic_cv,
            "grouped_cv_summary": logistic_grouped.summary,
            "grouped_cv_folds": _fold_summaries(logistic_grouped),
            "grouped_per_family": [vars(item) for item in logistic_grouped.per_family],
            "archetype_holdout_summary": logistic_archetype.summary,
            "archetype_holdout_folds": _fold_summaries(logistic_archetype),
            "archetype_holdout_per_archetype": [vars(item) for item in logistic_archetype.per_family],
        },
        "random_forest": {
            "config": rf_metadata["training_configuration"],
            "train_metrics": rf_train_metrics,
            "held_out_metrics": rf_held_out,
            "standard_cv_metrics": rf_cv,
            "grouped_cv_summary": rf_grouped.summary,
            "grouped_cv_folds": _fold_summaries(rf_grouped),
            "grouped_per_family": [vars(item) for item in rf_grouped.per_family],
            "archetype_holdout_summary": rf_archetype.summary,
            "archetype_holdout_folds": _fold_summaries(rf_archetype),
            "archetype_holdout_per_archetype": [vars(item) for item in rf_archetype.per_family],
        },
        "feature_group_ablation": ablation,
        "note": (
            "Dataset is synthetic (V3 proof-of-concept: enriched evidence, still synthetic labels and "
            "synthetic satellite FRP/brightness/news-signal generation - not the real LLM). Grouped CV and "
            "archetype holdout test generalization to unseen synthetic scenario families/compositions - "
            "neither constitutes real-world validation. No runtime classifier is selected by this comparison."
        ),
    }
    COMPARISON_V3_JSON_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    return {
        "rows": rows,
        "dataset_stats": dataset_stats,
        "n_splits_grouped": n_splits_grouped,
        "logistic_held_out": logistic_held_out,
        "logistic_cv": logistic_cv,
        "logistic_grouped": logistic_grouped,
        "logistic_archetype": logistic_archetype,
        "rf_train_metrics": rf_train_metrics,
        "rf_held_out": rf_held_out,
        "rf_cv": rf_cv,
        "rf_grouped": rf_grouped,
        "rf_archetype": rf_archetype,
        "ablation": ablation,
        "comparison": comparison,
    }


def _print_metric_table(title: str, logistic: dict, rf: dict, keys=("accuracy", "precision", "recall", "f1", "roc_auc")) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Metric':<16}{'Logistic Regression':<25}{'Random Forest'}")
    print("-" * 60)
    for key in keys:
        label = "ROC-AUC" if key == "roc_auc" else key.capitalize()
        print(f"{label:<16}{logistic[key]:<25.4f}{rf[key]:.4f}")


def _print_cv_table(title: str, logistic_cv: dict, rf_cv: dict) -> None:
    print(f"\n=== {title} (mean +/- std) ===")
    print(f"{'Metric':<16}{'Logistic Regression':<25}{'Random Forest'}")
    print("-" * 60)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        label = "ROC-AUC" if key == "roc_auc" else key.capitalize()
        lr_cell = f"{logistic_cv[key]['mean']:.3f} +/- {logistic_cv[key]['std']:.3f}"
        rf_cell = f"{rf_cv[key]['mean']:.3f} +/- {rf_cv[key]['std']:.3f}"
        print(f"{label:<16}{lr_cell:<25}{rf_cell}")


def _print_report(result: dict) -> None:
    print("=== V3 dataset ===")
    print(result["dataset_stats"])

    _print_metric_table("Held-Out Test Metrics (V3)", result["logistic_held_out"], result["rf_held_out"])
    print(
        f"FP/FN  LR={result['logistic_held_out']['false_positive']}/{result['logistic_held_out']['false_negative']}  "
        f"RF={result['rf_held_out']['false_positive']}/{result['rf_held_out']['false_negative']}"
    )

    _print_cv_table("Standard 5-Fold CV (V3)", result["logistic_cv"], result["rf_cv"])

    n_splits = result["n_splits_grouped"]
    print(f"\n=== Grouped CV by scenario_family (n_splits={n_splits}) ===")
    _print_cv_table(f"Grouped {n_splits}-Fold CV (unseen scenario families)", result["logistic_grouped"].summary, result["rf_grouped"].summary)

    archetype_splits = result["logistic_archetype"].n_splits
    print(f"\n=== Archetype holdout (leave-one-archetype-out, n_splits={archetype_splits}) ===")
    _print_cv_table("Archetype holdout", result["logistic_archetype"].summary, result["rf_archetype"].summary)
    print("\nPer-archetype (Random Forest):")
    print(f"{'archetype':<20}{'actual+frac':<13}{'count':<7}{'pred+rate':<12}{'accuracy'}")
    for item in result["rf_archetype"].per_family:
        print(
            f"{item.scenario_family:<20}{item.actual_positive_fraction:<13.3f}"
            f"{item.sample_count:<7}{item.predicted_positive_rate:<12.3f}{item.accuracy:.3f}"
        )

    print("\n=== Random Forest overfitting check (V3) ===")
    print(f"Train: accuracy={result['rf_train_metrics']['accuracy']:.4f}  f1={result['rf_train_metrics']['f1']:.4f}")
    print(f"Test:  accuracy={result['rf_held_out']['accuracy']:.4f}  f1={result['rf_held_out']['f1']:.4f}")

    print("\n=== Feature-group ablation (standard CV / grouped CV) ===")
    for group_name, group_result in result["ablation"].items():
        print(f"\nFeature group: {group_name} -> {group_result['feature_names']}")
        for model_name, metrics in group_result["models"].items():
            print(
                f"  {model_name:<22} "
                f"standard F1={metrics['standard_cv_f1']:.4f}+/-{metrics['standard_cv_f1_std']:.4f}  "
                f"standard ROC-AUC={metrics['standard_cv_roc_auc']:.4f}+/-{metrics['standard_cv_roc_auc_std']:.4f}  "
                f"grouped F1={metrics['grouped_cv_f1']:.4f}+/-{metrics['grouped_cv_f1_std']:.4f}  "
                f"grouped ROC-AUC={metrics['grouped_cv_roc_auc']:.4f}+/-{metrics['grouped_cv_roc_auc_std']:.4f}"
            )

    print(f"\nSaved Logistic Regression V3 model: {LOGISTIC_V3_MODEL_PATH}")
    print(f"Saved Logistic Regression V3 metadata: {LOGISTIC_V3_METADATA_PATH}")
    print(f"Saved Random Forest V3 model: {RF_V3_MODEL_PATH}")
    print(f"Saved Random Forest V3 metadata: {RF_V3_METADATA_PATH}")
    print(f"Saved comparison artifact: {COMPARISON_V3_JSON_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Logistic Regression vs Random Forest on training_v3.csv.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V3_DATASET_PATH)
    args = parser.parse_args()

    result = run_comparison(args.dataset)
    _print_report(result)


if __name__ == "__main__":
    main()
