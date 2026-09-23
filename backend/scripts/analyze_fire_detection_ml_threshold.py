"""Threshold analysis for the HYBRID-mode ML escalation (NO_EVENT -> SUSPECTED).

This does NOT retrain or overwrite the saved fire_detection_logistic_v3.joblib
production artifact. It fits the exact same Logistic Regression pipeline
configuration (StandardScaler -> LogisticRegression, from
scripts.train_fire_detection_model.build_pipeline) in-memory, purely to get
deterministic out-of-fold predicted probabilities via cross_val_predict on
training_v3.csv - the methodology Part 19 explicitly asks for, to avoid
picking a threshold by repeatedly probing the held-out test split.

For each candidate threshold, treats "predicted positive" as
out-of-fold P(fire) >= threshold, and reports precision, recall, false
positive rate, and the predicted-positive count against the true synthetic
label. High precision is preferred for this specific escalation path (ML
supplementing, not replacing, the rule-based detector) - see
backend/docs/fire_detection_runtime_ml.md for the selected threshold and
rationale.

Example:
    python -m scripts.analyze_fire_detection_ml_threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import cross_val_predict

from scripts.train_fire_detection_model import build_pipeline
from src.ml.fire_detection.fire_detection_dataset_v3 import load_training_dataset_rows_v3
from src.ml.fire_detection.fire_detection_model_comparison import CV_RANDOM_STATE, CV_SPLITS, make_cv_splitter

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fire_detection"
DEFAULT_V3_DATASET_PATH = DATA_DIR / "training_v3.csv"

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "fire_detection"
DEFAULT_OUTPUT_PATH = MODEL_DIR / "fire_detection_logistic_v3_threshold_analysis.json"

CANDIDATE_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PRECISION_TARGET = 0.90


def compute_out_of_fold_probabilities(dataset_path: Path) -> tuple[list[float], list[int]]:
    rows = load_training_dataset_rows_v3(dataset_path)
    features = [list(row.features) for row in rows]
    labels = [row.label for row in rows]

    cv = make_cv_splitter()
    probabilities = cross_val_predict(build_pipeline(), features, labels, cv=cv, method="predict_proba")[:, 1]
    return list(probabilities), labels


def evaluate_threshold(probabilities: list[float], labels: list[int], threshold: float) -> dict:
    true_positive = false_positive = true_negative = false_negative = 0
    for probability, label in zip(probabilities, labels):
        predicted_positive = probability >= threshold
        if predicted_positive and label == 1:
            true_positive += 1
        elif predicted_positive and label == 0:
            false_positive += 1
        elif not predicted_positive and label == 0:
            true_negative += 1
        else:
            false_negative += 1

    predicted_positive_count = true_positive + false_positive
    precision = true_positive / predicted_positive_count if predicted_positive_count else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    false_positive_rate = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "predicted_positive_count": predicted_positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "sample_count": len(labels),
    }


def select_threshold(results: list[dict], precision_target: float) -> float | None:
    """Smallest candidate threshold meeting the precision target, or None if none qualifies."""
    qualifying = [result for result in results if result["precision"] >= precision_target and result["predicted_positive_count"] > 0]
    if not qualifying:
        return None
    return min(qualifying, key=lambda result: result["threshold"])["threshold"]


def run_analysis(dataset_path: Path) -> dict:
    probabilities, labels = compute_out_of_fold_probabilities(dataset_path)
    results = [evaluate_threshold(probabilities, labels, threshold) for threshold in CANDIDATE_THRESHOLDS]
    selected_threshold = select_threshold(results, PRECISION_TARGET)

    return {
        "dataset": dataset_path.name,
        "cv_methodology": {
            "n_splits": CV_SPLITS,
            "random_state": CV_RANDOM_STATE,
            "shuffle": True,
            "note": "out-of-fold predictions via cross_val_predict - not the held-out test split, and not the saved production model.",
        },
        "candidate_thresholds": results,
        "precision_target": PRECISION_TARGET,
        "selected_threshold": selected_threshold,
        "escalation_enabled": selected_threshold is not None,
        "note": (
            "If selected_threshold is null, no candidate threshold reached the precision target on this "
            "synthetic cross-validated data; HYBRID NO_EVENT->SUSPECTED ML escalation should remain disabled "
            "rather than using a fabricated threshold."
        ),
    }


def _print_report(analysis: dict) -> None:
    print(f"=== ML escalation threshold analysis ({analysis['dataset']}) ===")
    print(f"{'threshold':<12}{'precision':<12}{'recall':<12}{'FPR':<12}{'pred_positive'}")
    for result in analysis["candidate_thresholds"]:
        print(
            f"{result['threshold']:<12.2f}{result['precision']:<12.4f}{result['recall']:<12.4f}"
            f"{result['false_positive_rate']:<12.4f}{result['predicted_positive_count']}"
        )
    if analysis["selected_threshold"] is not None:
        print(f"\nSelected threshold: {analysis['selected_threshold']} (precision target >= {analysis['precision_target']})")
    else:
        print(f"\nNo threshold reached the precision target (>= {analysis['precision_target']}). ML escalation disabled.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold analysis for HYBRID ML escalation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V3_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    analysis = run_analysis(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    _print_report(analysis)
    print(f"\nSaved threshold analysis: {args.output}")


if __name__ == "__main__":
    main()
