"""Train and evaluate the Logistic Regression Fire Detection ML baseline.

This is the first interpretable ML baseline for active wildfire detection
(User Story 2.2), trained on the synthetic dataset produced by
scripts.generate_fire_detection_training_data. It is an offline training
script only: it does not modify FireDetectionAgent, FireDetectionCalculator,
or any runtime/persistence path, and does not load the saved model anywhere
at runtime.

Example:
    python -m scripts.train_fire_detection_model
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES, LABEL_COLUMN

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "fire_detection" / "training_v1.csv"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "fire_detection"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "fire_detection_logistic_v1.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "fire_detection_logistic_v1_metadata.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ITER = 1000
MODEL_VERSION = "1.0"


def load_dataset(dataset_path: Path) -> tuple[list[list[float]], list[int]]:
    """Load only the centrally declared feature columns and the binary label."""
    features: list[list[float]] = []
    labels: list[int] = []
    with dataset_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            features.append([float(row[name]) for name in FIRE_DETECTION_FEATURE_NAMES])
            labels.append(int(row[LABEL_COLUMN]))
    return features, labels


def build_pipeline() -> Pipeline:
    """StandardScaler -> LogisticRegression. Scaling is fit only on the training split."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
        ]
    )


def train_and_evaluate(dataset_path: Path, model_path: Path, metadata_path: Path) -> dict:
    features, labels = load_dataset(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }

    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    coefficients = {
        name: float(coefficient) for name, coefficient in zip(FIRE_DETECTION_FEATURE_NAMES, classifier.coef_[0])
    }
    intercept = float(classifier.intercept_[0])

    _print_report(
        total_samples=len(labels),
        positive_samples=sum(labels),
        train_samples=len(y_train),
        test_samples=len(y_test),
        metrics=metrics,
        coefficients=coefficients,
        intercept=intercept,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_type": "LogisticRegression",
        "model_role": "interpretable_ml_baseline",
        "model_version": MODEL_VERSION,
        "dataset": dataset_path.name,
        "random_seed": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "feature_names": list(FIRE_DETECTION_FEATURE_NAMES),
        "total_samples": len(labels),
        "positive_samples": int(sum(labels)),
        "negative_samples": int(len(labels) - sum(labels)),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "metrics": metrics,
        "coefficients_standardized": coefficients,
        "intercept_standardized": intercept,
        "coefficient_note": (
            "Coefficients correspond to StandardScaler-standardized features "
            "(zero mean, unit variance), not raw feature units."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    return metadata


def _print_report(
    total_samples: int,
    positive_samples: int,
    train_samples: int,
    test_samples: int,
    metrics: dict,
    coefficients: dict,
    intercept: float,
) -> None:
    print("=== Fire Detection Logistic Regression baseline (interpretable ML baseline, synthetic data) ===")
    print(f"Total samples: {total_samples} (positive={positive_samples}, negative={total_samples - positive_samples})")
    print(f"Train samples: {train_samples}  Test samples: {test_samples}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion matrix:")
    print(f"  TN={metrics['true_negative']}  FP={metrics['false_positive']}")
    print(f"  FN={metrics['false_negative']}  TP={metrics['true_positive']}")
    print("\nLearned coefficients (StandardScaler-standardized features, not raw units):")
    for name in FIRE_DETECTION_FEATURE_NAMES:
        print(f"  {name:<28} {coefficients[name]:+.4f}")
    print(f"  {'intercept':<28} {intercept:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Fire Detection Logistic Regression baseline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    train_and_evaluate(args.dataset, args.model_path, args.metadata_path)


if __name__ == "__main__":
    main()
