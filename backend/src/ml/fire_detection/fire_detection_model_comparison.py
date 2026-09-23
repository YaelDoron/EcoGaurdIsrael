"""Shared, testable comparison logic for Logistic Regression vs Random Forest.

Used by backend/scripts/compare_fire_detection_models.py. Kept separate from
that script so the split/CV/evaluation logic is independently unit-testable
without re-running the full CLI. This module trains and evaluates models for
offline comparison only - it does not select a runtime classifier and is
never imported by FireDetectionAgent or FireDetectionCalculator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.ml.fire_detection.fire_detection_dataset import FireDetectionDatasetRow
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_training_data_generator import FireDetectionTrainingSample
from src.models.fire_detection_status import FireDetectionStatus

# Identical to scripts.train_fire_detection_model's held-out split configuration,
# so both classifiers are compared on the exact same held-out rows.
HELD_OUT_TEST_SIZE = 0.2
HELD_OUT_RANDOM_STATE = 42

CV_SPLITS = 5
CV_RANDOM_STATE = 42
CV_SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")

RF_N_ESTIMATORS = 300
RF_RANDOM_STATE = 42


@dataclass(frozen=True)
class HeldOutSplit:
    """One shared train/test split, reused by every compared classifier.

    train_indices/test_indices are positions into the original dataset row
    order (== sample_id - 1), so a non-feature artifact (e.g. the raw
    evidence used by the rule-based calculator) can be sliced with the exact
    same test rows as the ML classifiers.
    """

    x_train: list
    x_test: list
    y_train: list
    y_test: list
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def make_held_out_split(features: list[list[float]], labels: list[int]) -> HeldOutSplit:
    """Build the one shared 80/20 stratified split every model in the comparison must reuse."""
    indices = list(range(len(labels)))
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        features,
        labels,
        indices,
        test_size=HELD_OUT_TEST_SIZE,
        random_state=HELD_OUT_RANDOM_STATE,
        stratify=labels,
    )
    return HeldOutSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_indices=tuple(train_indices),
        test_indices=tuple(test_indices),
    )


def make_cv_splitter() -> StratifiedKFold:
    """Build the one shared 5-fold splitter every model in the comparison must reuse."""
    return StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)


def build_random_forest() -> RandomForestClassifier:
    """Simple baseline configuration - no scaling needed, no held-out-test-driven tuning applied."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        class_weight=None,
    )


def evaluate_predictions(estimator, x, y) -> dict:
    """Accuracy/F1 only - used for the RF train-vs-test overfitting check."""
    y_pred = estimator.predict(x)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }


def evaluate_held_out(estimator, x_test, y_test) -> dict:
    """Full held-out metric set, using predict_proba for ROC-AUC."""
    y_pred = estimator.predict(x_test)
    y_proba = estimator.predict_proba(x_test)[:, 1]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()
    return {
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


def evaluate_cross_validation(estimator, features: list[list[float]], labels: list[int], cv: StratifiedKFold) -> dict:
    """Stratified k-fold CV: mean +/- std per metric. A fresh clone of estimator is fit per fold."""
    results = cross_validate(estimator, features, labels, cv=cv, scoring=list(CV_SCORING))
    return {
        metric: {
            "mean": float(np.mean(results[f"test_{metric}"])),
            "std": float(np.std(results[f"test_{metric}"])),
        }
        for metric in CV_SCORING
    }


def compute_feature_importances(random_forest: RandomForestClassifier) -> tuple[tuple[str, float], ...]:
    """Impurity-based feature importances, sorted descending. See module docstring caveats."""
    pairs = tuple(zip(FIRE_DETECTION_FEATURE_NAMES, (float(value) for value in random_forest.feature_importances_)))
    return tuple(sorted(pairs, key=lambda item: item[1], reverse=True))


def compute_permutation_importance(
    estimator,
    x_test: list[list[float]],
    y_test: list[int],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "f1",
) -> tuple[tuple[str, float], ...]:
    """Held-out permutation importance - distinct from (and more robust to correlated
    features than) impurity-based feature_importances_. Sorted descending."""
    result = permutation_importance(
        estimator, x_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    pairs = tuple(zip(FIRE_DETECTION_FEATURE_NAMES, (float(value) for value in result.importances_mean)))
    return tuple(sorted(pairs, key=lambda item: item[1], reverse=True))


def regenerated_samples_match_dataset(
    csv_rows: tuple[FireDetectionDatasetRow, ...],
    generated_samples: tuple[FireDetectionTrainingSample, ...],
) -> bool:
    """Confirm re-running the generator with the same seed reproduces training_v1.csv exactly.

    This must hold before using freshly regenerated evidence (needed for the
    rule-based calculator, which the CSV's summarized features cannot
    reconstruct) as a stand-in for the rows already saved in the CSV. A
    mismatch would mean training_v1.csv is stale relative to the current
    generator code - Part 1 says not to regenerate the CSV without cause, so
    this check exists precisely to catch that "cause" rather than assume it.
    """
    if len(csv_rows) != len(generated_samples):
        return False
    extractor = FireDetectionFeatureExtractor()
    for csv_row, sample in zip(csv_rows, generated_samples):
        if csv_row.sample_id != sample.sample_id:
            return False
        if csv_row.scenario_family != sample.scenario_family:
            return False
        if csv_row.label != sample.label:
            return False
        recomputed_features = extractor.extract(sample.evidence).as_tuple()
        if csv_row.features != recomputed_features:
            return False
    return True


def evaluate_rule_based_calculator(
    samples: tuple[FireDetectionTrainingSample, ...],
    indices: tuple[int, ...],
) -> dict:
    """Evaluate the deterministic FireDetectionCalculator on the same held-out rows.

    NO_EVENT maps to predicted label 0; SUSPECTED/CONFIRMED map to predicted
    label 1. decision.confidence is also used as a continuous score for
    ROC-AUC - it is a rule-based heuristic value, not a trained/calibrated
    probability, so that ROC-AUC number must be read with that caveat.
    Requires no code change to FireDetectionCalculator itself.
    """
    calculator = FireDetectionCalculator()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    for index in indices:
        sample = samples[index]
        decision = calculator.evaluate(sample.evidence)
        y_true.append(sample.label)
        y_pred.append(0 if decision.status is FireDetectionStatus.NO_EVENT else 1)
        y_score.append(decision.confidence)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }
    if len(set(y_true)) == 2:
        metrics["roc_auc_using_rule_based_confidence_as_score"] = float(roc_auc_score(y_true, y_score))
    return metrics
