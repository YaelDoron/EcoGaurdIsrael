"""Evaluation logic for the Fire Detection ML V4 model comparison (offline only).

Kept separate from scripts/compare_fire_detection_models_v4.py so every piece is
independently unit-testable. Nothing here is imported by runtime Fire Detection.

Primary model-selection evidence is GROUPED evaluation - whole scenario families
(and, separately, whole evidence-shape archetypes) are held out, so the model is
always scored on situations it has never seen. Random splits are provided only as
a diagnostic (V3's random CV looked far better than its grouped CV).

All thresholds, calibration and importance analyses use out-of-fold predictions
from grouped cross-validation only. The rule-based FireDetectionCalculator is
evaluated here strictly as a BASELINE against ground-truth labels; its output is
never a training label.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.ml.fire_detection.fire_detection_dataset_v4 import (
    family_fold_assignments,
    grouped_family_folds,
    leave_one_archetype_out,
    row_from_sample,
)
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import is_hard_negative, is_weak_positive
from src.ml.fire_detection.fire_detection_model_v4 import (
    MODEL_SPECS_V4,
    RANDOM_SEED_V4,
    TrainingMatrixV4,
    build_pipeline_v4,
    transformed_feature_names,
)
from src.models.fire_detection_status import FireDetectionStatus

METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc")
DEFAULT_THRESHOLD = 0.50
CANDIDATE_THRESHOLDS = (0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80)
THRESHOLD_SEARCH_GRID = tuple(round(value, 2) for value in np.arange(0.05, 0.9601, 0.01))

# Operating-point targets used to RECOMMEND (never wire in) SUSPECTED / CONFIRMED thresholds.
SUSPECT_RECALL_TARGET = 0.85  # SUSPECTED = catch most real fires (high recall)
CONFIRM_PRECISION_TARGET = 0.90  # CONFIRMED = strong evidence (high precision); same target V3 used

# --- the a-priori generalisation gate (fixed BEFORE any model was evaluated; not tuned) ---
GATE_V4 = {
    "grouped_mean_roc_auc_min": 0.80,
    "grouped_min_fold_roc_auc_min": 0.70,
    "grouped_mean_f1_over_rule_min": 0.03,  # mean grouped F1 must beat the rule baseline's by this much
    "suspect_precision_over_rule_min": 0.08,  # at the SUSPECTED operating point: fewer false alarms than the rule ...
    "suspect_recall_min": SUSPECT_RECALL_TARGET,  # ... while still catching >= 85% of real fires
    "hard_negative_fpr_at_suspect_max": 0.60,
    "weak_positive_recall_at_suspect_min": 0.60,
    "archetype_mean_roc_auc_min": 0.75,
    "archetype_worst_roc_auc_min": 0.65,
    "calibration_ece_max": 0.10,
    "max_single_feature_importance_share": 0.40,
    "max_geometry_importance_share": 0.25,
}
# Two models within these margins of the best passing model are treated as tied; the simplest wins.
TIE_MARGIN_ROC_AUC = 0.01
TIE_MARGIN_F1 = 0.02
_TIE_EPSILON = 1e-9  # floating-point slack so a difference of exactly the margin still counts as a tie


# --- metrics ---


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD, include_auc: bool = True) -> dict:
    """Confusion-matrix metrics at `threshold` (prediction = score >= threshold) plus ROC-AUC of the raw scores."""
    y_true = np.asarray(y_true)
    predicted = np.asarray(scores) >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "rows": len(y_true),
    }
    if include_auc:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None
    return metrics


def summarize_folds(fold_metrics: Sequence[dict], keys: Iterable[str] = METRIC_KEYS) -> dict:
    """mean / std (population) / min / max of each metric across folds."""
    summary = {}
    for key in keys:
        values = np.array([fold[key] for fold in fold_metrics if fold.get(key) is not None], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


# --- splits ---


@dataclass(frozen=True)
class NamedSplit:
    name: str
    train: np.ndarray
    validation: np.ndarray


def family_splits(matrix: TrainingMatrixV4, n_splits: int = 5) -> tuple[NamedSplit, ...]:
    """Grouped-by-scenario_family folds (whole families held out) from the frozen dataset helper."""
    folds = grouped_family_folds(matrix.rows, n_splits)
    return tuple(
        NamedSplit(f"family_fold_{index}", np.array(train), np.array(validation))
        for index, (train, validation) in enumerate(folds)
    )


def archetype_splits(matrix: TrainingMatrixV4) -> tuple[NamedSplit, ...]:
    """Leave-one-archetype-out splits from the frozen dataset helper."""
    return tuple(
        NamedSplit(archetype, np.array(train), np.array(validation))
        for archetype, (train, validation) in leave_one_archetype_out(matrix.rows)
    )


def leave_one_family_out_splits(matrix: TrainingMatrixV4) -> tuple[NamedSplit, ...]:
    """SUPPLEMENTARY: hold out ONE scenario family at a time; its sibling families stay in training.

    Unlike the 5-fold family layout (which holds out four families at once and can remove every family of an
    evidence shape from training), a sibling with similar evidence stays available. Each split holds out a single
    label, so only pooled out-of-fold metrics are meaningful (per-split ROC-AUC is undefined), and pooling
    predictions from 20 different models plus single-label groups biases this design against the held-out family.
    Diagnostic only: never used for the gate or model selection.
    """
    splits = []
    for family in sorted(set(matrix.families.tolist())):
        held_out = matrix.families == family
        splits.append(NamedSplit(family, np.where(~held_out)[0], np.where(held_out)[0]))
    return tuple(splits)


def assert_splits_are_grouped(matrix: TrainingMatrixV4, splits: Sequence[NamedSplit], group_values: np.ndarray) -> None:
    """Raise if any validation group also appears in that split's training rows."""
    for split in splits:
        overlap = set(group_values[split.train].tolist()) & set(group_values[split.validation].tolist())
        if overlap:
            raise ValueError(f"{split.name}: groups appear in both train and validation: {sorted(overlap)}")


# --- cross-validated scoring ---


def cross_validated_scores(
    model_key: str,
    matrix: TrainingMatrixV4,
    splits: Sequence[NamedSplit],
    feature_names: tuple[str, ...] | None = None,
    classifier_factory: Callable[[], object] | None = None,
    keep_models: bool = False,
):
    """Out-of-fold P(fire) for every row (each row is scored by a model that never saw its group).

    Returns (oof_scores, fitted_pipelines_or_None). Every split's validation set must partition the rows.
    """
    names = feature_names or matrix.feature_names
    X = matrix.columns(names)
    oof = np.full(len(matrix.y), np.nan)
    models = []
    for split in splits:
        pipeline = clone(build_pipeline_v4(model_key, names, classifier_factory() if classifier_factory else None))
        pipeline.fit(X[split.train], matrix.y[split.train])
        oof[split.validation] = pipeline.predict_proba(X[split.validation])[:, 1]
        if keep_models:
            models.append(pipeline)
    if np.isnan(oof).any():
        raise ValueError("Validation sets do not cover every row.")
    return oof, (models if keep_models else None)


def calibrated_cross_validated_scores(
    model_key: str,
    matrix: TrainingMatrixV4,
    splits: Sequence[NamedSplit],
    method: str,
    inner_splits: int = 4,
) -> np.ndarray:
    """Out-of-fold P(fire) from a CalibratedClassifierCV fitted inside each outer training fold.

    The inner calibration folds are grouped by scenario family too, so calibration is fitted only on
    families the base model did not train on.
    """
    X = matrix.X
    oof = np.full(len(matrix.y), np.nan)
    for split in splits:
        train_rows = tuple(matrix.rows[index] for index in split.train)
        assignment = family_fold_assignments(train_rows, inner_splits)
        train_folds = np.array([assignment[row.scenario_family] for row in train_rows])
        inner = [
            (np.where(train_folds != fold)[0], np.where(train_folds == fold)[0]) for fold in range(inner_splits)
        ]
        calibrated = CalibratedClassifierCV(
            estimator=build_pipeline_v4(model_key, matrix.feature_names), method=method, cv=inner
        )
        calibrated.fit(X[split.train], matrix.y[split.train])
        oof[split.validation] = calibrated.predict_proba(X[split.validation])[:, 1]
    if np.isnan(oof).any():
        raise ValueError("Validation sets do not cover every row.")
    return oof


def per_split_metrics(y: np.ndarray, scores: np.ndarray, splits: Sequence[NamedSplit], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Metrics for each split's validation rows, from out-of-fold scores."""
    results = []
    for split in splits:
        metrics = binary_metrics(y[split.validation], scores[split.validation], threshold)
        results.append({"split": split.name, **metrics})
    return results


def paired_differences(full_folds: Sequence[dict], reduced_folds: Sequence[dict], keys: Iterable[str] = METRIC_KEYS) -> dict:
    """Per-fold (full - reduced) differences on the SAME folds: mean, std and how many folds improved."""
    result = {}
    for key in keys:
        deltas = np.array([full[key] - reduced[key] for full, reduced in zip(full_folds, reduced_folds)], dtype=float)
        result[key] = {
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std()),
            "folds_improved": int(np.sum(deltas > 0)),
            "folds_worse": int(np.sum(deltas < 0)),
            "per_fold": [float(delta) for delta in deltas],
        }
    return result


# --- random split (diagnostic only) ---


def random_split_diagnostic(model_key: str, matrix: TrainingMatrixV4, seed: int = RANDOM_SEED_V4) -> dict:
    """Stratified random 80/20 split plus 5-fold stratified CV. DIAGNOSTIC ONLY - never used to pick a model."""
    train, test = train_test_split(
        np.arange(len(matrix.y)), test_size=0.2, stratify=matrix.y, random_state=seed
    )
    pipeline = build_pipeline_v4(model_key, matrix.feature_names).fit(matrix.X[train], matrix.y[train])
    held_out = binary_metrics(matrix.y[test], pipeline.predict_proba(matrix.X[test])[:, 1])

    folds = []
    for fold_train, fold_test in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(matrix.X, matrix.y):
        fold_model = build_pipeline_v4(model_key, matrix.feature_names).fit(matrix.X[fold_train], matrix.y[fold_train])
        folds.append(binary_metrics(matrix.y[fold_test], fold_model.predict_proba(matrix.X[fold_test])[:, 1]))
    return {
        "note": "Random split - diagnostic only. Not used for model selection.",
        "held_out_20_percent": held_out,
        "stratified_5fold_cv": summarize_folds(folds),
    }


# --- rule baseline (evaluated against ground truth; never a training label) ---


@dataclass(frozen=True)
class RuleOutputs:
    predicted_fire: np.ndarray  # SUSPECTED or CONFIRMED
    predicted_confirmed: np.ndarray  # CONFIRMED only
    confidence: np.ndarray  # the rule's own (uncalibrated) confidence, used as a ranking score for ROC-AUC


def verify_samples_match_matrix(samples, matrix: TrainingMatrixV4) -> None:
    """The regenerated samples (which carry evidence for the rule calculator) must be exactly the CSV's rows."""
    if len(samples) != len(matrix.rows):
        raise ValueError("Regenerated sample count differs from the CSV.")
    for sample, row in zip(samples, matrix.rows):
        regenerated = row_from_sample(sample)
        if (
            sample.sample_id != row.sample_id
            or sample.label != row.label
            or sample.scenario_family != row.scenario_family
            or not np.array_equal(
                np.array(regenerated.to_features_v4().as_tuple(), dtype=float),
                np.array(row.to_features_v4().as_tuple(), dtype=float),
                equal_nan=True,
            )
        ):
            raise ValueError(f"Regenerated sample {sample.sample_id} does not match the frozen CSV row.")


def rule_baseline_outputs(samples) -> RuleOutputs:
    """Run the existing deterministic FireDetectionCalculator on every sample's evidence."""
    calculator = FireDetectionCalculator()
    decisions = [calculator.evaluate(sample.evidence) for sample in samples]
    return RuleOutputs(
        predicted_fire=np.array([d.status is not FireDetectionStatus.NO_EVENT for d in decisions], dtype=float),
        predicted_confirmed=np.array([d.status is FireDetectionStatus.CONFIRMED for d in decisions], dtype=float),
        confidence=np.array([d.confidence for d in decisions], dtype=float),
    )


def rule_baseline_report(y: np.ndarray, outputs: RuleOutputs, splits: Sequence[NamedSplit]) -> dict:
    def variant(predicted: np.ndarray) -> dict:
        overall = binary_metrics(y, predicted, threshold=0.5, include_auc=False)
        overall["roc_auc"] = float(roc_auc_score(y, outputs.confidence))  # ranking by the rule's confidence
        folds = [
            {"split": s.name, **binary_metrics(y[s.validation], predicted[s.validation], 0.5, include_auc=False),
             "roc_auc": float(roc_auc_score(y[s.validation], outputs.confidence[s.validation]))}
            for s in splits
        ]
        return {"overall": overall, "per_fold": folds, "summary": summarize_folds(folds)}

    return {
        "note": "FireDetectionCalculator evaluated against ground-truth labels; fire = SUSPECTED or CONFIRMED.",
        "suspected_or_confirmed_is_fire": variant(outputs.predicted_fire),
        "confirmed_only_is_fire": variant(outputs.predicted_confirmed),
    }


# --- subgroup analysis ---


def subgroup_rates(matrix: TrainingMatrixV4, predicted: np.ndarray) -> dict:
    """Predicted-fire rate per scenario family (= recall for fire families, false-positive rate for no-fire families)."""
    result = {}
    for family in sorted(set(matrix.families.tolist())):
        mask = matrix.families == family
        result[family] = {
            "label": int(matrix.y[mask][0]),
            "rows": int(mask.sum()),
            "predicted_fire_rate": float(np.mean(predicted[mask])),
        }
    return result


def hard_negative_mask(matrix: TrainingMatrixV4) -> np.ndarray:
    return np.array([is_hard_negative(row) for row in matrix.rows])


def weak_positive_mask(matrix: TrainingMatrixV4) -> np.ndarray:
    return np.array([is_weak_positive(row) for row in matrix.rows])


def subgroup_performance(matrix: TrainingMatrixV4, predicted: np.ndarray) -> dict:
    """Hard-negative false-positive rate, weak-positive recall, and the per-family breakdown at one operating point."""
    hard = hard_negative_mask(matrix)
    weak = weak_positive_mask(matrix)
    easy_negatives = (matrix.y == 0) & ~hard
    return {
        "hard_negative_rows": int(hard.sum()),
        "hard_negative_false_positive_rate": float(np.mean(predicted[hard])),
        "other_negative_false_positive_rate": float(np.mean(predicted[easy_negatives])),
        "weak_positive_rows": int(weak.sum()),
        "weak_positive_recall": float(np.mean(predicted[weak])),
        "other_positive_recall": float(np.mean(predicted[(matrix.y == 1) & ~weak])),
        "per_family": subgroup_rates(matrix, predicted),
    }


def per_family_accuracy(matrix: TrainingMatrixV4, scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
    correct = (scores >= threshold).astype(int) == matrix.y
    return {family: float(np.mean(correct[matrix.families == family])) for family in sorted(set(matrix.families.tolist()))}


# --- thresholds ---


def threshold_table(y: np.ndarray, scores: np.ndarray, thresholds: Iterable[float] = CANDIDATE_THRESHOLDS) -> list[dict]:
    rows = []
    for threshold in thresholds:
        metrics = binary_metrics(y, scores, threshold, include_auc=False)
        rows.append({"threshold": threshold, **{key: metrics[key] for key in (
            "precision", "recall", "f1", "false_positive_rate", "false_negative_rate",
            "true_positive", "false_positive", "true_negative", "false_negative")},
            "predicted_positive": metrics["true_positive"] + metrics["false_positive"]})
    return rows


def recommend_thresholds(
    y: np.ndarray,
    scores: np.ndarray,
    splits: Sequence[NamedSplit] = (),
    suspect_recall_target: float = SUSPECT_RECALL_TARGET,
    confirm_precision_target: float = CONFIRM_PRECISION_TARGET,
    grid: Iterable[float] = THRESHOLD_SEARCH_GRID,
) -> dict:
    """Recommend SUSPECTED / CONFIRMED thresholds from out-of-fold grouped scores (not wired into runtime).

    suspect  = the HIGHEST threshold that still reaches `suspect_recall_target` (as few alarms as possible
               while catching most real fires);
    confirm  = the LOWEST threshold that reaches `confirm_precision_target` (as many confirmations as possible
               while staying reliable).
    None means the target is not attainable on this data.
    """
    grid = tuple(grid)
    table = {threshold: binary_metrics(y, scores, threshold, include_auc=False) for threshold in grid}
    suspect = max((t for t in grid if table[t]["recall"] >= suspect_recall_target), default=None)
    # CONFIRMED must sit ABOVE SUSPECTED, so it is searched only among thresholds higher than `suspect`.
    confirm = min(
        (
            t
            for t in grid
            if table[t]["precision"] >= confirm_precision_target
            and table[t]["true_positive"] > 0
            and (suspect is None or t > suspect)
        ),
        default=None,
    )

    def at(threshold):
        if threshold is None:
            return None
        overall = binary_metrics(y, scores, threshold, include_auc=False)
        folds = [binary_metrics(y[s.validation], scores[s.validation], threshold, include_auc=False) for s in splits]
        return {
            "threshold": threshold,
            "overall": {k: overall[k] for k in ("precision", "recall", "f1", "false_positive_rate", "false_negative_rate")},
            "per_fold_precision": [f["precision"] for f in folds],
            "per_fold_recall": [f["recall"] for f in folds],
            "min_fold_precision": min((f["precision"] for f in folds), default=None),
            "min_fold_recall": min((f["recall"] for f in folds), default=None),
        }

    return {
        "targets": {"suspect_recall": suspect_recall_target, "confirm_precision": confirm_precision_target},
        "suspect_threshold_candidate": suspect,
        "confirm_threshold_candidate": confirm,
        "suspect_operating_point": at(suspect),
        "confirm_operating_point": at(confirm),
        "thresholds_are_ordered": bool(suspect is not None and confirm is not None and confirm > suspect),
        # Always reported, so an unattainable CONFIRMED target is still described honestly:
        "highest_precision_operating_point": highest_precision_point(y, scores, table, grid),
    }


HIGHEST_PRECISION_MIN_RECALL = 0.10


def highest_precision_point(y: np.ndarray, scores: np.ndarray, table: dict, grid: Iterable[float]) -> dict | None:
    """The grid threshold with the highest precision that still keeps recall >= HIGHEST_PRECISION_MIN_RECALL."""
    eligible = [t for t in grid if table[t]["recall"] >= HIGHEST_PRECISION_MIN_RECALL and table[t]["true_positive"] > 0]
    if not eligible:
        return None
    best = max(eligible, key=lambda t: (table[t]["precision"], -t))
    metrics = table[best]
    return {
        "threshold": best,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "false_positive_rate": metrics["false_positive_rate"],
        "min_recall_required": HIGHEST_PRECISION_MIN_RECALL,
    }


# --- calibration ---


def calibration_report(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score, expected calibration error and equal-width reliability bins."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    prevalence = float(y.mean())
    brier = float(np.mean((scores - y) ** 2))
    baseline = prevalence * (1 - prevalence)  # Brier of always predicting the prevalence
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        upper_inclusive = index == n_bins - 1
        mask = (scores >= edges[index]) & ((scores <= edges[index + 1]) if upper_inclusive else (scores < edges[index + 1]))
        count = int(mask.sum())
        if count:
            mean_predicted = float(scores[mask].mean())
            observed = float(y[mask].mean())
            ece += count / len(y) * abs(mean_predicted - observed)
        else:
            mean_predicted = observed = None
        bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count,
                     "mean_predicted": mean_predicted, "observed_fire_fraction": observed})
    return {
        "brier_score": brier,
        "brier_score_constant_prevalence_baseline": baseline,
        "brier_skill_score": float(1 - brier / baseline) if baseline else None,
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


# --- interpretability ---


def grouped_permutation_importance(
    fitted_pipelines: Sequence,
    matrix: TrainingMatrixV4,
    splits: Sequence[NamedSplit],
    n_repeats: int = 5,
    seed: int = RANDOM_SEED_V4,
) -> dict:
    """Held-out permutation importance: drop in ROC-AUC on each validation fold when one feature is shuffled.

    The three Fire Danger features are shuffled TOGETHER as one block (`fire_danger_block`), so a row never
    becomes an impossible mixture such as available=0 with a real score. Every other feature is shuffled alone.
    """
    names = matrix.feature_names
    fire_danger = [names.index(name) for name in ("fire_danger_available", "fire_danger_score", "fire_danger_age_minutes")]
    units: dict[str, list[int]] = {"fire_danger_block": fire_danger}
    for index, name in enumerate(names):
        if index not in fire_danger:
            units[name] = [index]

    rng = np.random.default_rng(seed)
    drops = {unit: [] for unit in units}
    for pipeline, split in zip(fitted_pipelines, splits):
        X_val, y_val = matrix.X[split.validation], matrix.y[split.validation]
        baseline = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        for unit, columns in units.items():
            for _ in range(n_repeats):
                permuted = X_val.copy()
                permuted[:, columns] = X_val[rng.permutation(len(X_val))][:, columns]
                drops[unit].append(baseline - roc_auc_score(y_val, pipeline.predict_proba(permuted)[:, 1]))
    means = {unit: float(np.mean(values)) for unit, values in drops.items()}
    positive_total = sum(max(value, 0.0) for value in means.values())
    return {
        "metric": "drop in held-out ROC-AUC when the feature (or block) is shuffled",
        "features": {
            unit: {
                "mean_auc_drop": means[unit],
                "std_auc_drop": float(np.std(drops[unit])),
                "share_of_total_positive_drop": (max(means[unit], 0.0) / positive_total) if positive_total else 0.0,
            }
            for unit in sorted(means, key=means.get, reverse=True)
        },
    }


def importance_shortcut_summary(permutation: dict) -> dict:
    """Largest single-feature share, and the share held by the geometry/time features."""
    features = permutation["features"]
    shares = {name: entry["share_of_total_positive_drop"] for name, entry in features.items()}
    top = max(shares, key=shares.get)
    geometry = sum(shares.get(name, 0.0) for name in ("max_pairwise_distance_km", "time_span_minutes"))
    return {
        "top_feature": top,
        "top_feature_share": shares[top],
        "geometry_time_share": geometry,
        "fire_danger_block_share": shares.get("fire_danger_block", 0.0),
    }


def logistic_coefficient_report(fitted_pipelines: Sequence, feature_names: tuple[str, ...]) -> dict:
    """Standardised coefficients per fold and their sign stability (Logistic Regression only)."""
    names = transformed_feature_names(fitted_pipelines[0], feature_names)
    coefficients = np.array([p.named_steps["classifier"].coef_[0] for p in fitted_pipelines])
    mean = coefficients.mean(axis=0)
    stability = np.mean(np.sign(coefficients) == np.sign(mean), axis=0)
    ordered = np.argsort(-np.abs(mean))
    return {
        "note": "Coefficients of the standardised, median-imputed inputs (comparable magnitudes); mean over the grouped folds.",
        "features": [
            {"feature": names[i], "mean_coefficient": float(mean[i]), "std_across_folds": float(coefficients[:, i].std()),
             "same_sign_fraction_of_folds": float(stability[i])}
            for i in ordered
        ],
    }


def tree_importance_report(pipeline, feature_names: tuple[str, ...]) -> dict | None:
    """Impurity importances (Random Forest). Biased towards high-cardinality/continuous features; see the docs."""
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None
    names = transformed_feature_names(pipeline, feature_names)
    order = np.argsort(-classifier.feature_importances_)
    return {
        "note": "Mean-decrease-in-impurity on the full-data fit; biased towards continuous features. Prefer permutation importance.",
        "features": [{"feature": names[i], "importance": float(classifier.feature_importances_[i])} for i in order],
    }


# --- the gate and selection ---


def evaluate_gate(candidate: dict, rule: dict, gate: dict = GATE_V4) -> dict:
    """Apply the a-priori acceptance gate. `candidate` and `rule` hold the pre-computed numbers (see the compare script)."""

    def check(name, value, comparison, threshold, description):
        if value is None:
            passed = False
        elif comparison == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        return {"criterion": name, "description": description, "value": value, "comparison": comparison,
                "threshold": threshold, "passed": bool(passed)}

    suspect = candidate.get("suspect_operating_point")
    checks = [
        check("grouped_mean_roc_auc", candidate["grouped_mean_roc_auc"], ">=", gate["grouped_mean_roc_auc_min"],
              "mean ROC-AUC over grouped-family folds"),
        check("grouped_min_fold_roc_auc", candidate["grouped_min_fold_roc_auc"], ">=", gate["grouped_min_fold_roc_auc_min"],
              "worst grouped-family fold ROC-AUC"),
        check("grouped_mean_f1_minus_rule", candidate["grouped_mean_f1"] - rule["grouped_mean_f1"], ">=",
              gate["grouped_mean_f1_over_rule_min"], "mean grouped F1 (threshold 0.5) minus the rule baseline's"),
        check("suspect_recall", suspect["recall"] if suspect else None, ">=", gate["suspect_recall_min"],
              "recall at the recommended SUSPECTED threshold (out-of-fold)"),
        check("suspect_precision_minus_rule", (suspect["precision"] - rule["overall_precision"]) if suspect else None, ">=",
              gate["suspect_precision_over_rule_min"], "precision at the SUSPECTED threshold minus the rule baseline's precision"),
        check("hard_negative_fpr_at_suspect", candidate.get("hard_negative_fpr_at_suspect"), "<=",
              gate["hard_negative_fpr_at_suspect_max"], "false-positive rate on hard negatives at the SUSPECTED threshold"),
        check("weak_positive_recall_at_suspect", candidate.get("weak_positive_recall_at_suspect"), ">=",
              gate["weak_positive_recall_at_suspect_min"], "recall on weak/incomplete-evidence fires at the SUSPECTED threshold"),
        check("archetype_mean_roc_auc", candidate["archetype_mean_roc_auc"], ">=", gate["archetype_mean_roc_auc_min"],
              "mean ROC-AUC over leave-one-archetype-out folds"),
        check("archetype_worst_roc_auc", candidate["archetype_worst_roc_auc"], ">=", gate["archetype_worst_roc_auc_min"],
              "worst held-out archetype ROC-AUC"),
        check("calibration_ece", candidate["calibration_ece"], "<=", gate["calibration_ece_max"],
              "expected calibration error of out-of-fold probabilities"),
        check("max_single_feature_importance_share", candidate["top_feature_share"], "<=",
              gate["max_single_feature_importance_share"], "largest single feature's share of held-out permutation importance"),
        check("geometry_time_importance_share", candidate["geometry_time_share"], "<=",
              gate["max_geometry_importance_share"], "share of permutation importance held by distance + time-span"),
    ]
    return {"passed": all(item["passed"] for item in checks), "criteria": checks,
            "failed": [item["criterion"] for item in checks if not item["passed"]]}


def select_model(candidates: Sequence[dict]) -> dict:
    """Choose among gate-passing candidates; near-ties go to the simplest model. Never forces a winner.

    Each candidate needs: model_key, gate_passed, grouped_mean_roc_auc, grouped_mean_f1.
    """
    passing = [candidate for candidate in candidates if candidate["gate_passed"]]
    if not passing:
        return {
            "selected_model": None,
            "reason": "No model passed the generalisation gate; no runtime-ready artifact is justified.",
            "passing_models": [],
        }
    best = max(passing, key=lambda c: c["grouped_mean_roc_auc"])
    tied = [
        c for c in passing
        if best["grouped_mean_roc_auc"] - c["grouped_mean_roc_auc"] <= TIE_MARGIN_ROC_AUC + _TIE_EPSILON
        and best["grouped_mean_f1"] - c["grouped_mean_f1"] <= TIE_MARGIN_F1 + _TIE_EPSILON
    ]
    chosen = min(tied, key=lambda c: (MODEL_SPECS_V4[c["model_key"]].simplicity_rank, -c["grouped_mean_roc_auc"]))
    return {
        "selected_model": chosen["model_key"],
        "reason": (
            f"{chosen['model_key']} passed the gate and is within the tie margins "
            f"(ROC-AUC {TIE_MARGIN_ROC_AUC}, F1 {TIE_MARGIN_F1}) of the best passing model {best['model_key']}; "
            "the simplest tied model is preferred."
        ),
        "passing_models": [c["model_key"] for c in passing],
        "tied_models": [c["model_key"] for c in tied],
    }


def to_builtin(value):
    """Recursively convert numpy scalars/arrays to JSON-serialisable Python types (NaN -> None)."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
