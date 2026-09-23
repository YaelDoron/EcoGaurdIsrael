"""Grouped (unseen-scenario-family) cross-validation for Fire Detection ML models.

Standard StratifiedKFold (see fire_detection_model_comparison) answers "can
the model generalize to new samples drawn from scenario families it has
already seen during training?" Grouped-by-scenario_family evaluation answers
the harder question: "can the model generalize to an entire scenario family
it has never seen during training at all?" scenario_family is the CV group,
never a model feature - this module only ever reads it as metadata to build
folds and to report per-family results.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from src.ml.fire_detection.fire_detection_model_comparison import CV_RANDOM_STATE, CV_SCORING, evaluate_held_out

MAX_GROUPED_FOLD_CANDIDATE = 10
MIN_GROUPED_FOLD_CANDIDATE = 3


@dataclass(frozen=True)
class GroupedFoldResult:
    """One grouped-CV fold: which families were held out, and its metrics."""

    fold_index: int
    train_families: tuple[str, ...]
    validation_families: tuple[str, ...]
    metrics: dict


@dataclass(frozen=True)
class FamilyValidationSummary:
    """How a model performed on one held-out group (scenario_family or scenario_archetype).

    `label` is the single shared ground-truth label when every row in the
    group has the same one (always true for scenario_family groups) - it is
    None when the group legitimately mixes both labels (true for
    scenario_archetype groups, which are designed to contain both).
    `actual_positive_fraction` is meaningful in both cases: it equals
    exactly 0.0 or 1.0 when the group is label-pure, matching `label`, and a
    genuine fraction when mixed.
    """

    scenario_family: str
    label: int | None
    actual_positive_fraction: float
    sample_count: int
    predicted_positive_rate: float
    accuracy: float


@dataclass(frozen=True)
class GroupedCrossValidationResult:
    n_splits: int
    fold_results: tuple[GroupedFoldResult, ...]
    summary: dict  # metric -> {"mean": ..., "std": ...}, same shape as evaluate_cross_validation
    per_family: tuple[FamilyValidationSummary, ...]


def choose_grouped_fold_count(
    labels: list[int],
    groups: list[str],
    max_candidate: int = MAX_GROUPED_FOLD_CANDIDATE,
    min_candidate: int = MIN_GROUPED_FOLD_CANDIDATE,
) -> int:
    """Return the largest fold count in [min_candidate, max_candidate] that is safe.

    "Safe" means: every family has at least n_splits distinct families of its
    own label available (a necessary condition for StratifiedGroupKFold to
    have any chance of balancing), the splitter does not raise, and every
    fold's train/validation sets each contain both labels with zero group
    overlap. Raises ValueError if no candidate in range is safe.
    """
    for candidate in range(max_candidate, min_candidate - 1, -1):
        if _is_safe_grouped_fold_count(labels, groups, candidate):
            return candidate
    raise ValueError(
        f"No grouped fold count between {min_candidate} and {max_candidate} keeps both labels "
        "represented, with zero family overlap, in every fold's train and validation sets."
    )


def _is_safe_grouped_fold_count(labels: list[int], groups: list[str], n_splits: int) -> bool:
    families_by_label: dict[int, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups):
        families_by_label[label].add(group)
    if any(len(families) < n_splits for families in families_by_label.values()):
        return False

    try:
        folds = list(build_grouped_splitter(n_splits).split([[0.0]] * len(labels), labels, groups))
    except ValueError:
        return False

    return all(_fold_is_valid(labels, groups, train_index, val_index) for train_index, val_index in folds)


def _fold_is_valid(labels: list[int], groups: list[str], train_index, val_index) -> bool:
    train_labels = {labels[i] for i in train_index}
    val_labels = {labels[i] for i in val_index}
    if train_labels != {0, 1} or val_labels != {0, 1}:
        return False
    train_groups = {groups[i] for i in train_index}
    val_groups = {groups[i] for i in val_index}
    return not (train_groups & val_groups)


def build_grouped_splitter(n_splits: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)


def verify_grouped_folds(labels: list[int], groups: list[str], n_splits: int) -> None:
    """Raise AssertionError if any fold violates group-disjointness or label coverage.

    Used both by scripts.compare_fire_detection_models_v2 before trusting
    its own chosen fold count, and directly by tests (Part 14).
    """
    splitter = build_grouped_splitter(n_splits)
    for fold_index, (train_index, val_index) in enumerate(splitter.split([[0.0]] * len(labels), labels, groups)):
        train_groups = {groups[i] for i in train_index}
        val_groups = {groups[i] for i in val_index}
        overlap = train_groups & val_groups
        assert not overlap, f"fold {fold_index}: train/validation scenario families overlap: {overlap}"
        assert {labels[i] for i in train_index} == {0, 1}, f"fold {fold_index}: train set missing a label"
        assert {labels[i] for i in val_index} == {0, 1}, f"fold {fold_index}: validation set missing a label"


def run_grouped_cross_validation(
    estimator_factory: Callable[[], object],
    features: list[list[float]],
    labels: list[int],
    groups: list[str],
    n_splits: int,
) -> GroupedCrossValidationResult:
    """Fit/evaluate a fresh estimator per fold, with zero scenario-family overlap between
    a fold's train and validation sets. estimator_factory must return an unfitted estimator."""
    splitter = build_grouped_splitter(n_splits)
    folds = list(splitter.split(features, labels, groups))
    return _run_cross_validation_over_folds(estimator_factory, features, labels, groups, folds, n_splits)


def run_leave_one_group_out_cross_validation(
    estimator_factory: Callable[[], object],
    features: list[list[float]],
    labels: list[int],
    groups: list[str],
) -> GroupedCrossValidationResult:
    """One fold per distinct group value, each holding exactly that one group out entirely.

    Used for the archetype-holdout robustness check (Part 36): "can the model
    generalize to an evidence composition/archetype it has never seen?"
    estimator_factory must return an unfitted estimator.
    """
    splitter = LeaveOneGroupOut()
    folds = list(splitter.split(features, labels, groups))
    return _run_cross_validation_over_folds(estimator_factory, features, labels, groups, folds, len(folds))


def _run_cross_validation_over_folds(
    estimator_factory: Callable[[], object],
    features: list[list[float]],
    labels: list[int],
    groups: list[str],
    folds: list,
    n_splits: int,
) -> GroupedCrossValidationResult:
    fold_results: list[GroupedFoldResult] = []
    family_predictions: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for fold_index, (train_index, val_index) in enumerate(folds):
        x_train = [features[i] for i in train_index]
        y_train = [labels[i] for i in train_index]
        x_val = [features[i] for i in val_index]
        y_val = [labels[i] for i in val_index]

        estimator = estimator_factory()
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_val)
        metrics = evaluate_held_out(estimator, x_val, y_val)

        train_families = tuple(sorted({groups[i] for i in train_index}))
        validation_families = tuple(sorted({groups[i] for i in val_index}))
        fold_results.append(
            GroupedFoldResult(
                fold_index=fold_index,
                train_families=train_families,
                validation_families=validation_families,
                metrics=metrics,
            )
        )

        for row_index, prediction in zip(val_index, y_pred):
            family_predictions[groups[row_index]].append((labels[row_index], int(prediction)))

    summary = {
        metric: {
            "mean": float(np.mean([fold.metrics[metric] for fold in fold_results])),
            "std": float(np.std([fold.metrics[metric] for fold in fold_results])),
        }
        for metric in CV_SCORING
    }
    per_family = _summarize_family_predictions(family_predictions)

    return GroupedCrossValidationResult(
        n_splits=n_splits, fold_results=tuple(fold_results), summary=summary, per_family=per_family
    )


def _summarize_family_predictions(
    family_predictions: dict[str, list[tuple[int, int]]],
) -> tuple[FamilyValidationSummary, ...]:
    results = []
    for family, rows in family_predictions.items():
        sample_count = len(rows)
        true_labels_seen = {true for true, _ in rows}
        label = next(iter(true_labels_seen)) if len(true_labels_seen) == 1 else None
        actual_positive_fraction = sum(true for true, _ in rows) / sample_count
        predicted_positive_rate = sum(predicted for _, predicted in rows) / sample_count
        accuracy = sum(1 for true, predicted in rows if true == predicted) / sample_count
        results.append(
            FamilyValidationSummary(
                scenario_family=family,
                label=label,
                actual_positive_fraction=actual_positive_fraction,
                sample_count=sample_count,
                predicted_positive_rate=predicted_positive_rate,
                accuracy=accuracy,
            )
        )
    return tuple(sorted(results, key=lambda item: item.scenario_family))
