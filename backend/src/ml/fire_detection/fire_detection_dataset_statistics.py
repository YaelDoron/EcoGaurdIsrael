"""Reusable descriptive statistics for auditing the Fire Detection ML dataset.

Pure analysis helpers - no model training, no runtime dependency. Used by
backend/scripts/analyze_fire_detection_dataset.py and by
backend/scripts/compare_fire_detection_models.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from src.ml.fire_detection.fire_detection_dataset import FireDetectionDatasetRow
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES

_FEATURE_INDEX = {name: index for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES)}


@dataclass(frozen=True)
class FeatureLabelStatistics:
    """Descriptive statistics for one feature, restricted to one label's rows."""

    feature_name: str
    label: int
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float


def compute_feature_statistics_by_label(
    rows: tuple[FireDetectionDatasetRow, ...],
) -> tuple[FeatureLabelStatistics, ...]:
    """Return count/mean/median/std/min/max per feature, separately for label=0 and label=1."""
    results: list[FeatureLabelStatistics] = []
    for feature_index, feature_name in enumerate(FIRE_DETECTION_FEATURE_NAMES):
        for label in (0, 1):
            values = [row.features[feature_index] for row in rows if row.label == label]
            if not values:
                continue
            results.append(
                FeatureLabelStatistics(
                    feature_name=feature_name,
                    label=label,
                    count=len(values),
                    mean=statistics.mean(values),
                    median=statistics.median(values),
                    std=statistics.pstdev(values) if len(values) > 1 else 0.0,
                    minimum=min(values),
                    maximum=max(values),
                )
            )
    return tuple(results)


@dataclass(frozen=True)
class ScenarioFamilyStatistics:
    """Descriptive statistics for one scenario_family/label group (auditing only - never a feature)."""

    scenario_family: str
    label: int
    count: int
    mean_features: dict[str, float]


def compute_scenario_family_statistics(
    rows: tuple[FireDetectionDatasetRow, ...],
) -> tuple[ScenarioFamilyStatistics, ...]:
    """Return sample count, label, and mean feature values per scenario family."""
    grouped: dict[tuple[str, int], list[FireDetectionDatasetRow]] = {}
    for row in rows:
        grouped.setdefault((row.scenario_family, row.label), []).append(row)

    results = [
        ScenarioFamilyStatistics(
            scenario_family=family,
            label=label,
            count=len(family_rows),
            mean_features={
                name: statistics.mean(row.features[index] for row in family_rows)
                for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES)
            },
        )
        for (family, label), family_rows in grouped.items()
    ]
    return tuple(sorted(results, key=lambda item: item.scenario_family))


def compute_feature_correlation_matrix(
    rows: tuple[FireDetectionDatasetRow, ...],
) -> dict[str, dict[str, float]]:
    """Return the Pearson correlation matrix between the numeric ML features."""
    columns = {
        name: [row.features[index] for row in rows] for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES)
    }
    return {
        first_name: {
            second_name: _pearson_correlation(columns[first_name], columns[second_name])
            for second_name in FIRE_DETECTION_FEATURE_NAMES
        }
        for first_name in FIRE_DETECTION_FEATURE_NAMES
    }


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denominator_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if denominator_x == 0.0 or denominator_y == 0.0:
        return 0.0
    return numerator / (denominator_x * denominator_y)


def satellite_count_identity_holds(rows: tuple[FireDetectionDatasetRow, ...]) -> bool:
    """Confirm the structural (exact) linear dependency: satellite_count == low+nominal+high for every row.

    This is a *structural* redundancy (guaranteed by FireDetectionFeatures'
    own validation, not merely a statistical correlation), which is the root
    cause of Logistic Regression multicollinearity between satellite_count
    and the three confidence-specific counts (see fire_detection_ml.md).
    """
    count_index = _FEATURE_INDEX["satellite_count"]
    low_index = _FEATURE_INDEX["satellite_low_count"]
    nominal_index = _FEATURE_INDEX["satellite_nominal_count"]
    high_index = _FEATURE_INDEX["satellite_high_count"]
    return all(
        row.features[count_index] == row.features[low_index] + row.features[nominal_index] + row.features[high_index]
        for row in rows
    )


def scenario_family_perfectly_predicts_label(rows: tuple[FireDetectionDatasetRow, ...]) -> bool:
    """Return True if every scenario_family maps to exactly one label (audit only, not a feature).

    A True result is expected and safe: scenario_family is deliberately
    excluded from FIRE_DETECTION_FEATURE_NAMES, so this potential leakage
    never reaches the model. It only documents *why* scenario_family must
    stay metadata-only.
    """
    labels_by_family: dict[str, set[int]] = {}
    for row in rows:
        labels_by_family.setdefault(row.scenario_family, set()).add(row.label)
    return all(len(labels) == 1 for labels in labels_by_family.values())
