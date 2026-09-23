"""Feature-set ablation helpers for the satellite_count redundancy experiment.

satellite_count == satellite_low_count + satellite_nominal_count +
satellite_high_count holds exactly for every row (see
fire_detection_dataset_statistics.satellite_count_identity_holds), which is
an exact structural redundancy. This module supports comparing the full
7-feature set against a 6-feature set without satellite_count - an
experiment only; it does not change FIRE_DETECTION_FEATURE_NAMES, the
official production feature set.
"""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES

ABLATION_FEATURE_SET_FULL: tuple[str, ...] = FIRE_DETECTION_FEATURE_NAMES
ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT: tuple[str, ...] = tuple(
    name for name in FIRE_DETECTION_FEATURE_NAMES if name != "satellite_count"
)


def select_feature_columns(
    features: list[list[float]],
    feature_names: tuple[str, ...],
    keep: tuple[str, ...],
) -> list[list[float]]:
    """Return features restricted to the columns in keep, preserving keep's order.

    feature_names must be the column order features rows are currently in
    (normally FIRE_DETECTION_FEATURE_NAMES); keep must be a subset of it.
    """
    missing = [name for name in keep if name not in feature_names]
    if missing:
        raise ValueError(f"keep contains names not present in feature_names: {missing}")
    keep_indices = [feature_names.index(name) for name in keep]
    return [[row[index] for index in keep_indices] for row in features]
