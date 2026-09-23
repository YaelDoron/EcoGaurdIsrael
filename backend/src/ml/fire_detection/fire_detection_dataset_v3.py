"""Load the Fire Detection ML V3 training dataset CSV, including metadata columns.

Sibling to fire_detection_dataset.py (V1/V2), which is unchanged. V3 adds
scenario_archetype metadata (Part 26) alongside sample_id/scenario_family.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.ml.fire_detection.fire_detection_features_v3 import (
    FIRE_DETECTION_FEATURE_NAMES_V3,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SCENARIO_ARCHETYPE_COLUMN,
    SCENARIO_FAMILY_COLUMN,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v3 import ScenarioArchetype


@dataclass(frozen=True)
class FireDetectionDatasetRowV3:
    """One row of the V3 training dataset CSV, feature values in FIRE_DETECTION_FEATURE_NAMES_V3 order."""

    sample_id: int
    scenario_family: str
    scenario_archetype: ScenarioArchetype
    features: tuple[float, ...]
    label: int


def load_training_dataset_rows_v3(csv_path: Path) -> tuple[FireDetectionDatasetRowV3, ...]:
    """Load every row of csv_path, preserving CSV order (== original sample_id order)."""
    rows: list[FireDetectionDatasetRowV3] = []
    with Path(csv_path).open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                FireDetectionDatasetRowV3(
                    sample_id=int(row[SAMPLE_ID_COLUMN]),
                    scenario_family=row[SCENARIO_FAMILY_COLUMN],
                    scenario_archetype=ScenarioArchetype(row[SCENARIO_ARCHETYPE_COLUMN]),
                    features=tuple(float(row[name]) for name in FIRE_DETECTION_FEATURE_NAMES_V3),
                    label=int(row[LABEL_COLUMN]),
                )
            )
    return tuple(rows)
