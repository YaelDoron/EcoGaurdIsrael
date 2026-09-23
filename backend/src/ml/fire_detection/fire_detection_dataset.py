"""Load the Fire Detection ML training dataset CSV, including metadata columns.

Distinct from scripts.train_fire_detection_model.load_dataset, which loads
only the ML feature columns and label for training. This loader additionally
keeps sample_id/scenario_family for dataset auditing and analysis
(backend/scripts/analyze_fire_detection_dataset.py,
backend/scripts/compare_fire_detection_models.py) - metadata that must never
be fed to a classifier as a feature.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.ml.fire_detection.fire_detection_features import (
    FIRE_DETECTION_FEATURE_NAMES,
    LABEL_COLUMN,
    SAMPLE_ID_COLUMN,
    SCENARIO_FAMILY_COLUMN,
)


@dataclass(frozen=True)
class FireDetectionDatasetRow:
    """One row of the training dataset CSV, feature values in FIRE_DETECTION_FEATURE_NAMES order."""

    sample_id: int
    scenario_family: str
    features: tuple[float, ...]
    label: int


def load_training_dataset_rows(csv_path: Path) -> tuple[FireDetectionDatasetRow, ...]:
    """Load every row of csv_path, preserving CSV order (== original sample_id order)."""
    rows: list[FireDetectionDatasetRow] = []
    with Path(csv_path).open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                FireDetectionDatasetRow(
                    sample_id=int(row[SAMPLE_ID_COLUMN]),
                    scenario_family=row[SCENARIO_FAMILY_COLUMN],
                    features=tuple(float(row[name]) for name in FIRE_DETECTION_FEATURE_NAMES),
                    label=int(row[LABEL_COLUMN]),
                )
            )
    return tuple(rows)
