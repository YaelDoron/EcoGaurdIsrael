"""Write/load the Fire Detection ML V4 training dataset and build grouped splits.

Every feature value is produced by the canonical FireDetectionFeatureExtractorV4
- the same class future runtime inference uses - from the synthetic candidate
and its FireDetectionContext. This module computes NO feature itself, so
training rows and runtime inputs cannot drift apart (no train/serving skew).

Missing Fire Danger: `fire_danger_score` and `fire_danger_age_minutes` are
written as EMPTY cells and loaded as None (the storage form of the extractor's
ML-facing NaN). They are never replaced by 0 (a real "very low danger" FFWI), so
the distinction survives until Task 4's preprocessing deliberately imputes
(using `fire_danger_available`). `FireDetectionDatasetRowV4.to_features_v4()`
turns a stored row back into the validated, NaN-based feature contract.

Grouped evaluation support (no model is trained here):
  * `grouped_family_folds`     - GroupKFold-style, by `scenario_family`
  * `leave_one_archetype_out`  - by `scenario_archetype`
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.ml.fire_detection.fire_detection_feature_extractor_v4 import FireDetectionFeatureExtractorV4
from src.ml.fire_detection.fire_detection_features_v4 import (
    AS_OF_UTC_COLUMN,
    FIRE_DANGER_BAND_COLUMN,
    FIRE_DANGER_BAND_MISSING,
    FIRE_DETECTION_FEATURE_NAMES_V4,
    FireDetectionFeaturesV4,
    GROUND_TRUTH_SCENARIO_TYPE_COLUMN,
    LABEL_COLUMN,
    NULLABLE_FEATURE_NAMES_V4,
    SAMPLE_ID_COLUMN,
    SCENARIO_ARCHETYPE_COLUMN,
    SCENARIO_FAMILY_COLUMN,
    SEED_COLUMN,
    TRAINING_DATA_CSV_COLUMNS_V4,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import FireDetectionTrainingSampleV4

_EXTRACTOR = FireDetectionFeatureExtractorV4()


@dataclass(frozen=True)
class FireDetectionDatasetRowV4:
    """One V4 row. `features` follows FIRE_DETECTION_FEATURE_NAMES_V4; nullable ones may be None."""

    sample_id: int
    seed: int
    scenario_family: str
    scenario_archetype: str
    ground_truth_scenario_type: str
    fire_danger_band: str
    as_of_utc: datetime
    features: tuple[float | None, ...]
    label: int

    def feature(self, name: str) -> float | None:
        return self.features[FIRE_DETECTION_FEATURE_NAMES_V4.index(name)]

    def to_features_v4(self) -> FireDetectionFeaturesV4:
        """This row's features as the validated, NaN-based V4 contract (raises if the row violates it)."""
        return FireDetectionFeaturesV4.from_optional_values(self.features)


def extract_features_for_sample(sample: FireDetectionTrainingSampleV4) -> FireDetectionFeaturesV4:
    """The sample's V4 features, produced by the canonical (runtime) extractor."""
    return _EXTRACTOR.extract(sample.evidence, sample.context)


def feature_values_for_sample(sample: FireDetectionTrainingSampleV4) -> dict[str, float | None]:
    """The 19 V4 feature values for one sample in storage form (None where Fire Danger is unavailable)."""
    return extract_features_for_sample(sample).to_nullable_dict()


def row_from_sample(sample: FireDetectionTrainingSampleV4) -> FireDetectionDatasetRowV4:
    """Convert a generated sample to the same row object `load_training_dataset_rows_v4` returns."""
    features = extract_features_for_sample(sample).to_nullable_tuple()
    level = sample.context.fire_danger_level
    return FireDetectionDatasetRowV4(
        sample_id=sample.sample_id,
        seed=sample.seed,
        scenario_family=sample.scenario_family,
        scenario_archetype=sample.scenario_archetype.value,
        ground_truth_scenario_type=sample.ground_truth_scenario_type.value,
        fire_danger_band=level.value if level is not None else FIRE_DANGER_BAND_MISSING,
        as_of_utc=sample.as_of,
        features=features,
        label=sample.label,
    )


def write_training_dataset_v4(samples: tuple[FireDetectionTrainingSampleV4, ...], csv_path: Path) -> Path:
    """Write samples as CSV (metadata + features + label). Missing values are empty cells."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(TRAINING_DATA_CSV_COLUMNS_V4)
        for sample in samples:
            row = row_from_sample(sample)
            writer.writerow(
                [
                    row.sample_id,
                    row.seed,
                    row.scenario_family,
                    row.scenario_archetype,
                    row.ground_truth_scenario_type,
                    row.fire_danger_band,
                    row.as_of_utc.isoformat(),
                    *("" if value is None else value for value in row.features),
                    row.label,
                ]
            )
    return csv_path


def load_training_dataset_rows_v4(csv_path: Path) -> tuple[FireDetectionDatasetRowV4, ...]:
    """Load every row, preserving CSV order. Empty nullable cells load as None."""
    rows: list[FireDetectionDatasetRowV4] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != TRAINING_DATA_CSV_COLUMNS_V4:
            raise ValueError(f"Unexpected V4 dataset columns: {reader.fieldnames!r}")
        for record in reader:
            features: list[float | None] = []
            for name in FIRE_DETECTION_FEATURE_NAMES_V4:
                cell = record[name]
                if cell == "":
                    if name not in NULLABLE_FEATURE_NAMES_V4:
                        raise ValueError(f"Feature {name!r} must not be empty (sample {record[SAMPLE_ID_COLUMN]}).")
                    features.append(None)
                else:
                    features.append(float(cell))
            rows.append(
                FireDetectionDatasetRowV4(
                    sample_id=int(record[SAMPLE_ID_COLUMN]),
                    seed=int(record[SEED_COLUMN]),
                    scenario_family=record[SCENARIO_FAMILY_COLUMN],
                    scenario_archetype=record[SCENARIO_ARCHETYPE_COLUMN],
                    ground_truth_scenario_type=record[GROUND_TRUTH_SCENARIO_TYPE_COLUMN],
                    fire_danger_band=record[FIRE_DANGER_BAND_COLUMN],
                    as_of_utc=datetime.fromisoformat(record[AS_OF_UTC_COLUMN]),
                    features=tuple(features),
                    label=int(record[LABEL_COLUMN]),
                )
            )
    return tuple(rows)


# --- grouped evaluation support (no model is trained here) ---

IndexFold = tuple[tuple[int, ...], tuple[int, ...]]  # (train row indices, validation row indices)


def family_fold_assignments(
    rows: tuple[FireDetectionDatasetRowV4, ...],
    n_splits: int = 5,
) -> dict[str, int]:
    """Deterministically map each scenario_family to a fold in [0, n_splits).

    Every family has exactly one label, so plain GroupKFold can produce folds
    containing a single class. Fire and no-fire families are therefore dealt
    round-robin separately (sorted by name), so each validation fold holds
    whole, unseen families of BOTH labels whenever there are >= n_splits
    families per label.
    """
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")
    label_by_family: dict[str, int] = {}
    for row in rows:
        previous = label_by_family.setdefault(row.scenario_family, row.label)
        if previous != row.label:
            raise ValueError(f"scenario_family {row.scenario_family!r} contains both labels.")
    assignments: dict[str, int] = {}
    for label in (1, 0):
        families = sorted(family for family, family_label in label_by_family.items() if family_label == label)
        for rank, family in enumerate(families):
            assignments[family] = rank % n_splits
    return assignments


def grouped_family_folds(rows: tuple[FireDetectionDatasetRowV4, ...], n_splits: int = 5) -> tuple[IndexFold, ...]:
    """GroupKFold-style folds by scenario_family: validation families are never seen in training."""
    assignments = family_fold_assignments(rows, n_splits)
    folds: list[IndexFold] = []
    for fold in range(n_splits):
        train = tuple(index for index, row in enumerate(rows) if assignments[row.scenario_family] != fold)
        validation = tuple(index for index, row in enumerate(rows) if assignments[row.scenario_family] == fold)
        if not train or not validation:
            raise ValueError(f"Fold {fold} is empty; too few scenario families for n_splits={n_splits}.")
        folds.append((train, validation))
    return tuple(folds)


def leave_one_archetype_out(
    rows: tuple[FireDetectionDatasetRowV4, ...],
) -> tuple[tuple[str, IndexFold], ...]:
    """One (archetype, (train, validation)) split per scenario_archetype, sorted by name."""
    splits: list[tuple[str, IndexFold]] = []
    for archetype in sorted({row.scenario_archetype for row in rows}):
        train = tuple(index for index, row in enumerate(rows) if row.scenario_archetype != archetype)
        validation = tuple(index for index, row in enumerate(rows) if row.scenario_archetype == archetype)
        splits.append((archetype, (train, validation)))
    return tuple(splits)
