"""Write/load the Fire Detection ML V5 training dataset and build grouped evaluation splits.

Every feature value is produced by the canonical FireDetectionFeatureExtractorV5 - the same class future
runtime inference uses - from the synthetic candidate and its FireDetectionEventHistory. This module (and
the generator) computes NO feature itself, so training rows and runtime inputs cannot drift apart.

Missing values: features that are "not computable" (see fire_detection_features_v5) are written as EMPTY
cells and loaded as None (the storage form of the extractor's ML-facing NaN). They are never replaced by 0.
`FireDetectionDatasetRowV5.to_features_v5()` turns a stored row back into the validated, NaN-based contract.

Grouped evaluation support for Task 7 (no model is trained here):
  * `grouped_environment_folds`        - PRIMARY: whole environments held out, regime x label balanced
  * `leave_one_regime_out`             - stress test: an unseen evidence situation
  * `leave_one_no_fire_subtype_out`    - stress test: an unseen kind of false alarm
  * `pair_indices`                     - paired analysis: fire vs no-fire inside one pair_id
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path

from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_features_v5 import (
    AS_OF_UTC_COLUMN,
    ENVIRONMENT_ID_COLUMN,
    FIRE_DETECTION_FEATURE_NAMES_V5,
    LABEL_COLUMN,
    LATENT_SUBTYPE_COLUMN,
    NULLABLE_FEATURE_NAMES_V5,
    PAIR_ID_COLUMN,
    PAIR_TYPE_COLUMN,
    REGIME_COLUMN,
    SAMPLE_ID_COLUMN,
    SEED_COLUMN,
    TRAINING_DATA_CSV_COLUMNS_V5,
    FireDetectionFeaturesV5,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v5 import FireDetectionTrainingSampleV5

_EXTRACTOR = FireDetectionFeatureExtractorV5()


@dataclass(frozen=True)
class FireDetectionDatasetRowV5:
    """One V5 row. `features` follows FIRE_DETECTION_FEATURE_NAMES_V5; nullable ones may be None."""

    sample_id: int
    seed: int
    environment_id: int
    regime: str
    latent_subtype: str
    pair_id: int | None
    pair_type: str | None
    as_of_utc: datetime
    features: tuple[float | None, ...]
    label: int

    def feature(self, name: str) -> float | None:
        return self.features[FIRE_DETECTION_FEATURE_NAMES_V5.index(name)]

    def to_features_v5(self) -> FireDetectionFeaturesV5:
        """This row's features as the validated, NaN-based V5 contract (raises if the row violates it)."""
        return FireDetectionFeaturesV5.from_optional_values(self.features)


def extract_features_for_sample(sample: FireDetectionTrainingSampleV5) -> FireDetectionFeaturesV5:
    """The sample's V5 features, produced by the canonical (runtime) extractor."""
    return _EXTRACTOR.extract(sample.candidate, sample.history)


def row_from_sample(sample: FireDetectionTrainingSampleV5) -> FireDetectionDatasetRowV5:
    """Convert a generated sample to the same row object `load_training_dataset_rows_v5` returns."""
    return FireDetectionDatasetRowV5(
        sample_id=sample.sample_id,
        seed=sample.seed,
        environment_id=sample.environment_id,
        regime=sample.regime.value,
        latent_subtype=sample.latent_subtype.value,
        pair_id=sample.pair_id,
        pair_type=sample.pair_type,
        as_of_utc=sample.as_of,
        features=extract_features_for_sample(sample).to_nullable_tuple(),
        label=sample.label,
    )


def write_training_dataset_v5(samples: tuple[FireDetectionTrainingSampleV5, ...], csv_path: Path) -> Path:
    """Write samples as CSV (metadata + features + label). Missing values are empty cells."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")  # fixed line endings: byte-identical on every OS
        writer.writerow(TRAINING_DATA_CSV_COLUMNS_V5)
        for sample in samples:
            row = row_from_sample(sample)
            writer.writerow(
                [
                    row.sample_id,
                    row.seed,
                    row.environment_id,
                    row.regime,
                    row.latent_subtype,
                    "" if row.pair_id is None else row.pair_id,
                    "" if row.pair_type is None else row.pair_type,
                    row.as_of_utc.isoformat(),
                    *("" if value is None else value for value in row.features),
                    row.label,
                ]
            )
    return csv_path


def load_training_dataset_rows_v5(csv_path: Path) -> tuple[FireDetectionDatasetRowV5, ...]:
    """Load every row, preserving CSV order. Empty nullable cells load as None."""
    rows: list[FireDetectionDatasetRowV5] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if tuple(reader.fieldnames or ()) != TRAINING_DATA_CSV_COLUMNS_V5:
            raise ValueError(f"Unexpected V5 dataset columns: {reader.fieldnames!r}")
        for record in reader:
            features: list[float | None] = []
            for name in FIRE_DETECTION_FEATURE_NAMES_V5:
                cell = record[name]
                if cell == "":
                    if name not in NULLABLE_FEATURE_NAMES_V5:
                        raise ValueError(f"Feature {name!r} must not be empty (sample {record[SAMPLE_ID_COLUMN]}).")
                    features.append(None)
                else:
                    features.append(float(cell))
            rows.append(
                FireDetectionDatasetRowV5(
                    sample_id=int(record[SAMPLE_ID_COLUMN]),
                    seed=int(record[SEED_COLUMN]),
                    environment_id=int(record[ENVIRONMENT_ID_COLUMN]),
                    regime=record[REGIME_COLUMN],
                    latent_subtype=record[LATENT_SUBTYPE_COLUMN],
                    pair_id=int(record[PAIR_ID_COLUMN]) if record[PAIR_ID_COLUMN] != "" else None,
                    pair_type=record[PAIR_TYPE_COLUMN] or None,
                    as_of_utc=datetime.fromisoformat(record[AS_OF_UTC_COLUMN]),
                    features=tuple(features),
                    label=int(record[LABEL_COLUMN]),
                )
            )
    return tuple(rows)


def dataset_sha256(csv_path: Path) -> str:
    """SHA-256 of the dataset file's bytes (reproducibility fingerprint)."""
    return hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()


# --- grouped evaluation support (no model is trained here) -------------------------------------------

IndexFold = tuple[tuple[int, ...], tuple[int, ...]]  # (train row indices, validation row indices)


def _check_splits(n_splits: int) -> None:
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError(f"n_splits must be an integer >= 2, got {n_splits!r}")


def environment_fold_assignments(rows: tuple[FireDetectionDatasetRowV5, ...], n_splits: int = 5) -> dict[int, int]:
    """Deterministically map each environment_id to a fold in [0, n_splits).

    Whole environments are kept together (rows of one environment share nuisance parameters, so
    splitting them would leak). Environments are dealt greedily, largest first, to the fold whose
    regime x label counts end up closest to an even split - so every validation fold keeps every regime
    and both labels represented. Paired rows share an environment, hence a fold.
    """
    _check_splits(n_splits)
    strata_by_environment: dict[int, dict[tuple[str, int], int]] = {}
    totals: dict[tuple[str, int], int] = {}
    for row in rows:
        stratum = (row.regime, row.label)
        cell = strata_by_environment.setdefault(row.environment_id, {})
        cell[stratum] = cell.get(stratum, 0) + 1
        totals[stratum] = totals.get(stratum, 0) + 1
    target = {stratum: count / n_splits for stratum, count in totals.items()}

    fold_counts: list[dict[tuple[str, int], int]] = [{} for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    assignments: dict[int, int] = {}
    order = sorted(strata_by_environment, key=lambda env: (-sum(strata_by_environment[env].values()), env))
    for environment in order:
        cell = strata_by_environment[environment]
        best_fold, best_cost = 0, None
        for fold in range(n_splits):
            # increase of the squared deviation from the even split (NOT the absolute deviation, which
            # would keep filling the first fold until it reached its target)
            cost = sum(
                (fold_counts[fold].get(stratum, 0) + cell.get(stratum, 0) - target[stratum]) ** 2
                - (fold_counts[fold].get(stratum, 0) - target[stratum]) ** 2
                for stratum in totals
            )
            key = (cost, fold_sizes[fold], fold)
            if best_cost is None or key < best_cost:
                best_fold, best_cost = fold, key
        assignments[environment] = best_fold
        fold_sizes[best_fold] += sum(cell.values())
        for stratum, count in cell.items():
            fold_counts[best_fold][stratum] = fold_counts[best_fold].get(stratum, 0) + count
    return assignments


def grouped_environment_folds(rows: tuple[FireDetectionDatasetRowV5, ...], n_splits: int = 5) -> tuple[IndexFold, ...]:
    """PRIMARY split: GroupKFold-style by environment_id; validation environments are never seen in training."""
    assignments = environment_fold_assignments(rows, n_splits)
    folds: list[IndexFold] = []
    for fold in range(n_splits):
        train = tuple(i for i, row in enumerate(rows) if assignments[row.environment_id] != fold)
        validation = tuple(i for i, row in enumerate(rows) if assignments[row.environment_id] == fold)
        if not train or not validation:
            raise ValueError(f"Fold {fold} is empty; too few environments for n_splits={n_splits}.")
        folds.append((train, validation))
    return tuple(folds)


def leave_one_regime_out(rows: tuple[FireDetectionDatasetRowV5, ...]) -> tuple[tuple[str, IndexFold], ...]:
    """One (regime, (train, validation)) split per regime, sorted by name: an unseen evidence situation.

    Note: persistent_thermal is the only regime with history, so holding it out also tests extrapolation
    of the history features. Every regime holds both labels, so validation always has fire and no-fire.
    """
    splits: list[tuple[str, IndexFold]] = []
    for regime in sorted({row.regime for row in rows}):
        train = tuple(i for i, row in enumerate(rows) if row.regime != regime)
        validation = tuple(i for i, row in enumerate(rows) if row.regime == regime)
        splits.append((regime, (train, validation)))
    return tuple(splits)


def leave_one_no_fire_subtype_out(
    rows: tuple[FireDetectionDatasetRowV5, ...],
) -> tuple[tuple[str, IndexFold], ...]:
    """One (no-fire latent_subtype, (train, validation)) split per no-fire subtype, sorted by name.

    validation = ONLY the held-out no-fire rows (an unseen kind of false alarm; the metric of interest is
    the false-positive rate / specificity on it). Training keeps every other row, all fires included.
    """
    splits: list[tuple[str, IndexFold]] = []
    for subtype in sorted({row.latent_subtype for row in rows if row.label == 0}):
        train = tuple(i for i, row in enumerate(rows) if not (row.label == 0 and row.latent_subtype == subtype))
        validation = tuple(i for i, row in enumerate(rows) if row.label == 0 and row.latent_subtype == subtype)
        splits.append((subtype, (train, validation)))
    return tuple(splits)


def pair_indices(rows: tuple[FireDetectionDatasetRowV5, ...]) -> dict[int, tuple[int, int]]:
    """pair_id -> (fire row index, no-fire row index). Raises unless every pair is exactly one of each."""
    members: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        if row.pair_id is not None:
            members.setdefault(row.pair_id, []).append(index)
    pairs: dict[int, tuple[int, int]] = {}
    for pair_id, indices in sorted(members.items()):
        labels = sorted(rows[i].label for i in indices)
        if len(indices) != 2 or labels != [0, 1]:
            raise ValueError(f"pair_id {pair_id} must hold exactly one fire and one no-fire row, got labels {labels}.")
        fire = next(i for i in indices if rows[i].label == 1)
        pairs[pair_id] = (fire, next(i for i in indices if i != fire))
    return pairs
