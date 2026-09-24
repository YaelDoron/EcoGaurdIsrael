"""Tests for V4 dataset CSV I/O, missing-value preservation, the committed file, and grouped splits."""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.ml.fire_detection.fire_detection_dataset_v4 import (
    family_fold_assignments,
    feature_values_for_sample,
    grouped_family_folds,
    leave_one_archetype_out,
    load_training_dataset_rows_v4,
    row_from_sample,
    write_training_dataset_v4,
)
from src.ml.fire_detection.fire_detection_dataset_validation_v4 import validate_dataset_rows_v4
from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DETECTION_FEATURE_NAMES_V4,
    LABEL_COLUMN,
    TRAINING_DATA_CSV_COLUMNS_V4,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v4 import (
    DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4,
    DEFAULT_TRAINING_DATA_SEED_V4,
    FireDetectionTrainingDataGeneratorV4,
    ScenarioArchetypeV4,
)
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_context import FireDetectionContext

COMMITTED_CSV = Path(__file__).resolve().parents[3] / "data" / "fire_detection" / "training_v4.csv"


def _sample_with_context(samples, context):
    sample = samples[0]
    return replace(sample, context=context)


def _normalize_newlines(data: bytes) -> bytes:
    """CRLF -> LF, since git may normalize line endings on checkout."""
    return data.replace(b"\r\n", b"\n")


# --- CSV round trip and missingness ---


def test_csv_round_trip_preserves_every_row(tmp_path, v4_samples):
    subset = v4_samples[:300]
    path = write_training_dataset_v4(subset, tmp_path / "v4.csv")

    loaded = load_training_dataset_rows_v4(path)

    assert loaded == tuple(row_from_sample(sample) for sample in subset)


def test_csv_header_is_metadata_then_features_then_label(tmp_path, v4_samples):
    path = write_training_dataset_v4(v4_samples[:5], tmp_path / "v4.csv")

    with path.open(encoding="utf-8", newline="") as csv_file:
        header = next(csv.reader(csv_file))

    assert tuple(header) == TRAINING_DATA_CSV_COLUMNS_V4
    assert header[-1] == LABEL_COLUMN


def test_missing_fire_danger_is_written_as_empty_cells_and_loaded_as_none(tmp_path, v4_samples):
    unavailable = next(sample for sample in v4_samples if not sample.context.fire_danger_available)
    path = write_training_dataset_v4((unavailable,), tmp_path / "v4.csv")

    with path.open(encoding="utf-8", newline="") as csv_file:
        record = next(csv.DictReader(csv_file))
    row = load_training_dataset_rows_v4(path)[0]

    assert record["fire_danger_available"] == "0"
    assert record["fire_danger_score"] == "" and record["fire_danger_age_minutes"] == ""
    assert row.feature("fire_danger_available") == 0
    assert row.feature("fire_danger_score") is None
    assert row.feature("fire_danger_age_minutes") is None
    assert row.fire_danger_band == "missing"


def test_a_real_zero_ffwi_stays_distinct_from_missing(tmp_path, v4_samples):
    zero_context = FireDetectionContext(
        fire_danger_available=True,
        fire_danger_score=0.0,
        fire_danger_age_minutes=12.5,
        fire_danger_level=FireDangerLevel.LOW,
        fire_danger_assessment_id=1,
        fire_danger_assessed_at=v4_samples[0].as_of - timedelta(minutes=12.5),
    )
    zero = _sample_with_context(v4_samples, zero_context)
    missing = _sample_with_context(v4_samples, FireDetectionContext.unavailable())
    path = write_training_dataset_v4((zero, missing), tmp_path / "v4.csv")

    zero_row, missing_row = load_training_dataset_rows_v4(path)

    assert zero_row.feature("fire_danger_score") == 0.0
    assert zero_row.feature("fire_danger_available") == 1
    assert missing_row.feature("fire_danger_score") is None
    assert missing_row.feature("fire_danger_available") == 0
    assert zero_row.features != missing_row.features


def test_loader_rejects_an_empty_non_nullable_feature(tmp_path, v4_samples):
    path = write_training_dataset_v4(v4_samples[:2], tmp_path / "v4.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    cells = lines[1].split(",")
    cells[header.index("satellite_low_count")] = ""
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_training_dataset_rows_v4(path)


def test_loader_rejects_an_unexpected_schema(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_training_dataset_rows_v4(path)


def test_ml_features_never_contain_datetimes_or_strings(v4_samples):
    for sample in v4_samples[:300]:
        for value in feature_values_for_sample(sample).values():
            assert value is None or isinstance(value, (int, float))


# --- the committed dataset file ---


def test_committed_dataset_is_reproducible_from_the_default_seed(tmp_path):
    samples = FireDetectionTrainingDataGeneratorV4(seed=DEFAULT_TRAINING_DATA_SEED_V4).generate(
        DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4
    )
    regenerated = write_training_dataset_v4(samples, tmp_path / "regenerated.csv")

    # Compare ignoring CRLF/LF (git may normalize line endings on checkout).
    assert _normalize_newlines(regenerated.read_bytes()) == _normalize_newlines(COMMITTED_CSV.read_bytes())


def test_committed_dataset_passes_validation():
    rows = load_training_dataset_rows_v4(COMMITTED_CSV)

    report = validate_dataset_rows_v4(rows, min_rows=5000)

    assert len(rows) == DEFAULT_TRAINING_DATA_SAMPLE_COUNT_V4
    assert report.is_valid, report.violations


def test_v3_dataset_files_are_untouched_by_v4():
    v3_csv = COMMITTED_CSV.with_name("training_v3.csv")

    assert v3_csv.exists()
    with v3_csv.open(encoding="utf-8", newline="") as csv_file:
        header = next(csv.reader(csv_file))
    assert "fire_danger_score" not in header


# --- grouped evaluation support ---


def test_family_folds_hold_out_whole_families_and_cover_every_row_once(v4_rows):
    folds = grouped_family_folds(v4_rows, n_splits=5)

    assert len(folds) == 5
    validation_indices = [index for _, validation in folds for index in validation]
    assert sorted(validation_indices) == list(range(len(v4_rows)))
    for train, validation in folds:
        train_families = {v4_rows[i].scenario_family for i in train}
        validation_families = {v4_rows[i].scenario_family for i in validation}
        assert not (train_families & validation_families)
        assert set(train) | set(validation) == set(range(len(v4_rows)))
        assert not (set(train) & set(validation))


def test_every_family_fold_validates_on_both_labels(v4_rows):
    for _, validation in grouped_family_folds(v4_rows, n_splits=5):
        labels = {v4_rows[i].label for i in validation}
        assert labels == {0, 1}


def test_family_fold_assignment_is_deterministic_and_label_balanced(v4_rows):
    first = family_fold_assignments(v4_rows, 5)
    second = family_fold_assignments(tuple(reversed(v4_rows)), 5)

    assert first == second
    label_by_family = {row.scenario_family: row.label for row in v4_rows}
    for fold in range(5):
        fire = sum(1 for family, assigned in first.items() if assigned == fold and label_by_family[family] == 1)
        no_fire = sum(1 for family, assigned in first.items() if assigned == fold and label_by_family[family] == 0)
        assert fire == no_fire == 2


def test_leave_one_archetype_out_holds_out_each_archetype_exactly(v4_rows):
    splits = leave_one_archetype_out(v4_rows)

    assert {archetype for archetype, _ in splits} == {archetype.value for archetype in ScenarioArchetypeV4}
    for archetype, (train, validation) in splits:
        assert {v4_rows[i].scenario_archetype for i in validation} == {archetype}
        assert all(v4_rows[i].scenario_archetype != archetype for i in train)
        assert len(train) + len(validation) == len(v4_rows)
        assert {v4_rows[i].label for i in validation} == {0, 1}  # both labels are present in every held-out archetype


def test_family_fold_assignment_rejects_a_family_that_mixes_labels(v4_rows):
    flipped = replace(v4_rows[0], label=1 - v4_rows[0].label)

    with pytest.raises(ValueError):
        family_fold_assignments((v4_rows[0], flipped), 2)


@pytest.mark.parametrize("bad", [1, 0, True, 2.5])
def test_family_fold_assignment_rejects_invalid_split_counts(v4_rows, bad):
    with pytest.raises(ValueError):
        family_fold_assignments(v4_rows, bad)


def test_group_metadata_columns_are_available_for_splitting(v4_rows):
    assert all(row.scenario_family and row.scenario_archetype for row in v4_rows)
    assert all(isinstance(row.as_of_utc, datetime) and row.as_of_utc.tzinfo == timezone.utc for row in v4_rows)
    assert len(FIRE_DETECTION_FEATURE_NAMES_V4) == len(v4_rows[0].features)
