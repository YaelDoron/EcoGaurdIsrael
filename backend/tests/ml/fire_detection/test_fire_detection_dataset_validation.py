"""Tests for validate_training_samples (Part 9 sanity checks)."""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_dataset_validation import validate_training_samples
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V1,
    DATASET_VERSION_V2,
    FireDetectionTrainingDataGenerator,
)

SAMPLE_COUNT = 600


def test_v2_dataset_passes_all_sanity_checks():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)

    report = validate_training_samples(samples)

    assert report.is_valid, report.violations


def test_v1_dataset_fails_the_low_confidence_overlap_check():
    """Confirms the validator actually detects the known V1 generator artifact (not a rubber stamp)."""
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SAMPLE_COUNT, dataset_version=DATASET_VERSION_V1)

    report = validate_training_samples(samples)

    assert not report.is_valid
    assert any("satellite_low_count" in violation for violation in report.violations)


def test_report_is_valid_property_reflects_empty_violations():
    from src.ml.fire_detection.fire_detection_dataset_validation import DatasetValidationReport

    assert DatasetValidationReport(violations=()).is_valid is True
    assert DatasetValidationReport(violations=("something wrong",)).is_valid is False


def test_empty_sample_list_reports_a_violation_instead_of_crashing():
    report = validate_training_samples(())

    assert not report.is_valid
