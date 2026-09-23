"""Tests for validate_training_samples_v3 (Part 27 sanity checks)."""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_dataset_validation_v3 import (
    DatasetValidationReportV3,
    validate_training_samples_v3,
)
from src.ml.fire_detection.fire_detection_training_data_generator_v3 import FireDetectionTrainingDataGeneratorV3

SAMPLE_COUNT = 1000


def test_v3_dataset_passes_all_sanity_checks():
    samples = FireDetectionTrainingDataGeneratorV3(seed=42).generate(num_samples=SAMPLE_COUNT)

    report = validate_training_samples_v3(samples)

    assert report.is_valid, report.violations


def test_report_is_valid_property_reflects_empty_violations():
    assert DatasetValidationReportV3(violations=()).is_valid is True
    assert DatasetValidationReportV3(violations=("something wrong",)).is_valid is False


def test_empty_sample_list_reports_a_violation_instead_of_crashing():
    report = validate_training_samples_v3(())

    assert not report.is_valid


def test_detects_single_label_dataset():
    samples = FireDetectionTrainingDataGeneratorV3(seed=42).generate(num_samples=SAMPLE_COUNT)
    only_positive = tuple(sample for sample in samples if sample.label == 1)

    report = validate_training_samples_v3(only_positive)

    assert not report.is_valid
    assert any("both labels" in violation for violation in report.violations)
