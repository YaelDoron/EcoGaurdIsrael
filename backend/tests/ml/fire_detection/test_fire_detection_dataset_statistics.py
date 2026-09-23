"""Tests for fire_detection_dataset_statistics using a small known fixture."""
from __future__ import annotations

import pytest

from src.ml.fire_detection.fire_detection_dataset import FireDetectionDatasetRow
from src.ml.fire_detection.fire_detection_dataset_statistics import (
    compute_feature_correlation_matrix,
    compute_feature_statistics_by_label,
    compute_scenario_family_statistics,
    satellite_count_identity_holds,
    scenario_family_perfectly_predicts_label,
)
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES

# features order: satellite_count, news_count, satellite_low_count,
# satellite_nominal_count, satellite_high_count, time_span_minutes, max_pairwise_distance_km
# Every feature varies across rows (no zero-variance column) so correlation/diagonal
# checks are meaningful, and satellite_count == low+nominal+high holds for every row.
FIXTURE_ROWS = (
    FireDetectionDatasetRow(sample_id=1, scenario_family="fam_a", features=(1, 0, 0, 1, 0, 5.0, 1.0), label=0),
    FireDetectionDatasetRow(sample_id=2, scenario_family="fam_a", features=(2, 0, 0, 2, 0, 15.0, 3.0), label=0),
    FireDetectionDatasetRow(sample_id=3, scenario_family="fam_b", features=(3, 1, 0, 2, 1, 10.0, 2.0), label=1),
    FireDetectionDatasetRow(sample_id=4, scenario_family="fam_b", features=(4, 1, 1, 3, 0, 20.0, 4.0), label=1),
)


def _stat_for(stats, feature_name, label):
    return next(stat for stat in stats if stat.feature_name == feature_name and stat.label == label)


def test_feature_statistics_are_calculated_correctly_on_a_known_fixture():
    stats = compute_feature_statistics_by_label(FIXTURE_ROWS)

    satellite_count_label_0 = _stat_for(stats, "satellite_count", 0)
    assert satellite_count_label_0.count == 2
    assert satellite_count_label_0.mean == pytest.approx(1.5)
    assert satellite_count_label_0.median == pytest.approx(1.5)
    assert satellite_count_label_0.std == pytest.approx(0.5)
    assert satellite_count_label_0.minimum == pytest.approx(1.0)
    assert satellite_count_label_0.maximum == pytest.approx(2.0)

    satellite_count_label_1 = _stat_for(stats, "satellite_count", 1)
    assert satellite_count_label_1.count == 2
    assert satellite_count_label_1.mean == pytest.approx(3.5)
    assert satellite_count_label_1.std == pytest.approx(0.5)
    assert satellite_count_label_1.minimum == pytest.approx(3.0)
    assert satellite_count_label_1.maximum == pytest.approx(4.0)

    time_span_label_1 = _stat_for(stats, "time_span_minutes", 1)
    assert time_span_label_1.mean == pytest.approx(15.0)
    assert time_span_label_1.median == pytest.approx(15.0)


def test_statistics_are_reported_for_every_declared_feature_and_both_labels():
    stats = compute_feature_statistics_by_label(FIXTURE_ROWS)

    feature_label_pairs = {(stat.feature_name, stat.label) for stat in stats}
    expected_pairs = {(name, label) for name in FIRE_DETECTION_FEATURE_NAMES for label in (0, 1)}
    assert feature_label_pairs == expected_pairs


def test_scenario_family_statistics_are_calculated_correctly():
    family_stats = compute_scenario_family_statistics(FIXTURE_ROWS)

    fam_a = next(stat for stat in family_stats if stat.scenario_family == "fam_a")
    assert fam_a.label == 0
    assert fam_a.count == 2
    assert fam_a.mean_features["satellite_count"] == pytest.approx(1.5)
    assert fam_a.mean_features["time_span_minutes"] == pytest.approx(10.0)

    fam_b = next(stat for stat in family_stats if stat.scenario_family == "fam_b")
    assert fam_b.label == 1
    assert fam_b.count == 2
    assert fam_b.mean_features["max_pairwise_distance_km"] == pytest.approx(3.0)


def test_scenario_family_perfectly_predicts_label_on_clean_fixture():
    assert scenario_family_perfectly_predicts_label(FIXTURE_ROWS) is True


def test_scenario_family_perfectly_predicts_label_is_false_when_a_family_mixes_labels():
    mixed_rows = FIXTURE_ROWS + (
        FireDetectionDatasetRow(sample_id=5, scenario_family="fam_a", features=(1, 0, 0, 1, 0, 5.0, 1.0), label=1),
    )

    assert scenario_family_perfectly_predicts_label(mixed_rows) is False


def test_satellite_count_identity_holds_on_valid_fixture():
    assert satellite_count_identity_holds(FIXTURE_ROWS) is True


def test_satellite_count_identity_detects_violation():
    broken_rows = (
        FireDetectionDatasetRow(sample_id=1, scenario_family="fam_a", features=(5, 0, 0, 1, 0, 5.0, 1.0), label=0),
    )

    assert satellite_count_identity_holds(broken_rows) is False


def test_correlation_matrix_uses_only_declared_numerical_feature_columns():
    matrix = compute_feature_correlation_matrix(FIXTURE_ROWS)

    assert set(matrix.keys()) == set(FIRE_DETECTION_FEATURE_NAMES)
    for row_key, row in matrix.items():
        assert set(row.keys()) == set(FIRE_DETECTION_FEATURE_NAMES)


def test_correlation_matrix_diagonal_is_one():
    matrix = compute_feature_correlation_matrix(FIXTURE_ROWS)

    for feature_name in FIRE_DETECTION_FEATURE_NAMES:
        assert matrix[feature_name][feature_name] == pytest.approx(1.0)


def test_correlation_matrix_is_symmetric():
    matrix = compute_feature_correlation_matrix(FIXTURE_ROWS)

    for first_name in FIRE_DETECTION_FEATURE_NAMES:
        for second_name in FIRE_DETECTION_FEATURE_NAMES:
            assert matrix[first_name][second_name] == pytest.approx(matrix[second_name][first_name])


def test_satellite_count_perfectly_correlates_with_itself_derived_sum():
    # In this fixture satellite_count grows in lockstep with satellite_nominal_count
    # (both double from row 1->2 and row 3->4), so their correlation should be strongly positive.
    matrix = compute_feature_correlation_matrix(FIXTURE_ROWS)

    assert matrix["satellite_count"]["satellite_nominal_count"] > 0.9
