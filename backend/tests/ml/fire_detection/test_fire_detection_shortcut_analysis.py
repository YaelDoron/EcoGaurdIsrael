"""Tests for univariate shortcut and feature-vector collision analysis."""
from __future__ import annotations

import pytest

from src.ml.fire_detection.fire_detection_shortcut_analysis import (
    SUSPICIOUS_AUC_THRESHOLD,
    compute_feature_vector_collisions,
    compute_univariate_shortcut_analysis,
)

FEATURE_NAMES = ("perfect_separator", "no_signal")


def test_perfectly_separating_feature_is_flagged_suspicious():
    features = [[0.0, 5.0], [0.0, 5.0], [1.0, 5.0], [1.0, 5.0]]
    labels = [0, 0, 1, 1]

    results = compute_univariate_shortcut_analysis(features, labels, FEATURE_NAMES)

    perfect = next(item for item in results if item.feature_name == "perfect_separator")
    assert perfect.auc == pytest.approx(1.0)
    assert perfect.is_suspicious_shortcut is True
    assert perfect.auc >= SUSPICIOUS_AUC_THRESHOLD


def test_uninformative_feature_is_not_flagged():
    features = [[0.0, 5.0], [0.0, 5.0], [1.0, 5.0], [1.0, 5.0]]
    labels = [0, 0, 1, 1]

    results = compute_univariate_shortcut_analysis(features, labels, FEATURE_NAMES)

    constant = next(item for item in results if item.feature_name == "no_signal")
    assert constant.is_suspicious_shortcut is False


def test_inversely_correlated_feature_still_flagged_via_best_orientation():
    # label=1 has the LOWER value - AUC computed directly would be ~0.0, but the
    # feature is still perfectly (inversely) discriminative, so best-orientation AUC = 1.0.
    features = [[10.0], [10.0], [1.0], [1.0]]
    labels = [0, 0, 1, 1]

    results = compute_univariate_shortcut_analysis(features, labels, ("inverse",))

    assert results[0].auc == pytest.approx(1.0)
    assert results[0].is_suspicious_shortcut is True


def test_mean_median_range_are_computed_per_label():
    features = [[1.0], [3.0], [10.0], [20.0]]
    labels = [0, 0, 1, 1]

    results = compute_univariate_shortcut_analysis(features, labels, ("feature",))

    stat = results[0]
    assert stat.mean_label_0 == pytest.approx(2.0)
    assert stat.mean_label_1 == pytest.approx(15.0)
    assert stat.min_label_0 == pytest.approx(1.0)
    assert stat.max_label_0 == pytest.approx(3.0)
    assert stat.min_label_1 == pytest.approx(10.0)
    assert stat.max_label_1 == pytest.approx(20.0)


# --- collision analysis ---


def test_no_collisions_when_vectors_are_all_distinct_and_pure():
    features = [[1.0], [2.0], [3.0], [4.0]]
    labels = [0, 0, 1, 1]

    result = compute_feature_vector_collisions(features, labels)

    assert result.colliding_vectors == 0
    assert result.colliding_rows == 0
    assert result.collision_rate == 0.0
    assert result.distinct_feature_vectors == 4


def test_exact_collision_between_opposite_labels_is_detected():
    features = [[1.0], [1.0], [2.0]]
    labels = [0, 1, 1]  # the [1.0] vector appears under BOTH labels

    result = compute_feature_vector_collisions(features, labels)

    assert result.colliding_vectors == 1
    assert result.colliding_rows == 2
    assert result.collision_rate == pytest.approx(2 / 3)


def test_same_label_repeats_are_not_counted_as_collisions():
    features = [[1.0], [1.0], [1.0]]
    labels = [0, 0, 0]  # repeated, but never a different label

    result = compute_feature_vector_collisions(features, labels)

    assert result.colliding_vectors == 0
    assert result.colliding_rows == 0


def test_collision_rate_is_zero_for_empty_dataset():
    result = compute_feature_vector_collisions([], [])

    assert result.total_rows == 0
    assert result.collision_rate == 0.0
