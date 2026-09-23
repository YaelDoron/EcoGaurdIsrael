"""Tests for the feature-ablation helpers (satellite_count redundancy experiment)."""
from __future__ import annotations

import pytest

from scripts.train_fire_detection_model import build_pipeline as build_logistic_pipeline
from src.ml.fire_detection.fire_detection_feature_ablation import (
    ABLATION_FEATURE_SET_FULL,
    ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT,
    select_feature_columns,
)
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_model_comparison import build_random_forest
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V2,
    FireDetectionTrainingDataGenerator,
)

SAMPLE_COUNT = 200


def test_full_feature_set_has_seven_features():
    assert len(ABLATION_FEATURE_SET_FULL) == 7
    assert ABLATION_FEATURE_SET_FULL == FIRE_DETECTION_FEATURE_NAMES


def test_ablated_feature_set_has_six_features():
    assert len(ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT) == 6


def test_satellite_count_is_excluded_only_from_the_ablated_set():
    assert "satellite_count" in ABLATION_FEATURE_SET_FULL
    assert "satellite_count" not in ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT


def test_ablated_set_feature_ordering_matches_the_full_set_relative_order():
    remaining_full_order = tuple(name for name in ABLATION_FEATURE_SET_FULL if name != "satellite_count")

    assert ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT == remaining_full_order


def test_select_feature_columns_extracts_the_requested_columns_in_order():
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    feature_names = ("a", "b", "c")

    selected = select_feature_columns(rows, feature_names, keep=("c", "a"))

    assert selected == [[3.0, 1.0], [6.0, 4.0]]


def test_select_feature_columns_rejects_unknown_column_names():
    rows = [[1.0, 2.0]]
    feature_names = ("a", "b")

    with pytest.raises(ValueError):
        select_feature_columns(rows, feature_names, keep=("a", "does_not_exist"))


def _dataset():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    extractor = FireDetectionFeatureExtractor()
    features = [list(extractor.extract(sample.evidence).as_tuple()) for sample in samples]
    labels = [sample.label for sample in samples]
    return features, labels


def test_both_models_can_train_under_the_full_feature_set():
    features, labels = _dataset()

    build_logistic_pipeline().fit(features, labels)
    build_random_forest().fit(features, labels)


def test_both_models_can_train_under_the_ablated_feature_set():
    features, labels = _dataset()
    ablated = select_feature_columns(features, FIRE_DETECTION_FEATURE_NAMES, ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT)

    logistic_pipeline = build_logistic_pipeline()
    logistic_pipeline.fit(ablated, labels)
    random_forest = build_random_forest()
    random_forest.fit(ablated, labels)

    assert len(random_forest.feature_importances_) == len(ABLATION_FEATURE_SET_WITHOUT_SATELLITE_COUNT)
