"""Lightweight tests for the Fire Detection Logistic Regression training pipeline.

Uses a small deterministically generated dataset - not the full 2000-row
training_v1.csv - to keep the unit-test suite fast.
"""
from __future__ import annotations

import joblib
import numpy as np

from scripts.train_fire_detection_model import build_pipeline
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_training_data_generator import FireDetectionTrainingDataGenerator

SMALL_SAMPLE_COUNT = 200


def _small_dataset():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SMALL_SAMPLE_COUNT)
    extractor = FireDetectionFeatureExtractor()
    features = [extractor.extract(sample.evidence).as_tuple() for sample in samples]
    labels = [sample.label for sample in samples]
    return features, labels


def test_model_fitting_succeeds():
    features, labels = _small_dataset()

    pipeline = build_pipeline()
    pipeline.fit(features, labels)

    assert hasattr(pipeline, "predict")


def test_model_exposes_predict_proba_within_valid_range():
    features, labels = _small_dataset()
    pipeline = build_pipeline()
    pipeline.fit(features, labels)

    probabilities = pipeline.predict_proba(features)

    assert probabilities.shape == (len(features), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_both_labels_are_represented_in_training_data():
    _, labels = _small_dataset()

    assert set(labels) == {0, 1}


def test_feature_count_matches_declared_feature_names():
    features, _ = _small_dataset()

    assert all(len(row) == len(FIRE_DETECTION_FEATURE_NAMES) for row in features)


def test_saved_pipeline_can_be_reloaded_with_equivalent_predictions(tmp_path):
    features, labels = _small_dataset()
    pipeline = build_pipeline()
    pipeline.fit(features, labels)

    model_path = tmp_path / "fire_detection_logistic_v1.joblib"
    joblib.dump(pipeline, model_path)

    reloaded_pipeline = joblib.load(model_path)

    original_probabilities = pipeline.predict_proba(features)
    reloaded_probabilities = reloaded_pipeline.predict_proba(features)

    assert np.allclose(original_probabilities, reloaded_probabilities)
