"""Tests for the shared Logistic Regression vs Random Forest comparison logic.

Uses a small deterministically generated dataset - not the full 2000-row
training_v1.csv - to keep this suite fast.
"""
from __future__ import annotations

import joblib
import numpy as np

from scripts.train_fire_detection_model import build_pipeline as build_logistic_pipeline
from src.ml.fire_detection.fire_detection_dataset import FireDetectionDatasetRow
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FIRE_DETECTION_FEATURE_NAMES
from src.ml.fire_detection.fire_detection_model_comparison import (
    CV_SCORING,
    CV_SPLITS,
    build_random_forest,
    compute_feature_importances,
    compute_permutation_importance,
    evaluate_cross_validation,
    evaluate_held_out,
    evaluate_predictions,
    evaluate_rule_based_calculator,
    make_cv_splitter,
    make_held_out_split,
    regenerated_samples_match_dataset,
)
from src.ml.fire_detection.fire_detection_training_data_generator import FireDetectionTrainingDataGenerator

SMALL_SAMPLE_COUNT = 300


def _small_dataset():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SMALL_SAMPLE_COUNT)
    extractor = FireDetectionFeatureExtractor()
    features = [list(extractor.extract(sample.evidence).as_tuple()) for sample in samples]
    labels = [sample.label for sample in samples]
    return samples, features, labels


# --- Shared splitting ---


def test_held_out_split_covers_every_row_exactly_once():
    _, features, labels = _small_dataset()

    split = make_held_out_split(features, labels)

    all_indices = sorted(split.train_indices + split.test_indices)
    assert all_indices == list(range(len(labels)))


def test_held_out_split_x_test_matches_original_rows_at_test_indices():
    _, features, labels = _small_dataset()

    split = make_held_out_split(features, labels)

    for position, original_index in enumerate(split.test_indices):
        assert split.x_test[position] == features[original_index]
        assert split.y_test[position] == labels[original_index]


def test_both_classifiers_receive_the_exact_same_split_object():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)

    logistic_pipeline = build_logistic_pipeline()
    logistic_pipeline.fit(split.x_train, split.y_train)

    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    # Both fits consumed identical x_train/y_train - the same HeldOutSplit instance,
    # not two independently generated splits.
    assert logistic_pipeline.predict(split.x_test).shape == random_forest.predict(split.x_test).shape


# --- Cross-validation ---


def test_cv_splitter_has_the_configured_number_of_folds():
    _, features, labels = _small_dataset()
    cv = make_cv_splitter()

    folds = list(cv.split(features, labels))

    assert len(folds) == CV_SPLITS


def test_both_labels_are_represented_in_every_cv_fold():
    _, features, labels = _small_dataset()
    cv = make_cv_splitter()

    for _, test_index in cv.split(features, labels):
        fold_labels = {labels[index] for index in test_index}
        assert fold_labels == {0, 1}


def test_cross_validation_result_schema_includes_all_expected_metrics():
    _, features, labels = _small_dataset()
    cv = make_cv_splitter()

    result = evaluate_cross_validation(build_logistic_pipeline(), features, labels, cv)

    assert set(result.keys()) == set(CV_SCORING)
    for metric_result in result.values():
        assert set(metric_result.keys()) == {"mean", "std"}
        assert isinstance(metric_result["mean"], float)
        assert isinstance(metric_result["std"], float)


# --- Random Forest ---


def test_random_forest_training_succeeds():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)

    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    assert hasattr(random_forest, "predict")


def test_random_forest_predict_proba_is_valid():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    probabilities = random_forest.predict_proba(split.x_test)

    assert probabilities.shape == (len(split.x_test), 2)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_random_forest_model_artifact_can_be_saved_and_reloaded(tmp_path):
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    model_path = tmp_path / "fire_detection_random_forest_v1.joblib"
    joblib.dump(random_forest, model_path)
    reloaded = joblib.load(model_path)

    original_probabilities = random_forest.predict_proba(split.x_test)
    reloaded_probabilities = reloaded.predict_proba(split.x_test)
    assert np.allclose(original_probabilities, reloaded_probabilities)


def test_random_forest_feature_importances_match_feature_count():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    importances = compute_feature_importances(random_forest)

    assert len(importances) == len(FIRE_DETECTION_FEATURE_NAMES)
    assert {name for name, _ in importances} == set(FIRE_DETECTION_FEATURE_NAMES)


def test_random_forest_feature_importances_are_sorted_descending():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    importances = compute_feature_importances(random_forest)
    values = [value for _, value in importances]

    assert values == sorted(values, reverse=True)


def test_permutation_importance_matches_feature_count():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    importances = compute_permutation_importance(random_forest, split.x_test, split.y_test, n_repeats=3)

    assert len(importances) == len(FIRE_DETECTION_FEATURE_NAMES)
    assert {name for name, _ in importances} == set(FIRE_DETECTION_FEATURE_NAMES)


def test_evaluate_predictions_returns_accuracy_and_f1_only():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    train_metrics = evaluate_predictions(random_forest, split.x_train, split.y_train)

    assert set(train_metrics.keys()) == {"accuracy", "f1"}


# --- Comparison (same labels, same metric names) ---


def test_both_models_are_evaluated_against_the_same_held_out_labels():
    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)

    logistic_pipeline = build_logistic_pipeline()
    logistic_pipeline.fit(split.x_train, split.y_train)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    logistic_metrics = evaluate_held_out(logistic_pipeline, split.x_test, split.y_test)
    rf_metrics = evaluate_held_out(random_forest, split.x_test, split.y_test)

    assert set(logistic_metrics.keys()) == set(rf_metrics.keys())


def test_comparison_output_contains_the_same_metric_names_for_both_models():
    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    }

    _, features, labels = _small_dataset()
    split = make_held_out_split(features, labels)
    logistic_pipeline = build_logistic_pipeline()
    logistic_pipeline.fit(split.x_train, split.y_train)
    random_forest = build_random_forest()
    random_forest.fit(split.x_train, split.y_train)

    assert set(evaluate_held_out(logistic_pipeline, split.x_test, split.y_test).keys()) == expected_keys
    assert set(evaluate_held_out(random_forest, split.x_test, split.y_test).keys()) == expected_keys


# --- Rule-based calculator comparison + dataset integrity check ---


def test_regenerated_samples_match_dataset_true_for_matching_csv():
    samples, features, labels = _small_dataset()
    extractor = FireDetectionFeatureExtractor()
    csv_rows = tuple(
        FireDetectionDatasetRow(
            sample_id=sample.sample_id,
            scenario_family=sample.scenario_family,
            features=extractor.extract(sample.evidence).as_tuple(),
            label=sample.label,
        )
        for sample in samples
    )

    assert regenerated_samples_match_dataset(csv_rows, samples) is True


def test_regenerated_samples_match_dataset_false_on_mismatch():
    samples, _, _ = _small_dataset()
    extractor = FireDetectionFeatureExtractor()
    mismatched_rows = tuple(
        FireDetectionDatasetRow(
            sample_id=sample.sample_id,
            scenario_family=sample.scenario_family,
            features=extractor.extract(sample.evidence).as_tuple(),
            label=1 - sample.label,  # deliberately flip every label
        )
        for sample in samples
    )

    assert regenerated_samples_match_dataset(mismatched_rows, samples) is False


def test_evaluate_rule_based_calculator_returns_expected_keys_and_consistent_confusion_counts():
    samples, _, _ = _small_dataset()
    indices = tuple(range(len(samples)))

    metrics = evaluate_rule_based_calculator(samples, indices)

    assert {"accuracy", "precision", "recall", "f1", "true_positive", "true_negative", "false_positive", "false_negative"} <= set(
        metrics.keys()
    )
    total = metrics["true_positive"] + metrics["true_negative"] + metrics["false_positive"] + metrics["false_negative"]
    assert total == len(indices)
