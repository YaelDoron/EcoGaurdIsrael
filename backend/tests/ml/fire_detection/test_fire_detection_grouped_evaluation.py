"""Tests for grouped (unseen-scenario-family) cross-validation helpers."""
from __future__ import annotations

import pytest

from scripts.train_fire_detection_model import build_pipeline as build_logistic_pipeline
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_grouped_evaluation import (
    CV_SCORING,
    build_grouped_splitter,
    choose_grouped_fold_count,
    run_grouped_cross_validation,
    run_leave_one_group_out_cross_validation,
    verify_grouped_folds,
)
from src.ml.fire_detection.fire_detection_model_comparison import build_random_forest
from src.ml.fire_detection.fire_detection_training_data_generator import (
    DATASET_VERSION_V2,
    FireDetectionTrainingDataGenerator,
)

SAMPLE_COUNT = 900


def _dataset():
    samples = FireDetectionTrainingDataGenerator(seed=42).generate(num_samples=SAMPLE_COUNT, dataset_version=DATASET_VERSION_V2)
    extractor = FireDetectionFeatureExtractor()
    features = [list(extractor.extract(sample.evidence).as_tuple()) for sample in samples]
    labels = [sample.label for sample in samples]
    groups = [sample.scenario_family for sample in samples]
    return features, labels, groups


def test_choose_grouped_fold_count_returns_a_safe_value_in_range():
    _, labels, groups = _dataset()

    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)

    assert 3 <= n_splits <= 6


def test_choose_grouped_fold_count_raises_when_nothing_is_safe():
    # Only 2 distinct groups total - cannot safely support any candidate >= 3.
    labels = [0, 0, 1, 1] * 10
    groups = ["fam_a"] * 20 + ["fam_b"] * 20

    with pytest.raises(ValueError):
        choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)


def test_no_scenario_family_appears_in_both_train_and_validation():
    _, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)
    splitter = build_grouped_splitter(n_splits)

    for train_index, val_index in splitter.split([[0.0]] * len(labels), labels, groups):
        train_families = {groups[i] for i in train_index}
        val_families = {groups[i] for i in val_index}
        assert train_families.isdisjoint(val_families)


def test_every_sample_appears_in_validation_exactly_once_across_folds():
    _, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)
    splitter = build_grouped_splitter(n_splits)

    seen_indices: list[int] = []
    for _, val_index in splitter.split([[0.0]] * len(labels), labels, groups):
        seen_indices.extend(val_index)

    assert sorted(seen_indices) == list(range(len(labels)))


def test_verify_grouped_folds_passes_for_a_valid_configuration():
    _, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)

    verify_grouped_folds(labels, groups, n_splits)  # must not raise


def test_verify_grouped_folds_fails_clearly_for_an_unsafe_configuration():
    labels = [0, 0, 1, 1] * 10
    groups = ["fam_a"] * 20 + ["fam_b"] * 20

    with pytest.raises((AssertionError, ValueError)):
        verify_grouped_folds(labels, groups, n_splits=5)


def test_grouped_split_is_deterministic():
    _, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=6, min_candidate=3)

    first_folds = list(build_grouped_splitter(n_splits).split([[0.0]] * len(labels), labels, groups))
    second_folds = list(build_grouped_splitter(n_splits).split([[0.0]] * len(labels), labels, groups))

    for (train_a, val_a), (train_b, val_b) in zip(first_folds, second_folds):
        assert list(train_a) == list(train_b)
        assert list(val_a) == list(val_b)


def test_run_grouped_cross_validation_result_schema():
    features, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=5, min_candidate=3)

    result = run_grouped_cross_validation(build_logistic_pipeline, features, labels, groups, n_splits)

    assert result.n_splits == n_splits
    assert len(result.fold_results) == n_splits
    assert set(result.summary.keys()) == set(CV_SCORING)
    for metric_result in result.summary.values():
        assert set(metric_result.keys()) == {"mean", "std"}


def test_run_grouped_cross_validation_per_family_covers_every_family_once():
    features, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=5, min_candidate=3)

    result = run_grouped_cross_validation(build_random_forest, features, labels, groups, n_splits)

    families_in_report = {item.scenario_family for item in result.per_family}
    assert families_in_report == set(groups)
    total_rows_reported = sum(item.sample_count for item in result.per_family)
    assert total_rows_reported == len(labels)


def test_run_grouped_cross_validation_fold_families_have_no_overlap():
    features, labels, groups = _dataset()
    n_splits = choose_grouped_fold_count(labels, groups, max_candidate=5, min_candidate=3)

    result = run_grouped_cross_validation(build_logistic_pipeline, features, labels, groups, n_splits)

    for fold in result.fold_results:
        assert set(fold.train_families).isdisjoint(set(fold.validation_families))


# --- leave-one-group-out (archetype holdout) ---


def _archetype_dataset():
    from src.ml.fire_detection.fire_detection_training_data_generator_v3 import FireDetectionTrainingDataGeneratorV3

    samples = FireDetectionTrainingDataGeneratorV3(seed=42).generate(num_samples=900)
    extractor = FireDetectionFeatureExtractorV3()
    features = [list(extractor.extract(sample.evidence).as_tuple()) for sample in samples]
    labels = [sample.label for sample in samples]
    archetypes = [sample.scenario_archetype.value for sample in samples]
    return features, labels, archetypes


def test_leave_one_group_out_produces_one_fold_per_distinct_group():
    features, labels, archetypes = _archetype_dataset()

    result = run_leave_one_group_out_cross_validation(build_logistic_pipeline, features, labels, archetypes)

    assert result.n_splits == len(set(archetypes))
    assert len(result.fold_results) == len(set(archetypes))


def test_leave_one_group_out_each_validation_fold_is_exactly_one_group():
    features, labels, archetypes = _archetype_dataset()

    result = run_leave_one_group_out_cross_validation(build_random_forest, features, labels, archetypes)

    for fold in result.fold_results:
        assert len(fold.validation_families) == 1
        assert set(fold.train_families).isdisjoint(set(fold.validation_families))


def test_leave_one_group_out_covers_every_row_exactly_once():
    features, labels, archetypes = _archetype_dataset()

    result = run_leave_one_group_out_cross_validation(build_logistic_pipeline, features, labels, archetypes)

    total_rows_reported = sum(item.sample_count for item in result.per_family)
    assert total_rows_reported == len(labels)


def test_leave_one_group_out_per_group_label_is_none_when_mixed():
    """Unlike scenario_family groups, an archetype legitimately mixes both labels -
    so its FamilyValidationSummary.label must be None, not a misleading single value."""
    features, labels, archetypes = _archetype_dataset()

    result = run_leave_one_group_out_cross_validation(build_logistic_pipeline, features, labels, archetypes)

    for item in result.per_family:
        true_labels_for_group = {label for label, group in zip(labels, archetypes) if group == item.scenario_family}
        if len(true_labels_for_group) > 1:
            assert item.label is None
            assert 0.0 < item.actual_positive_fraction < 1.0
