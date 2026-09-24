"""Tests for the V4 model definitions: schema validation before training, preprocessing, determinism."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DETECTION_FEATURE_NAMES_V4,
    LABEL_COLUMN,
    NULLABLE_FEATURE_NAMES_V4,
    TRAINING_DATA_METADATA_COLUMNS_V4,
)
from src.ml.fire_detection.fire_detection_model_v4 import (
    DEFAULT_TRAINING_CSV_V4,
    FEATURE_GROUPS_V4,
    MODEL_SPECS_V4,
    SENSITIVITY_VARIANTS_V4,
    build_pipeline_v4,
    build_preprocessor_v4,
    dataset_sha256,
    describe_preprocessing_v4,
    feature_names_without,
    load_training_matrix_v4,
    transformed_feature_names,
)

INDEX = {name: index for index, name in enumerate(FIRE_DETECTION_FEATURE_NAMES_V4)}


@pytest.fixture(scope="session")
def matrix():
    return load_training_matrix_v4(DEFAULT_TRAINING_CSV_V4)


def rewrite_header(tmp_path: Path, transform) -> Path:
    with DEFAULT_TRAINING_CSV_V4.open(encoding="utf-8", newline="") as source:
        lines = list(csv.reader(source))
    lines[0] = transform(lines[0])
    target = tmp_path / "broken.csv"
    with target.open("w", encoding="utf-8", newline="") as sink:
        csv.writer(sink).writerows(lines[:50])
    return target


# --- schema validation happens before training ---


def test_matrix_columns_are_exactly_the_canonical_v4_features_in_order(matrix):
    assert matrix.feature_names == FIRE_DETECTION_FEATURE_NAMES_V4
    assert matrix.X.shape == (9000, 19)
    assert matrix.y.shape == (9000,)


def test_a_reordered_feature_column_is_refused_before_training(tmp_path):
    def swap(header):
        header = list(header)
        a, b = header.index("satellite_low_count"), header.index("satellite_high_count")
        header[a], header[b] = header[b], header[a]
        return header

    with pytest.raises(ValueError):
        load_training_matrix_v4(rewrite_header(tmp_path, swap))


def test_a_dropped_feature_column_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_training_matrix_v4(rewrite_header(tmp_path, lambda header: [c for c in header if c != "news_weak_count"]))


def test_a_leakage_column_smuggled_in_as_a_feature_is_refused(tmp_path):
    def rename(header):
        return ["scenario_family_feature" if c == "time_span_minutes" else c for c in header]

    with pytest.raises(ValueError):
        load_training_matrix_v4(rewrite_header(tmp_path, rename))


def test_target_and_metadata_never_enter_the_feature_matrix(matrix):
    assert LABEL_COLUMN not in matrix.feature_names
    assert set(matrix.feature_names).isdisjoint(TRAINING_DATA_METADATA_COLUMNS_V4)
    assert not any(np.array_equal(matrix.X[:, i], matrix.y) for i in range(matrix.X.shape[1]))
    # metadata lives beside, not inside, the matrix
    assert matrix.families.shape == matrix.archetypes.shape == matrix.y.shape


def test_nan_marks_missing_fire_danger_and_only_there(matrix):
    nan_columns = {name for name, has_nan in zip(matrix.feature_names, np.isnan(matrix.X).any(axis=0)) if has_nan}
    unavailable = matrix.X[:, INDEX["fire_danger_available"]] == 0

    assert nan_columns == set(NULLABLE_FEATURE_NAMES_V4)
    assert np.array_equal(np.isnan(matrix.X[:, INDEX["fire_danger_score"]]), unavailable)
    assert np.array_equal(np.isnan(matrix.X[:, INDEX["fire_danger_age_minutes"]]), unavailable)


def test_dataset_hash_is_the_frozen_task2_file():
    assert dataset_sha256(DEFAULT_TRAINING_CSV_V4) == "6e14edc5ff536f3e55e680dfe5e89862b9e712ea1e5a6639be2a1e10767d812f"


# --- feature subsets for ablations ---


def test_feature_groups_are_derived_from_the_canonical_names():
    assert FEATURE_GROUPS_V4["fire_danger"] == FIRE_DETECTION_FEATURE_NAMES_V4[16:]
    assert len(FEATURE_GROUPS_V4["news"]) == 5
    assert len(FEATURE_GROUPS_V4["satellite_frp_brightness"]) == 6
    assert FEATURE_GROUPS_V4["geometry_time"] == ("time_span_minutes", "max_pairwise_distance_km")
    for names in FEATURE_GROUPS_V4.values():
        assert set(names) <= set(FIRE_DETECTION_FEATURE_NAMES_V4)


def test_feature_names_without_keeps_canonical_order_and_rejects_unknown_names():
    names = feature_names_without(FEATURE_GROUPS_V4["fire_danger"])

    assert names == FIRE_DETECTION_FEATURE_NAMES_V4[:16]
    with pytest.raises(ValueError):
        feature_names_without(["not_a_feature"])


# --- preprocessing ---


def test_only_the_nullable_fire_danger_numerics_are_imputed(matrix):
    pipeline = build_pipeline_v4("random_forest").fit(matrix.X, matrix.y)
    preprocess = pipeline.named_steps["preprocess"]
    transformed = preprocess.transform(matrix.X)
    names = transformed_feature_names(pipeline, matrix.feature_names)

    assert not np.isnan(transformed).any()
    assert names[:2] == ("fire_danger_score", "fire_danger_age_minutes")
    assert set(names) == set(FIRE_DETECTION_FEATURE_NAMES_V4)
    # everything except the two nullable columns passes through untouched, including fire_danger_available
    for index, name in enumerate(names[2:], start=2):
        assert np.array_equal(transformed[:, index], matrix.X[:, INDEX[name]]), name


def test_imputation_uses_the_training_median_of_observed_values_only():
    X = np.zeros((6, 19))
    X[:, INDEX["satellite_nominal_count"]] = 1
    X[:4, INDEX["fire_danger_available"]] = 1
    X[:4, INDEX["fire_danger_score"]] = [10.0, 20.0, 30.0, 100.0]  # observed median = 25
    X[:4, INDEX["fire_danger_age_minutes"]] = [1.0, 2.0, 3.0, 40.0]  # observed median = 2.5
    X[4:, INDEX["fire_danger_score"]] = np.nan
    X[4:, INDEX["fire_danger_age_minutes"]] = np.nan
    preprocessor = build_preprocessor_v4().fit(X)

    transformed = preprocessor.transform(X)

    assert list(transformed[4:, 0]) == [25.0, 25.0]
    assert list(transformed[4:, 1]) == [2.5, 2.5]
    assert list(transformed[:4, 0]) == [10.0, 20.0, 30.0, 100.0]


def test_every_model_accepts_unavailable_fire_danger(matrix):
    unavailable = matrix.X[matrix.X[:, INDEX["fire_danger_available"]] == 0][:50]
    assert np.isnan(unavailable[:, INDEX["fire_danger_score"]]).all()

    for key in MODEL_SPECS_V4:
        probabilities = build_pipeline_v4(key).fit(matrix.X[:3000], matrix.y[:3000]).predict_proba(unavailable)[:, 1]
        assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all(), key


def test_a_real_zero_ffwi_is_distinct_from_missing_after_preprocessing(matrix):
    base = matrix.X[0].copy()
    real_zero = base.copy()
    real_zero[[INDEX["fire_danger_available"], INDEX["fire_danger_score"], INDEX["fire_danger_age_minutes"]]] = [1, 0.0, 0.0]
    missing = base.copy()
    missing[[INDEX["fire_danger_available"], INDEX["fire_danger_score"], INDEX["fire_danger_age_minutes"]]] = [0, np.nan, np.nan]
    pipeline = build_pipeline_v4("logistic_regression").fit(matrix.X, matrix.y)
    names = transformed_feature_names(pipeline, matrix.feature_names)
    preprocess = pipeline.named_steps["preprocess"]

    zero_out = dict(zip(names, preprocess.transform(np.array([real_zero]))[0]))
    missing_out = dict(zip(names, preprocess.transform(np.array([missing]))[0]))

    assert zero_out["fire_danger_score"] == 0.0 and zero_out["fire_danger_available"] == 1
    assert missing_out["fire_danger_available"] == 0
    assert missing_out["fire_danger_score"] != 0.0  # median-imputed, never zero-filled
    assert missing_out["fire_danger_age_minutes"] != 0.0
    assert not np.array_equal(preprocess.transform(np.array([real_zero])), preprocess.transform(np.array([missing])))
    assert pipeline.predict_proba(np.array([real_zero]))[0, 1] != pipeline.predict_proba(np.array([missing]))[0, 1]


def test_scaling_lives_inside_the_pipeline_for_logistic_regression_only():
    assert [name for name, _ in build_pipeline_v4("logistic_regression").steps] == ["preprocess", "scale", "classifier"]
    assert [name for name, _ in build_pipeline_v4("random_forest").steps] == ["preprocess", "classifier"]
    assert [name for name, _ in build_pipeline_v4("hist_gradient_boosting").steps] == ["preprocess", "classifier"]


def test_the_no_fire_danger_pipeline_has_no_imputer_and_trains(matrix):
    names = feature_names_without(FEATURE_GROUPS_V4["fire_danger"])
    pipeline = build_pipeline_v4("logistic_regression", names).fit(matrix.columns(names)[:2000], matrix.y[:2000])

    assert pipeline.predict_proba(matrix.columns(names)[:5]).shape == (5, 2)
    assert transformed_feature_names(pipeline, names) == names


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_pipeline_v4("deep_neural_network")


def test_preprocessing_description_is_complete_and_json_serialisable():
    import json

    description = describe_preprocessing_v4("logistic_regression")

    assert description["feature_names"] == list(FIRE_DETECTION_FEATURE_NAMES_V4)
    steps = {step["step"]: step for step in description["steps"]}
    assert steps["impute_nullable_fire_danger"]["strategy"] == "median"
    assert steps["impute_nullable_fire_danger"]["applied_to"] == list(NULLABLE_FEATURE_NAMES_V4)
    assert "fire_danger_available" in steps["passthrough"]["applied_to"]
    assert "scale" in steps and steps["classifier"]["estimator"] == "LogisticRegression"
    json.dumps(description)
    assert "scale" not in {step["step"] for step in describe_preprocessing_v4("random_forest")["steps"]}


# --- determinism ---


@pytest.mark.parametrize("key", sorted(MODEL_SPECS_V4))
def test_training_is_deterministic(matrix, key):
    subset = slice(0, 2500)
    first = build_pipeline_v4(key).fit(matrix.X[subset], matrix.y[subset]).predict_proba(matrix.X[2500:2800])
    second = build_pipeline_v4(key).fit(matrix.X[subset], matrix.y[subset]).predict_proba(matrix.X[2500:2800])

    assert np.array_equal(first, second)


def test_every_sensitivity_variant_builds_a_working_pipeline(matrix):
    for key, variants in SENSITIVITY_VARIANTS_V4.items():
        assert len(variants) == 3
        for factory in variants.values():
            pipeline = build_pipeline_v4(key, classifier=factory()).fit(matrix.X[:600], matrix.y[:600])
            assert pipeline.predict_proba(matrix.X[:3]).shape == (3, 2)
