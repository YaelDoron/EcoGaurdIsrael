"""Tests for the V4 dataset schema: stable feature list, metadata separation, leakage protection."""
from __future__ import annotations

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_features_v4 import (
    FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4,
    FIRE_DETECTION_FEATURE_NAMES_V4,
    FORBIDDEN_FEATURE_NAMES_V4,
    LABEL_COLUMN,
    TRAINING_DATA_CSV_COLUMNS_V4,
    TRAINING_DATA_METADATA_COLUMNS_V4,
    forbidden_feature_names,
)

EXPECTED_FEATURES = (
    "satellite_low_count",
    "satellite_nominal_count",
    "satellite_high_count",
    "satellite_frp_available_ratio",
    "satellite_frp_mean",
    "satellite_frp_max",
    "satellite_brightness_available_ratio",
    "satellite_brightness_mean",
    "satellite_brightness_max",
    "news_none_count",
    "news_weak_count",
    "news_moderate_count",
    "news_strong_count",
    "news_unknown_count",
    "time_span_minutes",
    "max_pairwise_distance_km",
    "fire_danger_available",
    "fire_danger_score",
    "fire_danger_age_minutes",
)


def test_feature_schema_is_stable():
    assert FIRE_DETECTION_FEATURE_NAMES_V4 == EXPECTED_FEATURES
    assert len(set(FIRE_DETECTION_FEATURE_NAMES_V4)) == 19


def test_v4_extends_the_unchanged_v3_schema_with_only_the_three_context_features():
    assert FIRE_DETECTION_FEATURE_NAMES_V4[:16] == FIRE_DETECTION_FEATURE_NAMES_V3
    assert FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4 == (
        "fire_danger_available",
        "fire_danger_score",
        "fire_danger_age_minutes",
    )
    assert FIRE_DETECTION_FEATURE_NAMES_V4[16:] == FIRE_DANGER_CONTEXT_FEATURE_NAMES_V4


def test_fire_danger_level_is_not_an_ml_feature():
    assert "fire_danger_level" not in FIRE_DETECTION_FEATURE_NAMES_V4
    assert not any("level" in name for name in FIRE_DETECTION_FEATURE_NAMES_V4)


def test_csv_columns_are_metadata_then_features_then_label():
    assert TRAINING_DATA_CSV_COLUMNS_V4 == (
        TRAINING_DATA_METADATA_COLUMNS_V4 + FIRE_DETECTION_FEATURE_NAMES_V4 + (LABEL_COLUMN,)
    )
    assert len(set(TRAINING_DATA_CSV_COLUMNS_V4)) == len(TRAINING_DATA_CSV_COLUMNS_V4)


def test_no_metadata_or_target_column_is_an_ml_feature():
    features = set(FIRE_DETECTION_FEATURE_NAMES_V4)
    assert features.isdisjoint(TRAINING_DATA_METADATA_COLUMNS_V4)
    assert LABEL_COLUMN not in features
    assert features.isdisjoint(FORBIDDEN_FEATURE_NAMES_V4)


def test_forbidden_leakage_columns_are_excluded_from_ml_features():
    assert forbidden_feature_names(FIRE_DETECTION_FEATURE_NAMES_V4) == ()


def test_leakage_check_actually_flags_leaky_names():
    leaky = (
        "scenario_family",
        "scenario_archetype",
        "ground_truth_scenario_type",
        "is_active_fire_scenario",
        "preset_name",
        "rule_confidence",
        "rule_status",
        "ml_probability",
        "event_status",
        "suspected_flag",
        "confirmed_flag",
        "label",
        "fire_danger_level",
        "fire_danger_band",
        "as_of_utc",
        "observed_timestamp",
        "hour_of_day",
        "sample_id",
        "seed",
    )
    assert set(forbidden_feature_names(leaky)) == set(leaky)
