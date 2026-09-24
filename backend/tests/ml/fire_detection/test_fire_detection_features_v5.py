"""Task 6: the V5 feature CONTRACT (schema, nullable semantics, internal consistency)."""
from __future__ import annotations

import math

import pytest

from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.ml.fire_detection.fire_detection_features_v4 import FIRE_DETECTION_FEATURE_NAMES_V4
from src.ml.fire_detection.fire_detection_features_v5 import (
    CURRENT_EVIDENCE_FEATURE_NAMES_V5,
    FEATURE_COUNT_V5,
    FIRE_DETECTION_FEATURE_NAMES_V5,
    FORBIDDEN_FEATURE_NAMES_V5,
    HISTORY_FEATURE_NAMES_V5,
    NULLABLE_FEATURE_NAMES_V5,
    REMOVED_V4_FEATURE_NAMES,
    RETAINED_FEATURE_NAMES_V5,
    TRAINING_DATA_CSV_COLUMNS_V5,
    TRAINING_DATA_METADATA_COLUMNS_V5,
    FireDetectionFeaturesV5,
    forbidden_feature_names,
    validate_feature_names_v5,
)

EXPECTED_NAMES = (
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
    "satellite_frp_sum",
    "satellite_frp_std",
    "satellite_brightness_std",
    "satellite_night_fraction",
    "satellite_cluster_radius_km",
    "news_satellite_lag_minutes",
    "satellite_pass_count",
    "satellite_history_span_minutes",
    "satellite_centroid_stability_km",
    "satellite_frp_trend_per_hour",
    "satellite_brightness_trend_per_hour",
)
NAN = math.nan


def satellite_only_vector(**overrides) -> dict:
    """A valid one-hotspot, first-detection vector."""
    values = {
        "satellite_low_count": 0,
        "satellite_nominal_count": 1,
        "satellite_high_count": 0,
        "satellite_frp_available_ratio": 1.0,
        "satellite_frp_mean": 8.0,
        "satellite_frp_max": 8.0,
        "satellite_brightness_available_ratio": 1.0,
        "satellite_brightness_mean": 330.0,
        "satellite_brightness_max": 330.0,
        "news_none_count": 0,
        "news_weak_count": 0,
        "news_moderate_count": 0,
        "news_strong_count": 0,
        "news_unknown_count": 0,
        "satellite_frp_sum": 8.0,
        "satellite_frp_std": 0.0,
        "satellite_brightness_std": 0.0,
        "satellite_night_fraction": 1.0,
        "satellite_cluster_radius_km": 0.0,
        "news_satellite_lag_minutes": NAN,
        "satellite_pass_count": 1,
        "satellite_history_span_minutes": 0.0,
        "satellite_centroid_stability_km": NAN,
        "satellite_frp_trend_per_hour": NAN,
        "satellite_brightness_trend_per_hour": NAN,
    }
    values.update(overrides)
    return values


def build(**overrides) -> FireDetectionFeaturesV5:
    return FireDetectionFeaturesV5.from_mapping(satellite_only_vector(**overrides))


# --- schema ---


def test_canonical_schema_is_exactly_the_specified_ordered_25_features():
    assert FIRE_DETECTION_FEATURE_NAMES_V5 == EXPECTED_NAMES
    assert FEATURE_COUNT_V5 == 25
    assert len(set(FIRE_DETECTION_FEATURE_NAMES_V5)) == 25


def test_the_14_retained_v4_features_keep_their_names_and_relative_order():
    assert FIRE_DETECTION_FEATURE_NAMES_V5[:14] == RETAINED_FEATURE_NAMES_V5
    assert RETAINED_FEATURE_NAMES_V5 == tuple(n for n in FIRE_DETECTION_FEATURE_NAMES_V3 if n not in ("time_span_minutes", "max_pairwise_distance_km"))
    assert set(RETAINED_FEATURE_NAMES_V5) <= set(FIRE_DETECTION_FEATURE_NAMES_V4)
    assert len(RETAINED_FEATURE_NAMES_V5) == 14


def test_removed_v4_features_are_absent():
    assert set(REMOVED_V4_FEATURE_NAMES) == {
        "time_span_minutes",
        "max_pairwise_distance_km",
        "fire_danger_available",
        "fire_danger_score",
        "fire_danger_age_minutes",
    }
    assert not set(REMOVED_V4_FEATURE_NAMES) & set(FIRE_DETECTION_FEATURE_NAMES_V5)


def test_fire_danger_and_ffwi_are_not_in_v5():
    assert not [n for n in FIRE_DETECTION_FEATURE_NAMES_V5 if "danger" in n or "ffwi" in n]
    assert not [c for c in TRAINING_DATA_CSV_COLUMNS_V5 if "danger" in c or "ffwi" in c]


def test_feature_groups_partition_the_schema():
    assert RETAINED_FEATURE_NAMES_V5 + CURRENT_EVIDENCE_FEATURE_NAMES_V5 + HISTORY_FEATURE_NAMES_V5 == FIRE_DETECTION_FEATURE_NAMES_V5
    assert set(NULLABLE_FEATURE_NAMES_V5) <= set(FIRE_DETECTION_FEATURE_NAMES_V5)


def test_metadata_never_enters_the_feature_vector():
    assert not set(TRAINING_DATA_METADATA_COLUMNS_V5) & set(FIRE_DETECTION_FEATURE_NAMES_V5)
    assert "label" not in FIRE_DETECTION_FEATURE_NAMES_V5
    assert forbidden_feature_names() == ()  # `..._per_hour` trends are legitimate; clock-time names are not
    assert set(TRAINING_DATA_METADATA_COLUMNS_V5) <= FORBIDDEN_FEATURE_NAMES_V5
    for leaking in ("regime", "environment_id", "latent_subtype", "pair_id", "hour_of_day", "fire_danger_score", "label"):
        assert forbidden_feature_names((leaking,)) == (leaking,)
    assert TRAINING_DATA_METADATA_COLUMNS_V5 == (
        "sample_id", "seed", "environment_id", "regime", "latent_subtype", "pair_id", "pair_type", "as_of_utc",
    )
    assert TRAINING_DATA_CSV_COLUMNS_V5[-1] == "label"


def test_validate_feature_names_rejects_reordered_or_extended_lists():
    assert validate_feature_names_v5(list(EXPECTED_NAMES)) == EXPECTED_NAMES
    with pytest.raises(ValueError, match="same_set_different_order=True"):
        validate_feature_names_v5(list(reversed(EXPECTED_NAMES)))
    with pytest.raises(ValueError, match="unexpected"):
        validate_feature_names_v5((*EXPECTED_NAMES, "fire_danger_score"))
    with pytest.raises(ValueError, match="missing"):
        validate_feature_names_v5(EXPECTED_NAMES[:-1])


# --- vector validation ---


def test_a_valid_vector_round_trips_through_every_representation():
    features = build()
    assert features.names == EXPECTED_NAMES
    assert len(features.as_tuple()) == 25
    assert features["satellite_frp_sum"] == 8.0
    assert features.as_dict()["satellite_pass_count"] == 1
    nullable = features.to_nullable_tuple()
    assert nullable[EXPECTED_NAMES.index("news_satellite_lag_minutes")] is None
    assert FireDetectionFeaturesV5.from_optional_values(nullable).as_tuple()[:19] == features.as_tuple()[:19]
    assert isinstance(features["satellite_pass_count"], int)


def test_wrong_length_bool_and_non_numeric_are_rejected():
    with pytest.raises(ValueError, match="exactly 25"):
        FireDetectionFeaturesV5((0.0,) * 24)
    values = list(build().as_tuple())
    values[0] = True
    with pytest.raises(ValueError, match="numeric"):
        FireDetectionFeaturesV5(tuple(values))
    values[0] = "1"
    with pytest.raises(ValueError, match="numeric"):
        FireDetectionFeaturesV5(tuple(values))


def test_from_mapping_needs_exactly_the_canonical_keys():
    good = satellite_only_vector()
    with pytest.raises(ValueError, match="missing"):
        FireDetectionFeaturesV5.from_mapping({k: v for k, v in good.items() if k != "satellite_frp_sum"})
    with pytest.raises(ValueError, match="unexpected"):
        FireDetectionFeaturesV5.from_mapping({**good, "fire_danger_score": 1.0})


def test_none_is_accepted_only_for_nullable_features():
    with pytest.raises(ValueError, match="must not be missing"):
        FireDetectionFeaturesV5.from_optional_values([None] + [0] * 24)


@pytest.mark.parametrize("name", [n for n in EXPECTED_NAMES if n not in NULLABLE_FEATURE_NAMES_V5])
def test_nan_is_rejected_in_non_nullable_features(name):
    with pytest.raises(ValueError):
        build(**{name: NAN})


@pytest.mark.parametrize("name", [n for n in EXPECTED_NAMES if not n.endswith("_count")])
def test_infinity_is_rejected(name):
    with pytest.raises(ValueError):
        build(**{name: math.inf})


def test_counts_must_be_whole_and_non_negative():
    with pytest.raises(ValueError, match="whole number"):
        build(satellite_low_count=0.5)
    with pytest.raises(ValueError):
        build(news_weak_count=-1)


def test_ratios_and_measurements_have_ranges_but_signed_features_may_be_negative():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build(satellite_frp_available_ratio=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build(satellite_night_fraction=1.2)
    with pytest.raises(ValueError, match="non-negative"):
        build(satellite_frp_sum=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        build(satellite_cluster_radius_km=-0.1)
    # signed by nature: a lag (news first) and a declining trend
    three_passes = dict(
        satellite_pass_count=3,
        satellite_history_span_minutes=360.0,
        satellite_centroid_stability_km=0.2,
        satellite_frp_trend_per_hour=-2.5,
        satellite_brightness_trend_per_hour=-1.0,
        news_moderate_count=1,
        news_satellite_lag_minutes=-20.0,
    )
    assert build(**three_passes)["satellite_frp_trend_per_hour"] == -2.5


def test_a_candidate_needs_at_least_one_evidence_item():
    with pytest.raises(ValueError, match="at least one evidence item"):
        build(satellite_nominal_count=0, satellite_frp_sum=0.0, satellite_night_fraction=NAN, satellite_cluster_radius_km=NAN,
              satellite_pass_count=0, satellite_history_span_minutes=NAN)


# --- missing-value semantics: NaN only where "cannot be computed" ---


def test_one_pass_has_no_centroid_stability_or_trend():
    with pytest.raises(ValueError, match="at least 2 passes"):
        build(satellite_centroid_stability_km=0.0)  # zero would claim "observed twice and perfectly stable"
    with pytest.raises(ValueError, match="at least 3 passes"):
        build(satellite_frp_trend_per_hour=0.0)
    with pytest.raises(ValueError, match="at least 3 passes"):
        build(satellite_brightness_trend_per_hour=0.0)


def test_two_passes_have_stability_but_still_no_trend():
    two = dict(satellite_pass_count=2, satellite_history_span_minutes=180.0, satellite_centroid_stability_km=0.3)
    assert math.isnan(build(**two)["satellite_frp_trend_per_hour"])
    with pytest.raises(ValueError, match="at least 3 passes"):
        build(**two, satellite_frp_trend_per_hour=1.0)
    with pytest.raises(ValueError, match="must be computed from >= 2 passes"):
        build(satellite_pass_count=2, satellite_history_span_minutes=180.0)  # stability left NaN


def test_a_single_pass_spans_zero_minutes_and_a_span_is_required_once_a_pass_exists():
    with pytest.raises(ValueError, match="0.0 for a single pass"):
        build(satellite_history_span_minutes=15.0)
    with pytest.raises(ValueError, match="must be computed"):
        build(satellite_history_span_minutes=NAN)


def test_news_only_candidates_have_nan_satellite_geometry_and_no_pass():
    news_only = dict(
        satellite_nominal_count=0, news_strong_count=1,
        satellite_frp_available_ratio=0.0, satellite_frp_mean=0.0, satellite_frp_max=0.0,
        satellite_brightness_available_ratio=0.0, satellite_brightness_mean=0.0, satellite_brightness_max=0.0,
        satellite_frp_sum=0.0, satellite_night_fraction=NAN, satellite_cluster_radius_km=NAN,
        satellite_pass_count=0, satellite_history_span_minutes=NAN,
    )
    features = build(**news_only)
    assert features["satellite_pass_count"] == 0 and math.isnan(features["satellite_cluster_radius_km"])
    with pytest.raises(ValueError, match="no satellite pass"):
        build(**{**news_only, "satellite_history_span_minutes": 0.0})  # a span of 0.0 would invent a pass
    with pytest.raises(ValueError, match="without current satellite evidence"):
        build(**{**news_only, "satellite_cluster_radius_km": 0.0})  # 0.0 would read as "perfectly compact"
    with pytest.raises(ValueError, match="requires satellite_pass_count >= 1"):
        build(**{**news_only, "satellite_nominal_count": 1, "satellite_pass_count": 0})


def test_a_news_only_candidate_may_still_inherit_history_passes():
    features = build(
        satellite_nominal_count=0, news_moderate_count=1,
        satellite_frp_available_ratio=0.0, satellite_frp_mean=0.0, satellite_frp_max=0.0,
        satellite_brightness_available_ratio=0.0, satellite_brightness_mean=0.0, satellite_brightness_max=0.0,
        satellite_frp_sum=0.0, satellite_night_fraction=NAN, satellite_cluster_radius_km=NAN,
        satellite_pass_count=2, satellite_history_span_minutes=180.0, satellite_centroid_stability_km=0.2,
    )
    assert features["satellite_pass_count"] == 2


def test_the_lag_needs_both_source_families_and_is_never_zero_by_default():
    with pytest.raises(ValueError, match="BOTH"):
        build(news_satellite_lag_minutes=0.0)  # satellite only: 0.0 would claim simultaneous news
    both = dict(news_weak_count=1)
    with pytest.raises(ValueError, match="must be computed when both"):
        build(**both)  # lag left NaN although both families exist
    assert build(**both, news_satellite_lag_minutes=12.5)["news_satellite_lag_minutes"] == 12.5
    assert build(**both, news_satellite_lag_minutes=0.0)["news_satellite_lag_minutes"] == 0.0  # a real 0


def test_unknown_day_night_is_nan_not_zero_percent_night():
    unknown = build(satellite_night_fraction=NAN)
    assert math.isnan(unknown["satellite_night_fraction"])
    assert build(satellite_night_fraction=0.0)["satellite_night_fraction"] == 0.0  # a real "all daytime"
