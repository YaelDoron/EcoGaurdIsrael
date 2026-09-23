"""Tests for FireDetectionFeatureExtractorV3."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_config import MAX_EVIDENCE_DISTANCE_KM
from src.ml.fire_detection.fire_detection_feature_extractor_v3 import FireDetectionFeatureExtractorV3
from src.ml.fire_detection.fire_detection_features_v3 import FIRE_DETECTION_FEATURE_NAMES_V3
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength

OBSERVED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
LATITUDE = 32.731
LONGITUDE = 35.046


def satellite(
    evidence_id: int,
    confidence: str = "nominal",
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    observed_at: datetime = OBSERVED_AT,
    frp: float | None = None,
    brightness: float | None = None,
) -> FireDetectionEvidence:
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=latitude,
        longitude=longitude,
        observed_at=observed_at,
        satellite_confidence=confidence,
        satellite_frp=frp,
        satellite_brightness=brightness,
    )


def news(
    evidence_id: int,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    observed_at: datetime = OBSERVED_AT,
    signal: NewsWildfireSignalStrength | None = None,
) -> FireDetectionEvidence:
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=latitude,
        longitude=longitude,
        observed_at=observed_at,
        news_wildfire_signal_strength=signal,
    )


def extract(*evidence: FireDetectionEvidence):
    return FireDetectionFeatureExtractorV3().extract(tuple(evidence))


# --- FRP mean/max/availability ---


def test_frp_mean_and_max_over_available_values():
    features = extract(
        satellite(1, frp=10.0, latitude=LATITUDE),
        satellite(2, frp=30.0, latitude=LATITUDE + 0.001),
    )

    assert features.satellite_frp_mean == pytest.approx(20.0)
    assert features.satellite_frp_max == pytest.approx(30.0)


def test_frp_available_ratio_counts_only_measured_items():
    features = extract(
        satellite(1, frp=10.0, latitude=LATITUDE),
        satellite(2, frp=None, latitude=LATITUDE + 0.001),
    )

    assert features.satellite_frp_available_ratio == pytest.approx(0.5)


def test_missing_frp_for_all_satellite_items_yields_zero_not_none():
    features = extract(satellite(1, frp=None))

    assert features.satellite_frp_available_ratio == 0.0
    assert features.satellite_frp_mean == 0.0
    assert features.satellite_frp_max == 0.0


# --- brightness mean/max/availability ---


def test_brightness_mean_and_max_over_available_values():
    features = extract(
        satellite(1, brightness=300.0, latitude=LATITUDE),
        satellite(2, brightness=340.0, latitude=LATITUDE + 0.001),
    )

    assert features.satellite_brightness_mean == pytest.approx(320.0)
    assert features.satellite_brightness_max == pytest.approx(340.0)


def test_brightness_available_ratio_counts_only_measured_items():
    features = extract(
        satellite(1, brightness=300.0, latitude=LATITUDE),
        satellite(2, brightness=None, latitude=LATITUDE + 0.001),
        satellite(3, brightness=None, latitude=LATITUDE + 0.002),
    )

    assert features.satellite_brightness_available_ratio == pytest.approx(1 / 3)


def test_missing_brightness_for_all_satellite_items_yields_zero():
    features = extract(satellite(1, brightness=None))

    assert features.satellite_brightness_available_ratio == 0.0
    assert features.satellite_brightness_mean == 0.0
    assert features.satellite_brightness_max == 0.0


# --- no satellite items at all ---


def test_no_satellite_items_yields_zero_frp_and_brightness_stats():
    features = extract(news(1, signal=NewsWildfireSignalStrength.STRONG))

    assert features.satellite_frp_available_ratio == 0.0
    assert features.satellite_frp_mean == 0.0
    assert features.satellite_frp_max == 0.0
    assert features.satellite_brightness_available_ratio == 0.0
    assert features.satellite_brightness_mean == 0.0
    assert features.satellite_brightness_max == 0.0
    assert features.satellite_low_count == 0
    assert features.satellite_nominal_count == 0
    assert features.satellite_high_count == 0


# --- news signal counts ---


def test_news_signal_counts_none_weak_moderate_strong_unknown():
    features = extract(
        news(1, latitude=LATITUDE, signal=NewsWildfireSignalStrength.NONE),
        news(2, latitude=LATITUDE + 0.001, signal=NewsWildfireSignalStrength.WEAK),
        news(3, latitude=LATITUDE + 0.002, signal=NewsWildfireSignalStrength.MODERATE),
        news(4, latitude=LATITUDE + 0.003, signal=NewsWildfireSignalStrength.STRONG),
        news(5, latitude=LATITUDE + 0.004, signal=None),
    )

    assert features.news_none_count == 1
    assert features.news_weak_count == 1
    assert features.news_moderate_count == 1
    assert features.news_strong_count == 1
    assert features.news_unknown_count == 1


def test_no_news_items_yields_zero_news_counts():
    features = extract(satellite(1))

    assert features.news_none_count == 0
    assert features.news_weak_count == 0
    assert features.news_moderate_count == 0
    assert features.news_strong_count == 0
    assert features.news_unknown_count == 0


# --- candidate compositions ---


def test_satellite_only_candidate():
    features = extract(satellite(1, "high", frp=50.0, brightness=320.0))

    assert features.satellite_high_count == 1
    assert features.news_none_count == 0
    assert features.news_unknown_count == 0


def test_news_only_candidate():
    features = extract(news(1, signal=NewsWildfireSignalStrength.MODERATE))

    assert features.satellite_low_count == 0
    assert features.satellite_nominal_count == 0
    assert features.satellite_high_count == 0
    assert features.news_moderate_count == 1


def test_mixed_satellite_and_news_candidate():
    features = extract(
        satellite(1, "low", latitude=LATITUDE, frp=5.0),
        satellite(2, "high", latitude=LATITUDE + 0.001, frp=90.0),
        news(3, latitude=LATITUDE + 0.002, signal=NewsWildfireSignalStrength.STRONG),
    )

    assert features.satellite_low_count == 1
    assert features.satellite_high_count == 1
    assert features.news_strong_count == 1
    assert features.satellite_frp_mean == pytest.approx(47.5)
    assert features.satellite_frp_max == pytest.approx(90.0)


# --- determinism / ordering ---


def test_extracted_features_are_independent_of_input_order():
    evidence_a = satellite(1, "high", frp=40.0, observed_at=OBSERVED_AT)
    evidence_b = news(2, signal=NewsWildfireSignalStrength.WEAK, observed_at=OBSERVED_AT + timedelta(minutes=10))

    forward = FireDetectionFeatureExtractorV3().extract((evidence_a, evidence_b))
    reversed_order = FireDetectionFeatureExtractorV3().extract((evidence_b, evidence_a))

    assert forward == reversed_order


# --- time span / geographic spread (shared geometry logic) ---


def test_time_span_minutes_reflects_earliest_to_latest_gap():
    features = extract(
        satellite(1, observed_at=OBSERVED_AT),
        news(2, observed_at=OBSERVED_AT + timedelta(minutes=25)),
    )

    assert features.time_span_minutes == pytest.approx(25.0)


def test_single_evidence_item_has_zero_time_span_and_distance():
    features = extract(satellite(1))

    assert features.time_span_minutes == 0.0
    assert features.max_pairwise_distance_km == 0.0


def test_max_pairwise_distance_km_reflects_geographic_spread():
    import math

    earth_radius_km = 6371.0088
    lat_offset = math.degrees(2.0 / earth_radius_km)
    features = extract(satellite(1, latitude=LATITUDE), news(2, latitude=LATITUDE + lat_offset))

    assert features.max_pairwise_distance_km == pytest.approx(2.0, abs=1e-3)


# --- no metadata / raw-location leakage ---


def test_feature_vector_length_and_order_match_central_v3_schema():
    features = extract(satellite(1, "high", frp=40.0, brightness=320.0), news(2, signal=NewsWildfireSignalStrength.STRONG))

    assert len(features.as_tuple()) == len(FIRE_DETECTION_FEATURE_NAMES_V3)
    assert list(features.as_dict().keys()) == list(FIRE_DETECTION_FEATURE_NAMES_V3)


def test_as_dict_never_contains_location_or_identity_fields():
    features = extract(satellite(1, "high", frp=40.0), news(2, signal=NewsWildfireSignalStrength.STRONG))

    feature_dict = features.as_dict()
    for leaked_field in ("latitude", "longitude", "location_name", "evidence_id", "satellite", "instrument"):
        assert leaked_field not in feature_dict


# --- validity / rejection ---


def test_duplicate_evidence_identity_is_rejected():
    with pytest.raises(ValueError):
        extract(satellite(1), satellite(1))


def test_disconnected_candidate_evidence_is_rejected():
    import math

    earth_radius_km = 6371.0088
    lat_offset = math.degrees((MAX_EVIDENCE_DISTANCE_KM + 0.01) / earth_radius_km)
    with pytest.raises(ValueError):
        extract(satellite(1), news(2, latitude=LATITUDE + lat_offset))


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractorV3().extract(())
