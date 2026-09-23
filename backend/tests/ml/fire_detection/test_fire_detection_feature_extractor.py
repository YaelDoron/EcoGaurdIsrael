"""Tests for FireDetectionFeatureExtractor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest

from src.calculators.fire_detection.fire_detection_config import MAX_EVIDENCE_DISTANCE_KM
from src.ml.fire_detection.fire_detection_feature_extractor import FireDetectionFeatureExtractor
from src.ml.fire_detection.fire_detection_features import FireDetectionFeatures
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType

OBSERVED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
LATITUDE = 32.731
LONGITUDE = 35.046
EARTH_RADIUS_KM = 6371.0088


def satellite(
    evidence_id: int,
    confidence: str = "nominal",
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    observed_at: datetime = OBSERVED_AT,
) -> FireDetectionEvidence:
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=latitude,
        longitude=longitude,
        observed_at=observed_at,
        satellite_confidence=confidence,
    )


def news(
    evidence_id: int,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    observed_at: datetime = OBSERVED_AT,
) -> FireDetectionEvidence:
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=latitude,
        longitude=longitude,
        observed_at=observed_at,
    )


def coordinate_north_at_distance(latitude: float, longitude: float, distance_km: float) -> tuple[float, float]:
    return latitude + math.degrees(distance_km / EARTH_RADIUS_KM), longitude


def extract(*evidence: FireDetectionEvidence) -> FireDetectionFeatures:
    return FireDetectionFeatureExtractor().extract(tuple(evidence))


def test_single_satellite_evidence_item():
    features = extract(satellite(1, "high"))

    assert features.satellite_count == 1
    assert features.news_count == 0
    assert features.satellite_high_count == 1
    assert features.satellite_nominal_count == 0
    assert features.satellite_low_count == 0


def test_single_news_evidence_item():
    features = extract(news(1))

    assert features.satellite_count == 0
    assert features.news_count == 1
    assert features.satellite_low_count == 0
    assert features.satellite_nominal_count == 0
    assert features.satellite_high_count == 0


def test_mixed_satellite_and_news():
    features = extract(satellite(1, "nominal"), news(2))

    assert features.satellite_count == 1
    assert features.news_count == 1


def test_low_satellite_confidence_is_counted():
    features = extract(satellite(1, "low", latitude=LATITUDE), satellite(2, "low", latitude=LATITUDE + 0.001))

    assert features.satellite_low_count == 2
    assert features.satellite_nominal_count == 0
    assert features.satellite_high_count == 0


def test_nominal_satellite_confidence_is_counted():
    features = extract(satellite(1, "nominal"))

    assert features.satellite_nominal_count == 1
    assert features.satellite_low_count == 0
    assert features.satellite_high_count == 0


def test_high_satellite_confidence_is_counted():
    features = extract(satellite(1, "high"))

    assert features.satellite_high_count == 1
    assert features.satellite_low_count == 0
    assert features.satellite_nominal_count == 0


def test_multiple_satellite_evidence_items_with_mixed_confidence():
    features = extract(
        satellite(1, "low", latitude=LATITUDE),
        satellite(2, "nominal", latitude=LATITUDE + 0.001),
        satellite(3, "high", latitude=LATITUDE + 0.002),
    )

    assert features.satellite_count == 3
    assert features.satellite_low_count == 1
    assert features.satellite_nominal_count == 1
    assert features.satellite_high_count == 1


def test_time_span_minutes_reflects_earliest_to_latest_gap():
    features = extract(
        satellite(1, observed_at=OBSERVED_AT),
        news(2, observed_at=OBSERVED_AT + timedelta(minutes=25)),
    )

    assert features.time_span_minutes == pytest.approx(25.0)


def test_time_span_minutes_is_independent_of_evidence_order():
    features = extract(
        satellite(1, observed_at=OBSERVED_AT + timedelta(minutes=25)),
        news(2, observed_at=OBSERVED_AT),
    )

    assert features.time_span_minutes == pytest.approx(25.0)


def test_max_pairwise_distance_km_reflects_geographic_spread():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, 2.0)
    features = extract(satellite(1, latitude=LATITUDE, longitude=LONGITUDE), news(2, latitude=lat, longitude=lon))

    assert features.max_pairwise_distance_km == pytest.approx(2.0, abs=1e-3)


def test_max_pairwise_distance_km_uses_the_farthest_pair_among_three_items():
    lat_near, lon_near = coordinate_north_at_distance(LATITUDE, LONGITUDE, 1.0)
    lat_far, lon_far = coordinate_north_at_distance(LATITUDE, LONGITUDE, MAX_EVIDENCE_DISTANCE_KM)

    features = extract(
        satellite(1, latitude=LATITUDE, longitude=LONGITUDE),
        satellite(2, latitude=lat_near, longitude=lon_near),
        satellite(3, latitude=lat_far, longitude=lon_far),
    )

    assert features.max_pairwise_distance_km == pytest.approx(MAX_EVIDENCE_DISTANCE_KM, abs=1e-2)


def test_single_evidence_item_has_zero_time_span():
    features = extract(satellite(1))

    assert features.time_span_minutes == 0.0


def test_single_evidence_item_has_zero_distance():
    features = extract(satellite(1))

    assert features.max_pairwise_distance_km == 0.0


def test_extracted_features_are_independent_of_input_order():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, 1.0)
    evidence_a = satellite(1, "high", observed_at=OBSERVED_AT)
    evidence_b = news(2, latitude=lat, longitude=lon, observed_at=OBSERVED_AT + timedelta(minutes=10))

    forward = FireDetectionFeatureExtractor().extract((evidence_a, evidence_b))
    reversed_order = FireDetectionFeatureExtractor().extract((evidence_b, evidence_a))

    assert forward == reversed_order


def test_duplicate_evidence_identity_is_rejected():
    with pytest.raises(ValueError):
        extract(satellite(1), satellite(1))


def test_disconnected_candidate_evidence_is_rejected():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, MAX_EVIDENCE_DISTANCE_KM + 0.01)

    with pytest.raises(ValueError):
        extract(satellite(1), news(2, latitude=lat, longitude=lon))


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError):
        FireDetectionFeatureExtractor().extract(())
