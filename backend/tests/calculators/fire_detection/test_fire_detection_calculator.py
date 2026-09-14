"""Tests for FireDetectionCalculator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import math

import pytest

from src.calculators.fire_detection.fire_detection_calculator import FireDetectionCalculator
from src.calculators.fire_detection.fire_detection_config import (
    CONFIRMED_THRESHOLD,
    MAX_EVIDENCE_DISTANCE_KM,
)
from src.models import FireDetectionEvidence, FireDetectionStatus, FireEvidenceRef, FireEvidenceType

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


def evaluate(*evidence: FireDetectionEvidence):
    return FireDetectionCalculator().evaluate(tuple(evidence))


def evidence_ref(evidence_type: FireEvidenceType, evidence_id: int) -> FireEvidenceRef:
    return FireEvidenceRef(evidence_type=evidence_type, evidence_id=evidence_id)


def satellite_ref(evidence_id: int) -> FireEvidenceRef:
    return evidence_ref(FireEvidenceType.SATELLITE, evidence_id)


def news_ref(evidence_id: int) -> FireEvidenceRef:
    return evidence_ref(FireEvidenceType.NEWS, evidence_id)


def coordinate_north_at_distance(latitude: float, longitude: float, distance_km: float) -> tuple[float, float]:
    return latitude + math.degrees(distance_km / EARTH_RADIUS_KM), longitude


def test_news_only_evidence_is_suspected():
    decision = evaluate(news(1))

    assert decision.confidence == pytest.approx(0.50)
    assert decision.status is FireDetectionStatus.SUSPECTED


def test_low_satellite_only_is_no_event():
    decision = evaluate(satellite(1, "low"))

    assert decision.confidence == pytest.approx(0.40)
    assert decision.status is FireDetectionStatus.NO_EVENT
    assert decision.latitude is None
    assert decision.longitude is None


def test_nominal_satellite_only_is_suspected():
    decision = evaluate(satellite(1, "nominal"))

    assert decision.confidence == pytest.approx(0.60)
    assert decision.status is FireDetectionStatus.SUSPECTED


def test_high_satellite_only_is_suspected():
    decision = evaluate(satellite(1, "high"))

    assert decision.confidence == pytest.approx(0.75)
    assert decision.status is FireDetectionStatus.SUSPECTED


def test_nominal_satellite_and_correlated_news_is_confirmed():
    decision = evaluate(satellite(1, "nominal"), news(2))

    assert decision.confidence == pytest.approx(CONFIRMED_THRESHOLD)
    assert decision.status is FireDetectionStatus.CONFIRMED


def test_high_satellite_and_correlated_news_is_confirmed():
    decision = evaluate(satellite(1, "high"), news(2))

    assert decision.confidence == pytest.approx(0.875)
    assert decision.status is FireDetectionStatus.CONFIRMED


def test_confidence_is_deterministic():
    evidence = (satellite(1, "high"), news(2))

    assert FireDetectionCalculator().evaluate(evidence) == FireDetectionCalculator().evaluate(evidence)


@pytest.mark.parametrize(
    "confidence_label, expected_confidence",
    [
        ("low", 0.40),
        ("nominal", 0.60),
        ("high", 0.75),
    ],
)
def test_multiple_satellite_items_do_not_inflate_family_contribution(
    confidence_label,
    expected_confidence,
):
    decision = evaluate(
        satellite(1, confidence_label, latitude=LATITUDE),
        satellite(2, confidence_label, latitude=LATITUDE + 0.001),
        satellite(3, confidence_label, latitude=LATITUDE + 0.002),
    )

    assert decision.confidence == pytest.approx(expected_confidence)
    assert decision.supporting_evidence == (satellite_ref(1), satellite_ref(2), satellite_ref(3))


def test_multiple_news_reports_do_not_repeatedly_add_news_confidence():
    decision = evaluate(
        news(1, latitude=LATITUDE),
        news(2, latitude=LATITUDE + 0.001),
        news(3, latitude=LATITUDE + 0.002),
    )

    assert decision.confidence == pytest.approx(0.50)
    assert decision.status is FireDetectionStatus.SUSPECTED
    assert decision.supporting_evidence == (news_ref(1), news_ref(2), news_ref(3))


def test_high_satellite_and_multiple_news_equals_high_satellite_and_one_news_confidence():
    single_news = evaluate(satellite(1, "high"), news(2))
    multiple_news = evaluate(
        satellite(1, "high"),
        news(2, latitude=LATITUDE + 0.001),
        news(3, latitude=LATITUDE + 0.002),
    )

    assert multiple_news.confidence == pytest.approx(single_news.confidence)
    assert multiple_news.supporting_evidence == (news_ref(2), news_ref(3), satellite_ref(1))


def test_overlapping_numeric_ids_from_different_sources_are_preserved():
    decision = evaluate(satellite(5, "nominal"), news(5))

    assert decision.confidence == pytest.approx(CONFIRMED_THRESHOLD)
    assert decision.supporting_evidence == (news_ref(5), satellite_ref(5))


def test_same_coordinates_correlate():
    decision = evaluate(satellite(1), news(2))

    assert decision.status is FireDetectionStatus.CONFIRMED


def test_nearby_coordinates_inside_distance_limit_correlate():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, 1.0)

    decision = evaluate(satellite(1), news(2, latitude=lat, longitude=lon))

    assert decision.status is FireDetectionStatus.CONFIRMED


def test_coordinates_at_distance_boundary_correlate():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, MAX_EVIDENCE_DISTANCE_KM)

    decision = evaluate(satellite(1), news(2, latitude=lat, longitude=lon))

    assert decision.status is FireDetectionStatus.CONFIRMED


def test_coordinates_beyond_distance_limit_are_not_fused():
    lat, lon = coordinate_north_at_distance(LATITUDE, LONGITUDE, MAX_EVIDENCE_DISTANCE_KM + 0.01)

    with pytest.raises(ValueError):
        evaluate(satellite(1), news(2, latitude=lat, longitude=lon))


def test_carmel_and_distant_golan_evidence_are_not_combined():
    with pytest.raises(ValueError):
        evaluate(
            satellite(1, latitude=32.731, longitude=35.046),
            news(2, latitude=33.085, longitude=35.780),
        )


@pytest.mark.parametrize("minutes", [0, 30, 60])
def test_time_differences_within_or_at_limit_correlate(minutes):
    decision = evaluate(satellite(1), news(2, observed_at=OBSERVED_AT + timedelta(minutes=minutes)))

    assert decision.status is FireDetectionStatus.CONFIRMED


def test_time_difference_over_limit_does_not_correlate():
    with pytest.raises(ValueError):
        evaluate(satellite(1), news(2, observed_at=OBSERVED_AT + timedelta(minutes=60, seconds=1)))


def test_temporal_correlation_is_order_independent():
    decision = evaluate(
        satellite(1, observed_at=OBSERVED_AT + timedelta(minutes=30)),
        news(2, observed_at=OBSERVED_AT),
    )

    assert decision.status is FireDetectionStatus.CONFIRMED


def test_single_satellite_location_uses_satellite_coordinates():
    decision = evaluate(satellite(1, "nominal", latitude=32.0, longitude=35.0))

    assert decision.latitude == pytest.approx(32.0)
    assert decision.longitude == pytest.approx(35.0)


def test_multiple_satellite_location_uses_satellite_centroid():
    decision = evaluate(
        satellite(1, "nominal", latitude=32.0, longitude=35.0),
        satellite(2, "nominal", latitude=32.02, longitude=35.04),
    )

    assert decision.latitude == pytest.approx(32.01)
    assert decision.longitude == pytest.approx(35.02)


def test_satellite_and_news_location_derives_from_satellite_not_news():
    decision = evaluate(
        satellite(1, "nominal", latitude=32.0, longitude=35.0),
        news(2, latitude=32.02, longitude=35.04),
    )

    assert decision.latitude == pytest.approx(32.0)
    assert decision.longitude == pytest.approx(35.0)


def test_news_only_location_uses_news_coordinates():
    decision = evaluate(news(1, latitude=32.0, longitude=35.0))

    assert decision.latitude == pytest.approx(32.0)
    assert decision.longitude == pytest.approx(35.0)


def test_multiple_news_only_location_uses_news_centroid():
    decision = evaluate(
        news(1, latitude=32.0, longitude=35.0),
        news(2, latitude=32.02, longitude=35.04),
    )

    assert decision.latitude == pytest.approx(32.01)
    assert decision.longitude == pytest.approx(35.02)


def test_duplicate_same_source_evidence_identity_is_rejected_deterministically():
    with pytest.raises(ValueError):
        evaluate(satellite(1), satellite(1))


def test_empty_evidence_returns_no_event():
    decision = evaluate()

    assert decision.confidence == pytest.approx(0.0)
    assert decision.status is FireDetectionStatus.NO_EVENT
    assert decision.supporting_evidence == ()


def test_fire_detection_calculator_has_no_fire_danger_dependency():
    module = importlib.import_module("src.calculators.fire_detection.fire_detection_calculator")

    assert "FireDangerAssessment" not in module.__dict__
    assert "FFWICalculator" not in module.__dict__
