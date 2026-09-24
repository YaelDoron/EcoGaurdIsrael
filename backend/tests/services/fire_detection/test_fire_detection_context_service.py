"""Tests for FireDetectionContextService (Fire Danger context lookup for a candidate)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_candidate import FireDetectionCandidate
from src.models.fire_detection_context import FireDetectionContext
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.repositories.fire_danger_assessment_repository import StoredFireDangerAssessment
from src.services.fire_detection.fire_detection_context_config import MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES
from src.services.fire_detection.fire_detection_context_service import FireDetectionContextService
from src.utils.geo import destination_point

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
# An arbitrary synthetic point - deliberately not any named demo location.
CENTER_LATITUDE = 31.5
CENTER_LONGITUDE = 34.9
MAX_AGE = MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES


class FakeFireDangerRepository:
    """Returns every stored assessment regardless of the requested window.

    That is deliberately the worst case for the service: future and stale rows
    are handed back, so these tests prove the service itself enforces the
    as-of/freshness/geography rules rather than trusting the repository.
    """

    def __init__(self, stored=(), exc=None):
        self.stored = tuple(stored)
        self.exc = exc
        self.calls = []

    def get_assessed_between(self, start_time, end_time):
        self.calls.append((start_time, end_time))
        if self.exc is not None:
            raise self.exc
        return self.stored


def make_stored(
    assessment_id=1,
    minutes_old=10.0,
    score=42.5,
    level=FireDangerLevel.VERY_HIGH,
    status=FireDangerAssessmentStatus.VALID,
    area_id="area-a",
    area_latitude=CENTER_LATITUDE,
    area_longitude=CENTER_LONGITUDE,
    area_radius_km=5.0,
):
    if status is FireDangerAssessmentStatus.INSUFFICIENT_DATA:
        score, level = None, None
    assessed_at = AS_OF - timedelta(minutes=minutes_old)
    return StoredFireDangerAssessment(
        assessment_id=assessment_id,
        assessment=FireDangerAssessment(
            area_id=area_id,
            area_name=f"Name of {area_id}",
            area_latitude=area_latitude,
            area_longitude=area_longitude,
            area_radius_km=area_radius_km,
            assessed_at=assessed_at,
            status=status,
            score=score,
            level=level,
            methodology=FFWI_METHODOLOGY_NAME,
            methodology_version=FFWI_METHODOLOGY_VERSION,
        ),
        observation_ids=(),
        station_ids=(),
        created_at=assessed_at,
    )


def satellite(evidence_id=1, latitude=CENTER_LATITUDE, longitude=CENTER_LONGITUDE):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.SATELLITE,
        latitude=latitude,
        longitude=longitude,
        observed_at=AS_OF - timedelta(minutes=5),
        satellite_confidence="nominal",
    )


def news(evidence_id=2, latitude=CENTER_LATITUDE, longitude=CENTER_LONGITUDE):
    return FireDetectionEvidence(
        evidence_id=evidence_id,
        evidence_type=FireEvidenceType.NEWS,
        latitude=latitude,
        longitude=longitude,
        observed_at=AS_OF - timedelta(minutes=5),
    )


def candidate(*evidence):
    return FireDetectionCandidate(tuple(evidence or (satellite(),)))


def service_for(*stored, **kwargs):
    return FireDetectionContextService(fire_danger_repository=FakeFireDangerRepository(stored), **kwargs)


def area_center_km_north(distance_km):
    """Center of an area whose circle's center is `distance_km` due north of the candidate."""
    return destination_point(CENTER_LATITUDE, CENTER_LONGITUDE, 0.0, distance_km)


# --- available context ---


def test_recent_geographically_matching_assessment_is_returned():
    context = service_for(make_stored(assessment_id=9, minutes_old=20, score=37.0, level=FireDangerLevel.HIGH)).build_context(
        candidate(), AS_OF
    )

    assert context.fire_danger_available is True
    assert context.fire_danger_score == 37.0
    assert context.fire_danger_age_minutes == pytest.approx(20.0)
    assert context.fire_danger_level is FireDangerLevel.HIGH
    assert context.fire_danger_assessment_id == 9
    assert context.fire_danger_assessed_at == AS_OF - timedelta(minutes=20)


def test_repository_is_queried_with_the_configured_freshness_window_ending_at_as_of():
    repository = FakeFireDangerRepository([make_stored()])

    FireDetectionContextService(fire_danger_repository=repository).build_context(candidate(), AS_OF)

    assert repository.calls == [(AS_OF - timedelta(minutes=MAX_AGE), AS_OF)]


def test_latest_matching_assessment_is_selected_when_several_exist():
    context = service_for(
        make_stored(assessment_id=1, minutes_old=50, score=10.0, level=FireDangerLevel.LOW),
        make_stored(assessment_id=2, minutes_old=5, score=33.0, level=FireDangerLevel.HIGH),
        make_stored(assessment_id=3, minutes_old=25, score=20.0, level=FireDangerLevel.MODERATE),
    ).build_context(candidate(), AS_OF)

    assert context.fire_danger_assessment_id == 2
    assert context.fire_danger_score == 33.0
    assert context.fire_danger_age_minutes == pytest.approx(5.0)


def test_latest_is_selected_across_overlapping_areas_by_time_not_by_area():
    other_latitude, other_longitude = area_center_km_north(2.0)
    context = service_for(
        make_stored(assessment_id=1, minutes_old=30, area_id="area-a"),
        make_stored(
            assessment_id=2,
            minutes_old=10,
            area_id="area-b",
            area_latitude=other_latitude,
            area_longitude=other_longitude,
        ),
    ).build_context(candidate(), AS_OF)

    assert context.fire_danger_assessment_id == 2


def test_equal_assessed_at_prefers_nearest_area_center_then_highest_id():
    nearer = make_stored(assessment_id=1, minutes_old=10, area_id="near")
    farther_lat, farther_lon = area_center_km_north(3.0)
    farther = make_stored(
        assessment_id=2, minutes_old=10, area_id="far", area_latitude=farther_lat, area_longitude=farther_lon
    )
    same_area_higher_id = make_stored(assessment_id=3, minutes_old=10, area_id="near")

    assert service_for(farther, nearer).build_context(candidate(), AS_OF).fire_danger_assessment_id == 1
    assert service_for(nearer, same_area_higher_id).build_context(candidate(), AS_OF).fire_danger_assessment_id == 3


def test_zero_ffwi_score_is_reported_as_available_not_as_missing():
    context = service_for(make_stored(score=0.0, level=FireDangerLevel.LOW)).build_context(candidate(), AS_OF)

    assert context.fire_danger_available is True
    assert context.fire_danger_score == 0.0


def test_assessment_exactly_at_the_maximum_age_is_still_usable():
    context = service_for(make_stored(minutes_old=MAX_AGE)).build_context(candidate(), AS_OF)

    assert context.fire_danger_available is True
    assert context.fire_danger_age_minutes == pytest.approx(MAX_AGE)


def test_assessment_exactly_at_as_of_has_zero_age():
    context = service_for(make_stored(minutes_old=0)).build_context(candidate(), AS_OF)

    assert context.fire_danger_available is True
    assert context.fire_danger_age_minutes == 0.0


# --- unavailable context ---


def test_future_assessment_is_not_used_for_an_earlier_as_of():
    future = make_stored(assessment_id=1, minutes_old=-5)

    assert service_for(future).build_context(candidate(), AS_OF) == FireDetectionContext.unavailable()


def test_future_assessment_is_ignored_in_favor_of_an_older_valid_one():
    context = service_for(
        make_stored(assessment_id=1, minutes_old=-5, score=90.0, level=FireDangerLevel.EXTREME),
        make_stored(assessment_id=2, minutes_old=15, score=21.0, level=FireDangerLevel.MODERATE),
    ).build_context(candidate(), AS_OF)

    assert context.fire_danger_assessment_id == 2
    assert context.fire_danger_score == 21.0


def test_stale_assessment_returns_unavailable_context():
    stale = make_stored(minutes_old=MAX_AGE + 1)

    assert service_for(stale).build_context(candidate(), AS_OF) == FireDetectionContext.unavailable()


def test_insufficient_data_assessment_returns_unavailable_context():
    insufficient = make_stored(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA)

    context = service_for(insufficient).build_context(candidate(), AS_OF)

    assert context == FireDetectionContext.unavailable()
    assert context.fire_danger_score is None  # never a substitute value such as 0


def test_newer_insufficient_data_assessment_is_not_replaced_by_an_older_valid_one():
    context = service_for(
        make_stored(assessment_id=1, minutes_old=30, score=40.0),
        make_stored(assessment_id=2, minutes_old=5, status=FireDangerAssessmentStatus.INSUFFICIENT_DATA),
    ).build_context(candidate(), AS_OF)

    assert context == FireDetectionContext.unavailable()


def test_no_assessment_returns_unavailable_context():
    assert service_for().build_context(candidate(), AS_OF) == FireDetectionContext.unavailable()


def test_nearby_but_non_matching_area_is_not_selected():
    latitude, longitude = area_center_km_north(5.5)  # circle radius 5 km does not reach the candidate
    outside = make_stored(area_latitude=latitude, area_longitude=longitude, area_radius_km=5.0)

    assert service_for(outside).build_context(candidate(), AS_OF) == FireDetectionContext.unavailable()


def test_area_whose_circle_just_contains_the_candidate_is_selected():
    latitude, longitude = area_center_km_north(4.9)
    inside = make_stored(area_latitude=latitude, area_longitude=longitude, area_radius_km=5.0)

    assert service_for(inside).build_context(candidate(), AS_OF).fire_danger_available is True


def test_non_matching_area_is_skipped_even_when_it_is_newer_than_a_matching_one():
    latitude, longitude = area_center_km_north(20.0)
    context = service_for(
        make_stored(assessment_id=1, minutes_old=30, score=25.0, level=FireDangerLevel.MODERATE),
        make_stored(assessment_id=2, minutes_old=1, area_id="elsewhere", area_latitude=latitude, area_longitude=longitude),
    ).build_context(candidate(), AS_OF)

    assert context.fire_danger_assessment_id == 1


def test_area_radius_is_taken_from_each_assessment_not_a_global_constant():
    latitude, longitude = area_center_km_north(8.0)
    wide = make_stored(area_latitude=latitude, area_longitude=longitude, area_radius_km=10.0)

    assert service_for(wide).build_context(candidate(), AS_OF).fire_danger_available is True


# --- candidate location ---


def test_location_uses_satellite_priority_centroid_like_the_detection_decision():
    news_latitude, news_longitude = destination_point(CENTER_LATITUDE, CENTER_LONGITUDE, 0.0, 3.0)
    mixed_candidate = candidate(satellite(1), news(2, latitude=news_latitude, longitude=news_longitude))
    # A tiny circle around the satellite hotspot: the all-evidence centroid
    # (1.5 km away) would fall outside it, the satellite location does not.
    tight_area = make_stored(area_radius_km=1.0)

    assert service_for(tight_area).build_context(mixed_candidate, AS_OF).fire_danger_available is True


def test_news_only_candidate_uses_the_news_location():
    news_only = candidate(news(1))

    assert service_for(make_stored()).build_context(news_only, AS_OF).fire_danger_available is True


def test_accepts_a_raw_evidence_tuple_as_well_as_a_candidate():
    evidence = (satellite(1), news(2))

    from_candidate = service_for(make_stored()).build_context(FireDetectionCandidate(evidence), AS_OF)
    from_tuple = service_for(make_stored()).build_context(evidence, AS_OF)

    assert from_candidate == from_tuple
    assert from_tuple.fire_danger_available is True


def test_configured_maximum_age_is_respected():
    stored = make_stored(minutes_old=20)

    assert service_for(stored, max_fire_danger_age_minutes=30).build_context(candidate(), AS_OF).fire_danger_available
    assert not service_for(stored, max_fire_danger_age_minutes=10).build_context(candidate(), AS_OF).fire_danger_available


# --- failure handling ---


def test_repository_failure_returns_unavailable_context_instead_of_raising():
    repository = FakeFireDangerRepository(exc=RuntimeError("database is down"))

    context = FireDetectionContextService(fire_danger_repository=repository).build_context(candidate(), AS_OF)

    assert context == FireDetectionContext.unavailable()


def test_context_is_only_a_read_and_never_alters_the_candidate():
    original = candidate(satellite(1), news(2))
    evidence_before = original.evidence

    service_for(make_stored(score=95.0, level=FireDangerLevel.EXTREME)).build_context(original, AS_OF)

    assert original.evidence == evidence_before


def test_high_fire_danger_alone_cannot_create_a_candidate_it_only_annotates_one():
    # There is no API on the service that takes context and returns evidence or
    # a candidate: it only maps (existing candidate, as_of) -> FireDetectionContext.
    context = service_for(make_stored(score=95.0, level=FireDangerLevel.EXTREME)).build_context(candidate(), AS_OF)

    assert isinstance(context, FireDetectionContext)
    assert not hasattr(context, "evidence")


# --- argument validation (programming errors, not data problems) ---


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError):
        service_for().build_context(candidate(), datetime(2026, 9, 14, 12, 0))


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError):
        service_for().build_context((), AS_OF)


@pytest.mark.parametrize("value", [0, -1, True])
def test_non_positive_maximum_age_is_rejected(value):
    with pytest.raises(ValueError):
        FireDetectionContextService(fire_danger_repository=FakeFireDangerRepository(), max_fire_danger_age_minutes=value)


def test_default_maximum_age_is_a_positive_centralized_constant():
    assert MAX_FIRE_DANGER_ASSESSMENT_AGE_MINUTES > 0
