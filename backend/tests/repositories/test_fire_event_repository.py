"""Unit tests for FireEventRepository using SQLite in-memory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import pytest
from sqlalchemy import select

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.models import FireEvent, FireEventStatus, FireEvidenceRef, FireEvidenceType, SatelliteHotspot, WildfireReport
from src.repositories.exceptions import FireEventRepositoryError
from src.repositories.fire_event_config import ACTIVE_EVENT_MATCH_DISTANCE_KM
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

DETECTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=10)
EARTH_RADIUS_KM = 6371.0088


@pytest.fixture
def repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def news_repository(sqlite_session_factory) -> NewsRepository:
    return NewsRepository(session_factory=sqlite_session_factory)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        status=FireEventStatus.SUSPECTED,
        detection_confidence=0.6,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def make_hotspot(**overrides) -> SatelliteHotspot:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=DETECTED_AT,
        confidence="h",
        satellite="N20",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


def make_report(**overrides) -> WildfireReport:
    source_url = overrides.pop("source_url", f"https://example.com/fire-event/{DETECTED_AT.timestamp()}/{id(overrides)}")
    defaults = dict(
        source_url=source_url,
        source_feed="Example",
        title="Wildfire reported",
        summary="Smoke observed",
        location_name="Carmel",
        latitude=32.731,
        longitude=35.046,
        published_at=DETECTED_AT,
        fetched_at=DETECTED_AT + timedelta(minutes=5),
    )
    defaults.update(overrides)
    return WildfireReport(**defaults)


def persist_satellite(satellite_repository: SatelliteHotspotRepository, **overrides) -> int:
    as_of = overrides.get("detected_at", DETECTED_AT) + timedelta(minutes=1)
    satellite_repository.save_hotspot(make_hotspot(**overrides))
    return satellite_repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)[0].id


def persist_news(news_repository: NewsRepository, **overrides) -> int:
    observed_at = overrides.get("published_at", DETECTED_AT) or overrides.get("fetched_at", DETECTED_AT)
    news_repository.save_report(make_report(**overrides))
    return news_repository.get_recent_reports(as_of=observed_at + timedelta(minutes=1), lookback_minutes=120)[0].id


def satellite_ref(evidence_id: int) -> FireEvidenceRef:
    return FireEvidenceRef(FireEvidenceType.SATELLITE, evidence_id)


def news_ref(evidence_id: int) -> FireEvidenceRef:
    return FireEvidenceRef(FireEvidenceType.NEWS, evidence_id)


def coordinate_north_at_distance(distance_km: float) -> tuple[float, float]:
    return 32.731 + math.degrees(distance_km / EARTH_RADIUS_KM), 35.046


def create_event_with_satellite(
    repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
    **event_overrides,
) -> StoredFireEvent:
    hotspot_id = persist_satellite(satellite_repository, detected_at=event_overrides.get("detected_at", DETECTED_AT))
    return repository.create_event(make_event(**event_overrides), (satellite_ref(hotspot_id),))


def test_create_suspected_event(repository, satellite_repository):
    hotspot_id = persist_satellite(satellite_repository)

    saved = repository.create_event(make_event(status=FireEventStatus.SUSPECTED), (satellite_ref(hotspot_id),))

    assert isinstance(saved, StoredFireEvent)
    assert saved.id > 0
    assert saved.event.status is FireEventStatus.SUSPECTED


def test_create_confirmed_event(repository, satellite_repository):
    hotspot_id = persist_satellite(satellite_repository)

    saved = repository.create_event(
        make_event(status=FireEventStatus.CONFIRMED, detection_confidence=0.85),
        (satellite_ref(hotspot_id),),
    )

    assert saved.event.status is FireEventStatus.CONFIRMED
    assert saved.event.detection_confidence == pytest.approx(0.85)


def test_stored_fields_match_domain_values(repository, satellite_repository):
    hotspot_id = persist_satellite(satellite_repository)
    event = make_event(latitude=32.8, longitude=35.1, detection_confidence=0.75)

    saved = repository.create_event(event, (satellite_ref(hotspot_id),))
    found = repository.get_by_id(saved.id)

    assert found.event == event
    assert found.supporting_evidence == (satellite_ref(hotspot_id),)


def test_created_at_is_populated(repository, satellite_repository, sqlite_session_factory):
    hotspot_id = persist_satellite(satellite_repository)

    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    session = sqlite_session_factory()
    created_at = session.execute(select(FireEventDB.created_at).where(FireEventDB.id == saved.id)).scalar_one()
    session.close()
    assert isinstance(created_at, datetime)


def test_satellite_evidence_relationship_saved(repository, satellite_repository):
    hotspot_id = persist_satellite(satellite_repository)

    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    assert repository.get_evidence_refs(saved.id) == (satellite_ref(hotspot_id),)


def test_news_evidence_relationship_saved(repository, news_repository):
    report_id = persist_news(news_repository)

    saved = repository.create_event(make_event(), (news_ref(report_id),))

    assert repository.get_evidence_refs(saved.id) == (news_ref(report_id),)


def test_overlapping_numeric_ids_across_source_types_are_preserved(repository, satellite_repository, news_repository):
    hotspot_id = persist_satellite(satellite_repository)
    report_id = persist_news(news_repository)
    assert hotspot_id == report_id

    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id), news_ref(report_id)))

    assert repository.get_evidence_refs(saved.id) == (news_ref(report_id), satellite_ref(hotspot_id))


def test_multiple_evidence_refs_persist_correctly(repository, satellite_repository, news_repository):
    first_hotspot_id = persist_satellite(satellite_repository, detected_at=DETECTED_AT)
    second_hotspot_id = persist_satellite(satellite_repository, detected_at=DETECTED_AT + timedelta(minutes=1))
    report_id = persist_news(news_repository)

    saved = repository.create_event(
        make_event(),
        (satellite_ref(second_hotspot_id), news_ref(report_id), satellite_ref(first_hotspot_id)),
    )

    assert repository.get_evidence_refs(saved.id) == (
        news_ref(report_id),
        satellite_ref(first_hotspot_id),
        satellite_ref(second_hotspot_id),
    )


def test_empty_supporting_evidence_rejected(repository):
    with pytest.raises(FireEventRepositoryError):
        repository.create_event(make_event(), ())


def test_invalid_fk_evidence_causes_rollback(repository, sqlite_session_factory):
    with pytest.raises(FireEventRepositoryError):
        repository.create_event(make_event(), (satellite_ref(999),))

    session = sqlite_session_factory()
    event_count = len(session.execute(select(FireEventDB)).scalars().all())
    session.close()
    assert event_count == 0


def test_get_evidence_refs_returns_correct_satellite_refs(repository, satellite_repository):
    hotspot_id = persist_satellite(satellite_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    assert repository.get_evidence_refs(saved.id) == (satellite_ref(hotspot_id),)


def test_get_evidence_refs_returns_correct_news_refs(repository, news_repository):
    report_id = persist_news(news_repository)
    saved = repository.create_event(make_event(), (news_ref(report_id),))

    assert repository.get_evidence_refs(saved.id) == (news_ref(report_id),)


def test_attach_duplicate_same_satellite_is_idempotent(repository, satellite_repository, sqlite_session_factory):
    hotspot_id = persist_satellite(satellite_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    refs = repository.attach_evidence(saved.id, (satellite_ref(hotspot_id),))

    session = sqlite_session_factory()
    rows = session.execute(select(FireEventSatelliteEvidenceDB)).scalars().all()
    session.close()
    assert refs == (satellite_ref(hotspot_id),)
    assert len(rows) == 1


def test_attach_duplicate_same_news_is_idempotent(repository, news_repository, sqlite_session_factory):
    report_id = persist_news(news_repository)
    saved = repository.create_event(make_event(), (news_ref(report_id),))

    refs = repository.attach_evidence(saved.id, (news_ref(report_id),))

    session = sqlite_session_factory()
    rows = session.execute(select(FireEventNewsEvidenceDB)).scalars().all()
    session.close()
    assert refs == (news_ref(report_id),)
    assert len(rows) == 1


def test_attach_same_numeric_id_across_source_types_is_allowed(repository, satellite_repository, news_repository):
    hotspot_id = persist_satellite(satellite_repository)
    report_id = persist_news(news_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    refs = repository.attach_evidence(saved.id, (news_ref(report_id),))

    assert refs == (news_ref(report_id), satellite_ref(hotspot_id))


def test_attaching_new_evidence_preserves_existing_associations(repository, satellite_repository, news_repository):
    hotspot_id = persist_satellite(satellite_repository)
    report_id = persist_news(news_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    refs = repository.attach_evidence(saved.id, (news_ref(report_id),))

    assert refs == (news_ref(report_id), satellite_ref(hotspot_id))


def test_nearby_suspected_event_within_six_hours_matches(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT + timedelta(hours=5))

    assert found.id == saved.id


def test_nearby_confirmed_event_within_six_hours_matches(repository, satellite_repository):
    saved = create_event_with_satellite(
        repository,
        satellite_repository,
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
    )

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT + timedelta(hours=5))

    assert found.id == saved.id


@pytest.mark.parametrize("inactive_status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_event_does_not_match(repository, satellite_repository, inactive_status):
    saved = create_event_with_satellite(repository, satellite_repository)
    repository.update_event(saved.id, make_event(status=inactive_status, updated_at=UPDATED_AT + timedelta(minutes=1)))

    assert repository.find_matching_active_event(32.731, 35.046, UPDATED_AT + timedelta(minutes=2)) is None


def test_exact_distance_boundary_matches(repository, satellite_repository):
    lat, lon = coordinate_north_at_distance(ACTIVE_EVENT_MATCH_DISTANCE_KM)
    saved = create_event_with_satellite(repository, satellite_repository)

    found = repository.find_matching_active_event(lat, lon, UPDATED_AT)

    assert found.id == saved.id


def test_beyond_distance_boundary_does_not_match(repository, satellite_repository):
    lat, lon = coordinate_north_at_distance(ACTIVE_EVENT_MATCH_DISTANCE_KM + 0.01)
    create_event_with_satellite(repository, satellite_repository)

    assert repository.find_matching_active_event(lat, lon, UPDATED_AT) is None


def test_exact_six_hour_boundary_matches(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT + timedelta(hours=6))

    assert found.id == saved.id


def test_more_than_six_hours_does_not_match(repository, satellite_repository):
    create_event_with_satellite(repository, satellite_repository)

    assert repository.find_matching_active_event(32.731, 35.046, UPDATED_AT + timedelta(hours=6, seconds=1)) is None


def test_nearby_but_stale_event_does_not_match(repository, satellite_repository):
    create_event_with_satellite(
        repository,
        satellite_repository,
        detected_at=DETECTED_AT - timedelta(hours=8),
        updated_at=UPDATED_AT - timedelta(hours=7),
    )

    assert repository.find_matching_active_event(32.731, 35.046, UPDATED_AT) is None


def test_recent_but_distant_event_does_not_match(repository, satellite_repository):
    create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)

    assert repository.find_matching_active_event(32.731, 35.046, UPDATED_AT) is None


def test_get_active_events_near_returns_active_events_within_radius(repository, satellite_repository):
    nearby = create_event_with_satellite(repository, satellite_repository, latitude=32.732)
    create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)

    found = repository.get_active_events_near(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        as_of=UPDATED_AT,
    )

    assert tuple(event.id for event in found) == (nearby.id,)


@pytest.mark.parametrize("inactive_status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_get_active_events_near_excludes_inactive_events(repository, satellite_repository, inactive_status):
    saved = create_event_with_satellite(repository, satellite_repository)
    repository.update_event(saved.id, make_event(status=inactive_status, updated_at=UPDATED_AT + timedelta(minutes=1)))

    found = repository.get_active_events_near(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        as_of=UPDATED_AT + timedelta(minutes=2),
    )

    assert found == ()


def test_get_active_events_near_excludes_future_detected_events(repository, satellite_repository):
    create_event_with_satellite(
        repository,
        satellite_repository,
        detected_at=UPDATED_AT + timedelta(hours=1),
        updated_at=UPDATED_AT + timedelta(hours=1),
    )

    found = repository.get_active_events_near(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        as_of=UPDATED_AT,
    )

    assert found == ()


def test_get_active_events_near_orders_by_distance_then_id(repository, satellite_repository):
    farther = create_event_with_satellite(repository, satellite_repository, latitude=32.740)
    closer = create_event_with_satellite(repository, satellite_repository, latitude=32.732)

    found = repository.get_active_events_near(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        as_of=UPDATED_AT + timedelta(minutes=1),
    )

    assert tuple(event.id for event in found) == (closer.id, farther.id)


def test_get_active_events_near_exact_distance_boundary_matches(repository, satellite_repository):
    lat, lon = coordinate_north_at_distance(5)
    saved = create_event_with_satellite(repository, satellite_repository, latitude=lat, longitude=lon)

    found = repository.get_active_events_near(
        latitude=32.731,
        longitude=35.046,
        radius_km=5,
        as_of=UPDATED_AT,
    )

    assert tuple(event.id for event in found) == (saved.id,)


@pytest.mark.parametrize("invalid_radius", [0, -1, True, float("inf"), "5"])
def test_get_active_events_near_rejects_invalid_radius(repository, invalid_radius):
    with pytest.raises(FireEventRepositoryError):
        repository.get_active_events_near(
            latitude=32.731,
            longitude=35.046,
            radius_km=invalid_radius,
            as_of=UPDATED_AT,
        )


def test_get_active_fire_event_ids_returns_active_events_regardless_of_location(repository, satellite_repository):
    nearby = create_event_with_satellite(repository, satellite_repository, latitude=32.732)
    distant = create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)

    found = repository.get_active_fire_event_ids()

    assert set(found) == {nearby.id, distant.id}


@pytest.mark.parametrize("inactive_status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_get_active_fire_event_ids_excludes_inactive_events(repository, satellite_repository, inactive_status):
    active = create_event_with_satellite(repository, satellite_repository)
    inactive = create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)
    repository.update_event(
        inactive.id, make_event(status=inactive_status, latitude=33.5, longitude=35.5, updated_at=UPDATED_AT)
    )

    found = repository.get_active_fire_event_ids()

    assert found == (active.id,)


def test_get_active_fire_event_ids_returns_empty_tuple_when_none_active(repository):
    assert repository.get_active_fire_event_ids() == ()


def test_get_active_fire_event_ids_is_ordered_by_id(repository, satellite_repository):
    first = create_event_with_satellite(repository, satellite_repository, latitude=32.732)
    second = create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)

    found = repository.get_active_fire_event_ids()

    assert found == tuple(sorted((first.id, second.id)))


@pytest.mark.parametrize("active_status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_get_active_events_includes_suspected_and_confirmed(repository, satellite_repository, active_status):
    created = create_event_with_satellite(repository, satellite_repository, status=active_status)

    found = repository.get_active_events()

    assert [stored.id for stored in found] == [created.id]
    assert found[0].event.status is active_status


@pytest.mark.parametrize("inactive_status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_get_active_events_excludes_resolved_and_dismissed(repository, satellite_repository, inactive_status):
    active = create_event_with_satellite(repository, satellite_repository)
    inactive = create_event_with_satellite(repository, satellite_repository, latitude=33.5, longitude=35.5)
    repository.update_event(
        inactive.id, make_event(status=inactive_status, latitude=33.5, longitude=35.5, updated_at=UPDATED_AT)
    )

    found = repository.get_active_events()

    assert [stored.id for stored in found] == [active.id]


def test_get_active_events_returns_empty_tuple_when_none_active(repository):
    assert repository.get_active_events() == ()


def test_get_active_events_orders_most_recently_updated_first(repository, satellite_repository):
    older = create_event_with_satellite(repository, satellite_repository, updated_at=UPDATED_AT)
    newer = create_event_with_satellite(
        repository, satellite_repository, latitude=33.5, longitude=35.5, updated_at=UPDATED_AT + timedelta(hours=1)
    )

    found = repository.get_active_events()

    assert [stored.id for stored in found] == [newer.id, older.id]


def test_get_active_events_tie_break_uses_id_when_updated_at_matches(repository, satellite_repository):
    first = create_event_with_satellite(repository, satellite_repository, updated_at=UPDATED_AT)
    second = create_event_with_satellite(
        repository, satellite_repository, latitude=33.5, longitude=35.5, updated_at=UPDATED_AT
    )

    found = repository.get_active_events()

    assert [stored.id for stored in found] == sorted([first.id, second.id], reverse=True)


def test_multiple_matching_events_choose_closest(repository, satellite_repository):
    farther = create_event_with_satellite(repository, satellite_repository, latitude=32.740, detected_at=DETECTED_AT)
    closer = create_event_with_satellite(
        repository,
        satellite_repository,
        latitude=32.732,
        detected_at=DETECTED_AT + timedelta(minutes=1),
        updated_at=UPDATED_AT + timedelta(minutes=1),
    )

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT)

    assert found.id == closer.id
    assert found.id != farther.id


def test_equal_distance_chooses_closest_time(repository, satellite_repository):
    older = create_event_with_satellite(
        repository,
        satellite_repository,
        detected_at=DETECTED_AT - timedelta(hours=3),
        updated_at=UPDATED_AT - timedelta(hours=2),
    )
    newer = create_event_with_satellite(
        repository,
        satellite_repository,
        detected_at=DETECTED_AT - timedelta(minutes=1),
        updated_at=UPDATED_AT - timedelta(minutes=10),
    )

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT)

    assert found.id == newer.id
    assert found.id != older.id


def test_stable_final_tie_break_uses_smallest_id(repository, satellite_repository):
    first = create_event_with_satellite(repository, satellite_repository)
    second = create_event_with_satellite(
        repository,
        satellite_repository,
        detected_at=DETECTED_AT + timedelta(minutes=1),
        updated_at=UPDATED_AT,
    )

    found = repository.find_matching_active_event(32.731, 35.046, UPDATED_AT)

    assert found.id == first.id
    assert found.id != second.id


def test_update_confidence(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)

    updated = repository.update_event(saved.id, make_event(detection_confidence=0.9))

    assert updated.event.detection_confidence == pytest.approx(0.9)


def test_update_suspected_to_confirmed(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)

    updated = repository.update_event(
        saved.id,
        make_event(status=FireEventStatus.CONFIRMED, detection_confidence=0.85),
    )

    assert updated.event.status is FireEventStatus.CONFIRMED


def test_update_location(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)

    updated = repository.update_event(saved.id, make_event(latitude=32.8, longitude=35.1))

    assert updated.event.latitude == pytest.approx(32.8)
    assert updated.event.longitude == pytest.approx(35.1)


def test_update_updated_at(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)
    new_updated_at = UPDATED_AT + timedelta(minutes=30)

    updated = repository.update_event(saved.id, make_event(updated_at=new_updated_at))

    assert updated.event.updated_at == new_updated_at


def test_evidence_associations_remain_intact_after_update(repository, satellite_repository, news_repository):
    hotspot_id = persist_satellite(satellite_repository)
    report_id = persist_news(news_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id), news_ref(report_id)))

    updated = repository.update_event(saved.id, make_event(detection_confidence=0.7))

    assert updated.supporting_evidence == (news_ref(report_id), satellite_ref(hotspot_id))


def test_update_does_not_duplicate_evidence(repository, satellite_repository, sqlite_session_factory):
    hotspot_id = persist_satellite(satellite_repository)
    saved = repository.create_event(make_event(), (satellite_ref(hotspot_id),))

    repository.update_event(saved.id, make_event(detection_confidence=0.7))

    session = sqlite_session_factory()
    rows = session.execute(select(FireEventSatelliteEvidenceDB)).scalars().all()
    session.close()
    assert len(rows) == 1


def test_update_preserves_methodology(repository, satellite_repository):
    saved = create_event_with_satellite(repository, satellite_repository)
    repository.update_event(
        saved.id,
        make_event(methodology="OTHER", methodology_version="2.0", detection_confidence=0.7),
    )

    found = repository.get_by_id(saved.id)

    assert found.event.methodology == FIRE_DETECTION_METHODOLOGY_NAME
    assert found.event.methodology_version == FIRE_DETECTION_METHODOLOGY_VERSION
