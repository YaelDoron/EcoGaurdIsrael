"""Unit tests for SatelliteHotspotRepository using SQLite in-memory."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.exceptions import SatelliteHotspotRepositoryError
from src.repositories.satellite_hotspot_repository import SaveHotspotResult, SatelliteHotspotRepository


@pytest.fixture
def repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


def make_hotspot(**overrides) -> SatelliteHotspot:
    defaults = dict(
        latitude=32.7001,
        longitude=35.0123,
        detected_at=datetime(2026, 9, 5, 14, 32, 0, tzinfo=timezone.utc),
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite="N20",
        instrument="VIIRS",
        day_night="D",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


def count_hotspots(sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    count = len(session.execute(select(SatelliteHotspotDB)).scalars().all())
    session.close()
    return count


def test_new_hotspot_saves_successfully(repository):
    result = repository.save_hotspot(make_hotspot())

    assert isinstance(result, SaveHotspotResult)
    assert result.hotspot.latitude == 32.7001


def test_new_hotspot_save_result_is_not_duplicate(repository):
    result = repository.save_hotspot(make_hotspot())

    assert result.is_duplicate is False


# ---------------------------------------------------------------------------
# Timezone fix regression tests: detected_at is now DateTime(timezone=True)
# (was plain DateTime - the root cause of the ~3h Activity Feed display
# mismatch). These prove the round-trip preserves the actual instant and
# that a naive input is normalized rather than silently misinterpreted.
# ---------------------------------------------------------------------------


def test_detected_at_column_is_declared_timezone_aware():
    assert SatelliteHotspotDB.__table__.columns["detected_at"].type.timezone is True


def test_aware_detected_at_round_trips_preserving_the_exact_instant(repository):
    aware_instant = datetime(2026, 9, 5, 10, 12, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=aware_instant))

    stored = repository.get_latest_hotspots()[0]

    assert stored.detected_at.tzinfo is not None
    assert stored.detected_at == aware_instant


def test_naive_detected_at_is_normalized_to_utc_rather_than_silently_misinterpreted(repository):
    naive_value = datetime(2026, 9, 5, 10, 12, 0)
    repository.save_hotspot(make_hotspot(detected_at=naive_value))

    stored = repository.get_latest_hotspots()[0]

    assert stored.detected_at.tzinfo is not None
    assert stored.detected_at == naive_value.replace(tzinfo=timezone.utc)


def test_read_back_values_match_domain_object(repository):
    hotspot = make_hotspot()
    repository.save_hotspot(hotspot)

    latest = repository.get_latest_hotspots()

    assert latest == [hotspot]


def test_optional_fields_none_are_persisted_correctly(repository):
    hotspot = make_hotspot(
        confidence=None,
        frp=None,
        brightness=None,
        satellite=None,
        instrument=None,
        day_night=None,
    )

    result = repository.save_hotspot(hotspot)

    assert result.hotspot.confidence is None
    assert result.hotspot.frp is None
    assert result.hotspot.brightness is None
    assert result.hotspot.satellite is None
    assert result.hotspot.instrument is None
    assert result.hotspot.day_night is None


def test_same_hotspot_twice_creates_one_row_and_duplicate_result(repository, sqlite_session_factory):
    first = repository.save_hotspot(make_hotspot(frp=12.4))
    second = repository.save_hotspot(make_hotspot(frp=99.9))

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert second.hotspot.frp == 12.4
    assert count_hotspots(sqlite_session_factory) == 1


def test_duplicate_with_satellite_none_is_prevented(repository, sqlite_session_factory):
    first = repository.save_hotspot(make_hotspot(satellite=None, frp=1.0))
    second = repository.save_hotspot(make_hotspot(satellite=None, frp=2.0))

    assert first.is_duplicate is False
    assert second.is_duplicate is True
    assert second.hotspot.frp == 1.0
    assert count_hotspots(sqlite_session_factory) == 1


def test_same_location_time_different_satellite_allows_two_rows(repository, sqlite_session_factory):
    repository.save_hotspot(make_hotspot(satellite="N20"))
    repository.save_hotspot(make_hotspot(satellite="N21"))

    assert count_hotspots(sqlite_session_factory) == 2


def test_same_location_different_time_allows_two_rows(repository, sqlite_session_factory):
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 32, 0)))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 33, 0)))

    assert count_hotspots(sqlite_session_factory) == 2


def test_different_location_same_time_allows_two_rows(repository, sqlite_session_factory):
    repository.save_hotspot(make_hotspot(latitude=32.7001))
    repository.save_hotspot(make_hotspot(latitude=32.7002))

    assert count_hotspots(sqlite_session_factory) == 2


def test_same_hotspot_values_generate_same_detection_key(repository):
    first_key = repository._generate_detection_key(make_hotspot())
    second_key = repository._generate_detection_key(make_hotspot())

    assert first_key == second_key


def test_different_detected_at_generates_different_key(repository):
    first_key = repository._generate_detection_key(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 32)))
    second_key = repository._generate_detection_key(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 33)))

    assert first_key != second_key


def test_different_latitude_generates_different_key(repository):
    first_key = repository._generate_detection_key(make_hotspot(latitude=32.7001))
    second_key = repository._generate_detection_key(make_hotspot(latitude=32.7002))

    assert first_key != second_key


def test_different_longitude_generates_different_key(repository):
    first_key = repository._generate_detection_key(make_hotspot(longitude=35.0123))
    second_key = repository._generate_detection_key(make_hotspot(longitude=35.0124))

    assert first_key != second_key


def test_different_satellite_generates_different_key(repository):
    first_key = repository._generate_detection_key(make_hotspot(satellite="N20"))
    second_key = repository._generate_detection_key(make_hotspot(satellite="N21"))

    assert first_key != second_key


def test_satellite_none_detection_key_is_deterministic(repository):
    first_key = repository._generate_detection_key(make_hotspot(satellite=None))
    second_key = repository._generate_detection_key(make_hotspot(satellite=None))

    assert first_key == second_key


def test_detection_key_is_sha256_hex_and_not_python_hash(repository):
    detection_key = repository._generate_detection_key(make_hotspot())

    assert len(detection_key) == 64
    assert all(character in "0123456789abcdef" for character in detection_key)
    assert detection_key != str(hash(make_hotspot().detected_at.isoformat()))


def test_get_latest_hotspots_returns_newest_first(repository):
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 12, 0), frp=1.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 0), frp=3.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 13, 0), frp=2.0))

    latest = repository.get_latest_hotspots()

    assert [hotspot.frp for hotspot in latest] == [3.0, 2.0, 1.0]


def test_get_latest_hotspots_respects_limit(repository):
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 12, 0), frp=1.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 13, 0), frp=2.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 14, 0), frp=3.0))

    latest_two = repository.get_latest_hotspots(limit=2)

    assert [hotspot.frp for hotspot in latest_two] == [3.0, 2.0]


def test_get_latest_hotspots_empty_table_returns_empty_list(repository):
    assert repository.get_latest_hotspots() == []


@pytest.mark.parametrize("limit", [0, -1, None, "10", True])
def test_get_latest_hotspots_invalid_limit_fails(repository, limit):
    with pytest.raises(SatelliteHotspotRepositoryError):
        repository.get_latest_hotspots(limit=limit)


def test_get_hotspots_between_filters_range(repository):
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 11, 0), frp=1.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 12, 0), frp=2.0))
    repository.save_hotspot(make_hotspot(detected_at=datetime(2026, 9, 5, 13, 0), frp=3.0))

    results = repository.get_hotspots_between(
        datetime(2026, 9, 5, 11, 30),
        datetime(2026, 9, 5, 13, 0),
    )

    assert [hotspot.frp for hotspot in results] == [3.0, 2.0]


def test_get_hotspots_between_validates_start_before_end(repository):
    with pytest.raises(SatelliteHotspotRepositoryError):
        repository.get_hotspots_between(datetime(2026, 9, 5, 13, 0), datetime(2026, 9, 5, 12, 0))


def test_get_hotspots_between_validates_datetime_arguments(repository):
    with pytest.raises(SatelliteHotspotRepositoryError):
        repository.get_hotspots_between("2026-09-05", datetime(2026, 9, 5, 12, 0))


def test_get_recent_hotspots_returns_recent_persisted_ids(repository):
    as_of = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=10), frp=10.0))

    results = repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)

    assert len(results) == 1
    assert results[0].id > 0
    assert results[0].hotspot.frp == 10.0


def test_get_recent_hotspots_includes_exact_lookback_boundary(repository):
    as_of = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=120), frp=10.0))

    results = repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)

    assert [result.hotspot.frp for result in results] == [10.0]


def test_get_recent_hotspots_excludes_older_and_future_rows(repository):
    as_of = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=121), frp=1.0))
    repository.save_hotspot(make_hotspot(detected_at=as_of + timedelta(minutes=1), frp=2.0))
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=1), frp=3.0))

    results = repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)

    assert [result.hotspot.frp for result in results] == [3.0]


def test_get_recent_hotspots_ordering_is_deterministic(repository):
    as_of = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=30), frp=1.0))
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=10), frp=2.0))
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=20), frp=3.0))

    results = repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)

    assert [result.hotspot.frp for result in results] == [2.0, 3.0, 1.0]


def test_created_at_is_a_real_persisted_timestamp_never_none_for_a_fresh_row(repository):
    """Task 4, Part D: created_at is set ON INSERT, distinct from and never
    derived from detected_at."""
    old_detected_at = datetime(2026, 9, 5, 14, 32, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=old_detected_at))

    stored = repository.get_recent(1)[0]

    assert stored.created_at is not None
    assert stored.created_at.tzinfo is not None
    assert stored.created_at != old_detected_at
    assert stored.created_at > old_detected_at


def test_get_recent_hotspots_and_get_by_id_also_expose_created_at(repository):
    as_of = datetime(2026, 9, 5, 14, 0, tzinfo=timezone.utc)
    repository.save_hotspot(make_hotspot(detected_at=as_of - timedelta(minutes=10), frp=10.0))
    hotspot_id = repository.get_recent(1)[0].id

    from_recent_hotspots = repository.get_recent_hotspots(as_of=as_of, lookback_minutes=120)[0]
    from_get_by_id = repository.get_by_id(hotspot_id)

    assert from_recent_hotspots.created_at is not None
    assert from_get_by_id is not None
    assert from_get_by_id.created_at is not None
    assert from_recent_hotspots.created_at == from_get_by_id.created_at


def test_integrity_error_race_fallback_returns_duplicate_result(repository, monkeypatch, sqlite_session_factory):
    repository.save_hotspot(make_hotspot(frp=11.0))

    original_find = SatelliteHotspotRepository._find_by_detection_key
    call_count = {"n": 0}

    def fake_find_by_detection_key(session, detection_key):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None
        return original_find(session, detection_key)

    monkeypatch.setattr(
        SatelliteHotspotRepository,
        "_find_by_detection_key",
        staticmethod(fake_find_by_detection_key),
    )

    result = repository.save_hotspot(make_hotspot(frp=99.0))

    assert result.is_duplicate is True
    assert result.hotspot.frp == 11.0
    assert count_hotspots(sqlite_session_factory) == 1


def test_unrecoverable_integrity_error_is_converted(repository, monkeypatch):
    repository.save_hotspot(make_hotspot())

    def fake_find_by_detection_key(session, detection_key):
        return None

    monkeypatch.setattr(
        SatelliteHotspotRepository,
        "_find_by_detection_key",
        staticmethod(fake_find_by_detection_key),
    )

    with pytest.raises(SatelliteHotspotRepositoryError):
        repository.save_hotspot(make_hotspot())


def test_repository_session_is_usable_after_rolled_back_write(repository, sqlite_session_factory):
    raw_session = sqlite_session_factory()
    raw_session.add(
        SatelliteHotspotDB(
            detection_key=None,
            latitude=32.7,
            longitude=35.0,
            detected_at=datetime(2026, 9, 5, 12, 0),
        )
    )
    with pytest.raises(IntegrityError):
        raw_session.commit()
    raw_session.rollback()
    raw_session.close()

    result = repository.save_hotspot(make_hotspot())

    assert result.is_duplicate is False
