"""Live integration test for satellite hotspot persistence against Neon.

Run explicitly with:

    python -m pytest tests/integration/test_satellite_hotspot_neon.py -m integration -v

The test creates and deletes only clearly identifiable temporary satellite
hotspot rows and never prints DATABASE_URL or credentials.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import text

from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

pytestmark = pytest.mark.integration

TEST_LATITUDE = 31.123456
TEST_LONGITUDE = 35.654321
TEST_DETECTED_AT = datetime(2026, 1, 2, 3, 4, 0)
TEST_SATELLITE = "ECOGUARD_INTEGRATION_TEST_SAT"
TEST_INSTRUMENT = "ECOGUARD_INTEGRATION_TEST"


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM satellite_hotspots "
                "WHERE instrument = :instrument "
                "OR satellite = :satellite "
                "OR (latitude = :latitude AND longitude = :longitude AND detected_at = :detected_at)"
            ),
            {
                "instrument": TEST_INSTRUMENT,
                "satellite": TEST_SATELLITE,
                "latitude": TEST_LATITUDE,
                "longitude": TEST_LONGITUDE,
                "detected_at": TEST_DETECTED_AT,
            },
        )


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def make_hotspot(**overrides) -> SatelliteHotspot:
    defaults = dict(
        latitude=TEST_LATITUDE,
        longitude=TEST_LONGITUDE,
        detected_at=TEST_DETECTED_AT,
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite=TEST_SATELLITE,
        instrument=TEST_INSTRUMENT,
        day_night="D",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


def _count_test_rows() -> int:
    with get_session() as session:
        return session.execute(
            text(
                "SELECT COUNT(*) FROM satellite_hotspots "
                "WHERE instrument = :instrument "
                "OR satellite = :satellite "
                "OR (latitude = :latitude AND longitude = :longitude AND detected_at = :detected_at)"
            ),
            {
                "instrument": TEST_INSTRUMENT,
                "satellite": TEST_SATELLITE,
                "latitude": TEST_LATITUDE,
                "longitude": TEST_LONGITUDE,
                "detected_at": TEST_DETECTED_AT,
            },
        ).scalar_one()


def _find_test_hotspot(repository: SatelliteHotspotRepository) -> SatelliteHotspot | None:
    hotspots = repository.get_hotspots_between(TEST_DETECTED_AT, TEST_DETECTED_AT)
    for hotspot in hotspots:
        if (
            hotspot.latitude == TEST_LATITUDE
            and hotspot.longitude == TEST_LONGITUDE
            and hotspot.detected_at == TEST_DETECTED_AT
            and hotspot.satellite == TEST_SATELLITE
            and hotspot.instrument == TEST_INSTRUMENT
        ):
            return hotspot
    return None


def test_satellite_hotspot_table_save_read_duplicate_and_cleanup_against_neon() -> None:
    init_db()
    repository = SatelliteHotspotRepository()

    with get_session() as session:
        table_exists = session.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'satellite_hotspots'"
                ")"
            )
        ).scalar_one()
    assert table_exists is True

    first = repository.save_hotspot(make_hotspot())
    assert first.is_duplicate is False

    stored = _find_test_hotspot(repository)
    assert stored is not None
    assert stored.latitude == TEST_LATITUDE
    assert stored.longitude == TEST_LONGITUDE
    assert stored.detected_at == TEST_DETECTED_AT
    assert stored.satellite == TEST_SATELLITE
    assert stored.instrument == TEST_INSTRUMENT

    second = repository.save_hotspot(make_hotspot(frp=99.9))
    assert second.is_duplicate is True
    assert second.hotspot.frp == 12.4
    assert _count_test_rows() == 1

    none_satellite = make_hotspot(
        detected_at=datetime(2026, 1, 2, 3, 5, 0),
        satellite=None,
        instrument=TEST_INSTRUMENT,
    )
    first_none = repository.save_hotspot(none_satellite)
    second_none = repository.save_hotspot(none_satellite)
    assert first_none.is_duplicate is False
    assert second_none.is_duplicate is True

    _delete_test_rows()
    assert _count_test_rows() == 0
