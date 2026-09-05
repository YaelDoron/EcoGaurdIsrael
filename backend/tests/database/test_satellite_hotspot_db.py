"""Tests for the SatelliteHotspotDB SQLAlchemy model."""
from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from src.database.models.satellite_hotspot_db import SatelliteHotspotDB

DETECTED_AT = datetime(2026, 9, 5, 14, 32, 0)


def make_db_hotspot(**overrides) -> SatelliteHotspotDB:
    defaults = dict(
        detection_key="test-detection-key",
        latitude=32.7001,
        longitude=35.0123,
        detected_at=DETECTED_AT,
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite="N20",
        instrument="VIIRS",
        day_night="D",
    )
    defaults.update(overrides)
    return SatelliteHotspotDB(**defaults)


def test_table_name_is_satellite_hotspots():
    assert SatelliteHotspotDB.__tablename__ == "satellite_hotspots"


def test_id_is_primary_key():
    primary_keys = {column.name for column in inspect(SatelliteHotspotDB).primary_key}

    assert primary_keys == {"id"}


@pytest.mark.parametrize("field_name", ["latitude", "longitude", "detected_at", "detection_key"])
def test_required_columns_are_non_nullable(field_name):
    column = SatelliteHotspotDB.__table__.columns[field_name]

    assert column.nullable is False


@pytest.mark.parametrize(
    "field_name",
    ["confidence", "frp", "brightness", "satellite", "instrument", "day_night"],
)
def test_optional_columns_are_nullable(field_name):
    column = SatelliteHotspotDB.__table__.columns[field_name]

    assert column.nullable is True


def test_detection_key_is_unique():
    assert SatelliteHotspotDB.__table__.columns["detection_key"].unique is True


def test_detection_key_is_indexed():
    assert SatelliteHotspotDB.__table__.columns["detection_key"].index is True


def test_detection_key_unique_constraint_is_enforced(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(make_db_hotspot(detection_key="same-key"))
    session.commit()

    session.add(make_db_hotspot(detection_key="same-key"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


@pytest.mark.parametrize("field_name", ["latitude", "longitude", "detected_at", "detection_key"])
def test_required_column_constraints_are_enforced(sqlite_session_factory, field_name):
    session = sqlite_session_factory()
    session.add(make_db_hotspot(**{field_name: None}))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_optional_fields_can_be_null(sqlite_session_factory):
    session = sqlite_session_factory()
    db_hotspot = make_db_hotspot(
        confidence=None,
        frp=None,
        brightness=None,
        satellite=None,
        instrument=None,
        day_night=None,
    )
    session.add(db_hotspot)
    session.commit()

    assert db_hotspot.confidence is None
    assert db_hotspot.frp is None
    assert db_hotspot.brightness is None
    assert db_hotspot.satellite is None
    assert db_hotspot.instrument is None
    assert db_hotspot.day_night is None
    session.close()
