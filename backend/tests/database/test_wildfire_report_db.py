"""Tests for the WildfireReportDB SQLAlchemy model."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import DateTime
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from src.database.models.wildfire_report_db import WildfireReportDB

PUBLISHED_AT = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
FETCHED_AT = datetime(2026, 9, 12, 10, 5, tzinfo=timezone.utc)


def make_db_report(**overrides) -> WildfireReportDB:
    defaults = dict(
        source_url="https://example.com/wildfire/1",
        source_feed="Example Feed",
        title="Wildfire reported near Haifa",
        summary="A short summary",
        location_name="Haifa",
        latitude=32.794,
        longitude=34.9896,
        published_at=PUBLISHED_AT,
        fetched_at=FETCHED_AT,
    )
    defaults.update(overrides)
    return WildfireReportDB(**defaults)


def test_table_name_is_wildfire_reports():
    assert WildfireReportDB.__tablename__ == "wildfire_reports"


def test_id_is_primary_key():
    primary_keys = {column.name for column in inspect(WildfireReportDB).primary_key}

    assert primary_keys == {"id"}


@pytest.mark.parametrize("field_name", ["source_url", "title", "fetched_at"])
def test_required_columns_are_non_nullable(field_name):
    column = WildfireReportDB.__table__.columns[field_name]

    assert column.nullable is False


@pytest.mark.parametrize(
    "field_name",
    ["source_feed", "summary", "location_name", "latitude", "longitude", "published_at"],
)
def test_optional_columns_are_nullable(field_name):
    column = WildfireReportDB.__table__.columns[field_name]

    assert column.nullable is True


def test_source_url_is_unique_and_indexed():
    column = WildfireReportDB.__table__.columns["source_url"]

    assert column.unique is True
    assert column.index is True


@pytest.mark.parametrize("field_name", ["published_at", "fetched_at"])
def test_timestamp_columns_are_timezone_aware_datetime(field_name):
    column_type = WildfireReportDB.__table__.columns[field_name].type

    assert isinstance(column_type, DateTime)
    assert column_type.timezone is True


def test_valid_wildfire_report_row_can_be_created(sqlite_session_factory):
    session = sqlite_session_factory()
    db_report = make_db_report()
    session.add(db_report)
    session.commit()

    assert db_report.id is not None
    assert db_report.source_url == "https://example.com/wildfire/1"
    assert db_report.published_at == PUBLISHED_AT
    assert db_report.fetched_at == FETCHED_AT
    session.close()


def test_source_url_unique_constraint_is_enforced(sqlite_session_factory):
    session = sqlite_session_factory()
    session.add(make_db_report(source_url="https://example.com/same"))
    session.commit()

    session.add(make_db_report(source_url="https://example.com/same"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


@pytest.mark.parametrize("field_name", ["source_url", "title", "fetched_at"])
def test_required_column_constraints_are_enforced(sqlite_session_factory, field_name):
    session = sqlite_session_factory()
    session.add(make_db_report(**{field_name: None}))

    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_optional_fields_can_be_null(sqlite_session_factory):
    session = sqlite_session_factory()
    db_report = make_db_report(
        source_feed=None,
        summary=None,
        location_name=None,
        latitude=None,
        longitude=None,
        published_at=None,
    )
    session.add(db_report)
    session.commit()

    assert db_report.source_feed is None
    assert db_report.summary is None
    assert db_report.location_name is None
    assert db_report.latitude is None
    assert db_report.longitude is None
    assert db_report.published_at is None
    session.close()
