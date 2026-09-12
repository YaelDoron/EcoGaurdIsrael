"""Live integration tests for SimulationEventExecutor persistence wiring."""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

import pytest
from sqlalchemy import text

from scripts.run_demo_simulation import execute_and_report_event
from src.config.settings import settings
from src.database.connection import get_engine, get_session, init_db
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.simulation import (
    GOLAN_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutor,
    SimulationEventType,
    SimulationLocation,
    SimulationScenario,
    simulation_event_timestamp,
)
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator

pytestmark = pytest.mark.integration

TEST_SEED = 505
TEST_STARTED_AT = datetime(2026, 9, 12, 15, 55, 0, tzinfo=timezone.utc)
TEST_LOCATION = SimulationLocation(
    name="Executor Integration Area",
    latitude=30.123456,
    longitude=35.234567,
)
TEST_SECOND_LOCATION = SimulationLocation(
    name="Executor Integration Second Area",
    latitude=30.923456,
    longitude=35.934567,
)
TEST_INCIDENT = SimulatedIncident(
    incident_id="incident-executor-integration-01",
    scenario_type=ScenarioType.ACTIVE_FIRE,
    location=TEST_LOCATION,
)
TEST_SECOND_INCIDENT = SimulatedIncident(
    incident_id="incident-executor-integration-02",
    scenario_type=ScenarioType.ACTIVE_FIRE,
    location=TEST_SECOND_LOCATION,
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


def make_scenario(events, incidents=(TEST_INCIDENT,)) -> SimulationScenario:
    return SimulationScenario(
        duration_seconds=120,
        seed=TEST_SEED,
        incidents=incidents,
        events=events,
    )


def make_event(
    event_type: SimulationEventType,
    offset_seconds: int = 0,
    incident_id: str = TEST_INCIDENT.incident_id,
    source_event_index: int = 0,
) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=offset_seconds,
        event_type=event_type,
        incident_id=incident_id,
        source_event_index=source_event_index,
    )


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        _delete_expected_weather_rows(connection)
        _delete_expected_satellite_rows(connection)
        _delete_expected_news_rows(connection)


def _delete_expected_weather_rows(connection) -> None:  # noqa: ANN001
    for location in (TEST_LOCATION, TEST_SECOND_LOCATION, GOLAN_LOCATION):
        generated = WeatherDataGenerator(seed=TEST_SEED).generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            timestamp=TEST_STARTED_AT,
            location=location,
        )
        for measurement in generated.measurements:
            connection.execute(
                text(
                    "DELETE FROM weather_observations "
                    "WHERE station_id IN ("
                    "SELECT id FROM weather_stations WHERE external_station_id = :external_station_id"
                    ")"
                ),
                {"external_station_id": measurement.station.external_station_id},
            )
            connection.execute(
                text("DELETE FROM weather_stations WHERE external_station_id = :external_station_id"),
                {"external_station_id": measurement.station.external_station_id},
            )


def _delete_expected_satellite_rows(connection) -> None:  # noqa: ANN001
    for location, timestamp in (
        (TEST_LOCATION, TEST_STARTED_AT),
        (TEST_LOCATION, TEST_STARTED_AT.replace(second=20)),
        (TEST_SECOND_LOCATION, TEST_STARTED_AT.replace(second=35)),
    ):
        generated = SatelliteDataGenerator(seed=TEST_SEED).generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            timestamp=timestamp,
            location=location,
        )
        for hotspot in generated.hotspots:
            connection.execute(
                text(
                    "DELETE FROM satellite_hotspots "
                    "WHERE latitude = :latitude "
                    "AND longitude = :longitude "
                    "AND detected_at = :detected_at "
                    "AND satellite = :satellite "
                    "AND instrument = :instrument"
                ),
                {
                    "latitude": hotspot.latitude,
                    "longitude": hotspot.longitude,
                    "detected_at": hotspot.detected_at,
                    "satellite": hotspot.satellite,
                    "instrument": hotspot.instrument,
                },
            )


def _delete_expected_news_rows(connection) -> None:  # noqa: ANN001
    for report_index in (0, 1):
        generated = NewsDataGenerator(seed=TEST_SEED).generate(
            scenario_type=ScenarioType.ACTIVE_FIRE,
            timestamp=TEST_STARTED_AT.replace(second=40),
            location=TEST_LOCATION,
            report_index=report_index,
        )
        for report in generated.reports:
            connection.execute(
                text("DELETE FROM wildfire_reports WHERE source_url = :source_url"),
                {"source_url": report.source_url},
            )


def _assert_same_timestamp(stored: datetime, expected: datetime) -> None:
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=timezone.utc)
    assert stored == expected


def test_weather_event_executes_generates_and_persists_to_neon() -> None:
    event = make_event(SimulationEventType.WEATHER)
    scenario = make_scenario(events=(event,))
    timestamp = simulation_event_timestamp(TEST_STARTED_AT, event)
    executor = SimulationEventExecutor()

    result = executor.execute(scenario=scenario, event=event, event_timestamp=timestamp)

    assert result.success is True
    assert result.generated_count == 3
    assert result.saved_count == 3
    assert result.duplicates_skipped == 0

    repository = WeatherRepository()
    expected = WeatherDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=timestamp,
        location=TEST_LOCATION,
    )
    for measurement in expected.measurements:
        station = repository.get_station_by_external_id(measurement.station.external_station_id)
        assert station is not None
        assert station.name == measurement.station.name
        observations = repository.get_observations_for_station(measurement.station.external_station_id)
        assert len(observations) == 1
        _assert_same_timestamp(observations[0].timestamp, timestamp)


def test_runner_helper_executes_first_weather_event_without_waiting_against_neon() -> None:
    event = make_event(SimulationEventType.WEATHER)
    scenario = make_scenario(events=(event,))
    output = StringIO()

    result = execute_and_report_event(
        scenario=scenario,
        event=event,
        scenario_started_at=TEST_STARTED_AT,
        executor=SimulationEventExecutor(),
        output=output,
    )

    assert result.success is True
    assert result.generated_count == 3
    assert result.saved_count == 3
    assert "[T+0s] WEATHER" in output.getvalue()

    repository = WeatherRepository()
    expected = WeatherDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=TEST_STARTED_AT,
        location=TEST_LOCATION,
    )
    for measurement in expected.measurements:
        assert repository.get_station_by_external_id(measurement.station.external_station_id) is not None


def test_satellite_event_executes_generates_and_persists_to_neon() -> None:
    event = make_event(SimulationEventType.SATELLITE, offset_seconds=20)
    scenario = make_scenario(events=(event,))
    timestamp = simulation_event_timestamp(TEST_STARTED_AT, event)
    executor = SimulationEventExecutor()

    result = executor.execute(scenario=scenario, event=event, event_timestamp=timestamp)

    assert result.success is True
    assert result.generated_count == 1
    assert result.saved_count == 1
    assert result.duplicates_skipped == 0

    expected = SatelliteDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=timestamp,
        location=TEST_LOCATION,
    ).hotspots[0]
    stored = SatelliteHotspotRepository().get_hotspots_between(timestamp, timestamp)
    assert any(
        hotspot.latitude == expected.latitude
        and hotspot.longitude == expected.longitude
        and hotspot.satellite == expected.satellite
        for hotspot in stored
    )


def test_news_event_executes_generates_persists_and_reports_duplicate_to_neon() -> None:
    event = make_event(SimulationEventType.NEWS, offset_seconds=40, source_event_index=1)
    scenario = make_scenario(events=(event,))
    timestamp = simulation_event_timestamp(TEST_STARTED_AT, event)
    executor = SimulationEventExecutor()

    first = executor.execute(scenario=scenario, event=event, event_timestamp=timestamp)
    second = executor.execute(scenario=scenario, event=event, event_timestamp=timestamp)

    assert first.success is True
    assert first.generated_count == 1
    assert first.saved_count == 1
    assert first.duplicates_skipped == 0
    assert second.success is True
    assert second.generated_count == 1
    assert second.saved_count == 0
    assert second.duplicates_skipped == 1

    expected = NewsDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=timestamp,
        location=TEST_LOCATION,
        report_index=1,
    ).reports[0]
    stored = NewsRepository().get_by_source_url(expected.source_url)
    assert stored is not None
    assert stored.source_url == expected.source_url
    assert stored.location_name == expected.location_name
    _assert_same_timestamp(stored.published_at, timestamp)
    _assert_same_timestamp(stored.fetched_at, timestamp)


def test_multi_incident_satellite_events_persist_distinct_hotspots_to_neon() -> None:
    carmel_event = make_event(
        SimulationEventType.SATELLITE,
        offset_seconds=20,
        incident_id=TEST_INCIDENT.incident_id,
    )
    second_event = make_event(
        SimulationEventType.SATELLITE,
        offset_seconds=35,
        incident_id=TEST_SECOND_INCIDENT.incident_id,
    )
    scenario = make_scenario(
        incidents=(TEST_INCIDENT, TEST_SECOND_INCIDENT),
        events=(carmel_event, second_event),
    )
    executor = SimulationEventExecutor()

    first = executor.execute(
        scenario=scenario,
        event=carmel_event,
        event_timestamp=simulation_event_timestamp(TEST_STARTED_AT, carmel_event),
    )
    second = executor.execute(
        scenario=scenario,
        event=second_event,
        event_timestamp=simulation_event_timestamp(TEST_STARTED_AT, second_event),
    )

    assert first.success is True
    assert second.success is True
    assert first.saved_count == 1
    assert second.saved_count == 1

    first_timestamp = simulation_event_timestamp(TEST_STARTED_AT, carmel_event)
    second_timestamp = simulation_event_timestamp(TEST_STARTED_AT, second_event)
    stored = SatelliteHotspotRepository().get_hotspots_between(first_timestamp, second_timestamp)
    stored_coordinates = {(hotspot.latitude, hotspot.longitude) for hotspot in stored}
    expected_first = SatelliteDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=first_timestamp,
        location=TEST_LOCATION,
    ).hotspots[0]
    expected_second = SatelliteDataGenerator(seed=TEST_SEED).generate(
        scenario_type=ScenarioType.ACTIVE_FIRE,
        timestamp=second_timestamp,
        location=TEST_SECOND_LOCATION,
    ).hotspots[0]
    assert (expected_first.latitude, expected_first.longitude) in stored_coordinates
    assert (expected_second.latitude, expected_second.longitude) in stored_coordinates
    assert (expected_first.latitude, expected_first.longitude) != (
        expected_second.latitude,
        expected_second.longitude,
    )
