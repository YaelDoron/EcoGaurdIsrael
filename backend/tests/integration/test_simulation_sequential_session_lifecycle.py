"""Task A1.5 regression tests: sequential simulation events must not leak DB
sessions/connections in one long-lived process.

Context: a live `operations_demo` run appeared to "hang" after a small
number of events. Instrumenting SQLAlchemy pool checkout/checkin events
during a real (Neon) run proved checkouts and checkins stayed exactly
balanced (outstanding=0) right up to the point of the stall - the session
lifecycle in `_session_scope`-style repositories (`with ...: finally:
session.close()`) was never the problem. The actual stall was an unbounded
live OSM/Overpass fetch inside RoadNetworkFetcher (see
test_road_network_fetcher.py for that fix).

These tests exist to (a) lock in that the session lifecycle really is
leak-free across sequential real-production-code events, using the
project's normal SQLite test DB (no Neon dependency), and (b) prove it with
a pool small enough that a real leak would deterministically fail the test,
not just "happen to pass" on a generously-sized pool.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from src.database.base import Base
from src.database.models import (  # noqa: F401
    fire_danger_assessment_db,
    fire_danger_assessment_weather_input_db,
    fire_event_db,
    fire_event_news_evidence_db,
    fire_event_satellite_evidence_db,
    fire_severity_assessment_db,
    fire_severity_assessment_satellite_input_db,
    fire_severity_assessment_weather_input_db,
    fire_spread_prediction_cell_db,
    fire_spread_prediction_db,
    fire_spread_prediction_weather_input_db,
    fire_station_db,
    firefighting_resource_db,
    global_planning_run_db,
    global_planning_run_event_db,
    graph_edge_db,
    graph_node_db,
    response_action_db,
    response_plan_db,
    response_plan_planning_state_db,
    response_plan_uncovered_target_db,
    response_target_db,
    response_target_set_db,
    resource_commitment_db,
    route_planning_run_db,
    route_result_db,
    satellite_hotspot_db,
    weather_observation_db,
    weather_station_db,
    wildfire_report_db,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import WeatherStationNotStoredError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.weather_repository import WeatherRepository
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    build_carmel_golan_active_fire_scenario,
    build_low_risk_no_fire_scenario,
)
from tests.integration.test_simulation_response_planning_e2e import (
    CARMEL_RESOURCE_ID,
    CARMEL_STATION_ID,
    GOLAN_RESOURCE_ID,
    GOLAN_STATION_ID,
    RealSimulationStack,
    seed_station_and_road_network,
)

TIMESTAMP = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


class PoolCounters:
    """Tracks checkout/checkin counts for one engine's pool."""

    def __init__(self) -> None:
        self.checkouts = 0
        self.checkins = 0

    @property
    def outstanding(self) -> int:
        return self.checkouts - self.checkins


def instrument(engine) -> PoolCounters:
    counters = PoolCounters()

    @event.listens_for(engine, "checkout")
    def _on_checkout(dbapi_connection, connection_record, connection_proxy):  # noqa: ANN001
        counters.checkouts += 1

    @event.listens_for(engine, "checkin")
    def _on_checkin(dbapi_connection, connection_record):  # noqa: ANN001
        counters.checkins += 1

    return counters


# ---------------------------------------------------------------------------
# 1 & 6: multiple sequential events (active-fire + detection + severity +
# spread + response targets + operational refresh + global planning) release
# every session they open.
# ---------------------------------------------------------------------------


def test_multi_incident_active_fire_sequence_does_not_leak_sessions(sqlite_session_factory, sqlite_engine):
    counters = instrument(sqlite_engine)
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=CARMEL_STATION_ID,
        resource_id=CARMEL_RESOURCE_ID,
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
    )
    seed_station_and_road_network(
        sqlite_session_factory,
        station_id=GOLAN_STATION_ID,
        resource_id=GOLAN_RESOURCE_ID,
        latitude=GOLAN_LOCATION.latitude,
        longitude=GOLAN_LOCATION.longitude,
    )
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_carmel_golan_active_fire_scenario(seed=42)

    stack.run_scenario(scenario)

    assert counters.outstanding == 0, (
        f"{counters.outstanding} connection(s) were checked out but never returned "
        f"(checkouts={counters.checkouts}, checkins={counters.checkins})"
    )


# ---------------------------------------------------------------------------
# 5: Fire Danger only / no-fire paths do not leak.
# ---------------------------------------------------------------------------


def test_no_fire_weather_only_sequence_does_not_leak_sessions(sqlite_session_factory, sqlite_engine):
    counters = instrument(sqlite_engine)
    stack = RealSimulationStack(sqlite_session_factory)
    scenario = build_low_risk_no_fire_scenario(location=CARMEL_LOCATION, seed=42)

    stack.run_scenario(scenario)

    assert counters.outstanding == 0


# ---------------------------------------------------------------------------
# 3: an exception during event processing rolls back and releases the
# session (not just the success path).
# ---------------------------------------------------------------------------


def test_repository_exception_still_releases_the_session(sqlite_session_factory, sqlite_engine):
    counters = instrument(sqlite_engine)
    weather_repository = WeatherRepository(sqlite_session_factory)

    with pytest.raises(WeatherStationNotStoredError):
        weather_repository.save_observation(
            WeatherObservation(
                station_external_id=999999,  # never saved -> raises
                timestamp=TIMESTAMP,
                temperature=25.0,
                relative_humidity=40.0,
                wind_speed=10.0,
                wind_direction=180.0,
                wind_gust=12.0,
                rainfall=0.0,
            )
        )

    assert counters.outstanding == 0

    # The pool must still be usable afterward - a leaked/broken session from
    # the failed call would leave the next call blocked or erroring.
    station = WeatherStation(
        external_station_id=999999,
        name="Post-failure station",
        latitude=32.7,
        longitude=35.0,
        region_id=None,
        active=True,
    )
    weather_repository.save_station(station)
    assert counters.outstanding == 0


# ---------------------------------------------------------------------------
# 4: early-return read paths (no matching rows) also release the session.
# ---------------------------------------------------------------------------


def test_early_return_empty_query_releases_the_session(sqlite_session_factory, sqlite_engine):
    counters = instrument(sqlite_engine)
    fire_event_repository = FireEventRepository(sqlite_session_factory)

    active_events = fire_event_repository.get_active_events_near(
        latitude=CARMEL_LOCATION.latitude,
        longitude=CARMEL_LOCATION.longitude,
        radius_km=5.0,
        as_of=TIMESTAMP,
    )

    assert active_events == ()
    assert counters.outstanding == 0


# ---------------------------------------------------------------------------
# 2 (strong form): a deliberately tiny pool (size=1, no overflow) would
# deterministically time out on the SECOND sequential repository call if a
# session leaked from the first one. This is the regression test that would
# fail under the old (hypothetical) leaking behavior and passes now.
# ---------------------------------------------------------------------------


def test_sequential_calls_survive_a_pool_of_size_one(tmp_path):
    db_path = tmp_path / "pool_exhaustion_regression.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=2,
    )
    try:
        Base.metadata.create_all(bind=engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        weather_repository = WeatherRepository(session_factory)

        # 10 sequential calls against a pool that can hold exactly one
        # connection: if any call failed to return its connection, the next
        # checkout would block for pool_timeout=2s and then raise.
        for index in range(10):
            station = WeatherStation(
                external_station_id=500_000 + index,
                name=f"Pool-test station {index}",
                latitude=32.7,
                longitude=35.0,
                region_id=None,
                active=True,
            )
            weather_repository.save_station(station)
            weather_repository.get_station_by_external_id(500_000 + index)
    finally:
        engine.dispose()
