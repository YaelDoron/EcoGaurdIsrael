"""Unit tests for WeatherConditionsQueryService.

Real WeatherRepository/FireDangerAssessmentRepository backed by SQLite
in-memory (the shared `sqlite_session_factory` fixture) - no Neon, no IMS,
no FFWI calculation, no network access.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.operations_activity import OperationsActivityType
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService
from src.services.simulation_control.simulation_run_manager import _idle_snapshot
from src.services.weather.weather_conditions_query_service import (
    AreaWeatherConditions,
    WeatherConditionsQueryService,
)

AS_OF = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)
AREA_LAT, AREA_LON = 31.774, 35.139


class _Repos:
    def __init__(self, session_factory):
        self.weather = WeatherRepository(session_factory=session_factory)
        self.fire_danger = FireDangerAssessmentRepository(session_factory=session_factory)

    def service(self) -> WeatherConditionsQueryService:
        return WeatherConditionsQueryService(
            fire_danger_assessment_repository=self.fire_danger, weather_repository=self.weather
        )


@pytest.fixture
def repos(sqlite_session_factory) -> _Repos:
    return _Repos(sqlite_session_factory)


def persist_observation(repos: _Repos, station_offset: int, *, minutes_ago: int = 5, **values) -> tuple[int, int]:
    """Persist one station + observation near the test area; return (observation_id, station_id)."""
    station = WeatherStation(
        external_station_id=930000 + station_offset,
        name=f"SIM-TEST-{station_offset:02d}",
        latitude=AREA_LAT,
        longitude=AREA_LON,
    )
    repos.weather.save_station(station)
    timestamp = AS_OF - timedelta(minutes=minutes_ago)
    observation_values = dict(temperature=30.0, relative_humidity=25.0, wind_speed=20.0, wind_gust=28.0)
    observation_values.update(values)
    repos.weather.save_observation(
        WeatherObservation(station_external_id=station.external_station_id, timestamp=timestamp, **observation_values)
    )
    candidates = repos.weather.get_recent_observations_for_area_candidates(
        latitude=AREA_LAT, longitude=AREA_LON, radius_km=5.0, start_time=AS_OF - timedelta(hours=2), end_time=AS_OF
    )
    record = next(
        c
        for c in candidates
        if c.observation.station_external_id == station.external_station_id and c.observation.timestamp == timestamp
    )
    return record.observation_id, record.station_id


def save_assessment(repos: _Repos, traced: list[tuple[int, int]], **overrides) -> StoredFireDangerAssessment:
    values = dict(
        area_id="area-jerusalem",
        area_name="Jerusalem Forest Demo Area",
        area_latitude=AREA_LAT,
        area_longitude=AREA_LON,
        area_radius_km=5.0,
        assessed_at=AS_OF,
        status=FireDangerAssessmentStatus.VALID,
        score=55.6,
        level=FireDangerLevel.VERY_HIGH,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    return repos.fire_danger.save_assessment(
        FireDangerAssessment(**values),
        tuple(observation_id for observation_id, _ in traced),
        tuple(station_id for _, station_id in traced),
    )


# ---------------------------------------------------------------------------
# Averaging semantics (moved unchanged from OperationsOverviewQueryService)
# ---------------------------------------------------------------------------


def test_single_station_conditions_are_that_observations_own_values(repos):
    save_assessment(repos, [persist_observation(repos, 1)])

    [conditions] = repos.service().get_latest_for_all_areas()

    assert conditions == AreaWeatherConditions(
        area_name="Jerusalem Forest Demo Area",
        observed_at=AS_OF - timedelta(minutes=5),
        assessed_at=AS_OF,
        station_count=1,
        temperature_c=30.0,
        relative_humidity_pct=25.0,
        wind_speed_kmh=20.0,
        wind_gust_kmh=28.0,
    )


def test_multiple_stations_are_averaged_per_field_and_observed_at_is_the_latest(repos):
    traced = [
        persist_observation(repos, 1, minutes_ago=9, temperature=30.0, relative_humidity=20.0, wind_speed=10.0, wind_gust=20.0),
        persist_observation(repos, 2, minutes_ago=4, temperature=34.0, relative_humidity=30.0, wind_speed=30.0, wind_gust=None),
    ]
    save_assessment(repos, traced)

    [conditions] = repos.service().get_latest_for_all_areas()

    assert conditions.station_count == 2
    assert conditions.temperature_c == pytest.approx(32.0)
    assert conditions.relative_humidity_pct == pytest.approx(25.0)
    assert conditions.wind_speed_kmh == pytest.approx(20.0)
    # Gust is averaged only over stations that report one.
    assert conditions.wind_gust_kmh == pytest.approx(20.0)
    assert conditions.observed_at == AS_OF - timedelta(minutes=4)


def test_missing_gust_everywhere_stays_none(repos):
    save_assessment(repos, [persist_observation(repos, 1, wind_gust=None)])

    [conditions] = repos.service().get_latest_for_all_areas()

    assert conditions.wind_gust_kmh is None


def test_field_missing_on_one_station_is_averaged_over_the_others(repos):
    traced = [
        persist_observation(repos, 1, temperature=30.0),
        persist_observation(repos, 2, temperature=None),
    ]
    save_assessment(repos, traced)

    [conditions] = repos.service().get_latest_for_all_areas()

    assert conditions.temperature_c == pytest.approx(30.0)
    assert conditions.station_count == 2


def test_assessment_whose_stations_all_lack_temperature_is_skipped(repos):
    save_assessment(repos, [persist_observation(repos, 1, temperature=None)])

    assert repos.service().get_latest_for_all_areas() == ()


def test_assessment_whose_traced_observations_cannot_be_loaded_is_skipped(repos):
    stored = save_assessment(repos, [persist_observation(repos, 1)])
    untraceable = StoredFireDangerAssessment(
        assessment_id=stored.assessment_id,
        assessment=stored.assessment,
        observation_ids=(987654,),
        station_ids=(1,),
        created_at=stored.created_at,
    )

    assert repos.service().summarize_assessments([untraceable]) == ()


def test_summarize_assessments_of_nothing_is_empty(repos):
    assert repos.service().summarize_assessments([]) == ()


# ---------------------------------------------------------------------------
# Latest per area
# ---------------------------------------------------------------------------


def test_latest_assessment_per_area_is_used_and_areas_are_sorted_by_name(repos):
    save_assessment(repos, [persist_observation(repos, 1, temperature=20.0)], assessed_at=AS_OF - timedelta(hours=1))
    save_assessment(repos, [persist_observation(repos, 2, temperature=36.0)], assessed_at=AS_OF)
    save_assessment(
        repos,
        [persist_observation(repos, 3, temperature=33.0)],
        area_id="area-judean",
        area_name="Judean Hills Demo Area",
        level=FireDangerLevel.HIGH,
        score=32.5,
    )

    result = repos.service().get_latest_for_all_areas()

    assert [c.area_name for c in result] == ["Jerusalem Forest Demo Area", "Judean Hills Demo Area"]
    assert result[0].temperature_c == pytest.approx(36.0)  # the newer Jerusalem assessment, not the older one
    assert result[1].temperature_c == pytest.approx(33.0)


def test_area_whose_latest_assessment_is_insufficient_data_is_omitted_not_backfilled(repos):
    save_assessment(repos, [persist_observation(repos, 1)], assessed_at=AS_OF - timedelta(hours=1))
    save_assessment(
        repos, [], assessed_at=AS_OF, status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None
    )

    assert repos.service().get_latest_for_all_areas() == ()


def test_no_assessments_means_no_conditions(repos):
    assert repos.service().get_latest_for_all_areas() == ()


# ---------------------------------------------------------------------------
# Parity with the Operations Overview Weather Conditions rows
# ---------------------------------------------------------------------------


class _IdleRunManager:
    def get_current_snapshot(self):
        return _idle_snapshot()


def test_values_match_the_operations_overview_weather_conditions_row(repos, sqlite_session_factory):
    traced = [
        persist_observation(repos, 1, temperature=36.1, relative_humidity=15.2, wind_speed=18.4, wind_gust=25.0),
        persist_observation(repos, 2, temperature=38.9, relative_humidity=17.9, wind_speed=21.2, wind_gust=35.6),
    ]
    save_assessment(repos, traced)
    overview = OperationsOverviewQueryService(
        fire_danger_query_service=FireDangerQueryService(
            fire_danger_assessment_repository=repos.fire_danger, weather_repository=repos.weather
        ),
        active_fire_events_service=ActiveFireEventsService(
            fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory=sqlite_session_factory),
        ),
        fire_danger_assessment_repository=repos.fire_danger,
        satellite_hotspot_repository=SatelliteHotspotRepository(session_factory=sqlite_session_factory),
        news_repository=NewsRepository(session_factory=sqlite_session_factory),
        weather_repository=repos.weather,
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(session_factory=sqlite_session_factory),
        global_planning_run_repository=GlobalPlanningRunRepository(session_factory=sqlite_session_factory),
        simulation_run_manager=_IdleRunManager(),
    )

    [row] = [
        item
        for item in overview.get_overview(as_of=AS_OF).activity_feed.items
        if item.activity_type is OperationsActivityType.WEATHER_CONDITIONS
    ]
    [conditions] = repos.service().get_latest_for_all_areas()

    assert conditions.area_name == row.preview.area_name
    assert conditions.temperature_c == row.preview.temperature_c
    assert conditions.relative_humidity_pct == row.preview.relative_humidity_pct
    assert conditions.wind_speed_kmh == row.preview.wind_speed_kmh
    assert conditions.wind_gust_kmh == row.preview.wind_gust_kmh
    assert conditions.observed_at == row.occurred_at


# ---------------------------------------------------------------------------
# Architecture guard: read-only, no recalculation, no external calls
# ---------------------------------------------------------------------------


def test_service_imports_no_external_client_calculator_or_agent():
    path = Path(__file__).resolve().parents[4] / "backend/src/services/weather/weather_conditions_query_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ] + [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    for module in imported:
        for forbidden in ("src.external", "src.calculators", "src.agents", "src.simulation", "requests", "httpx"):
            assert forbidden not in module, module


def test_service_never_writes():
    source = (
        Path(__file__).resolve().parents[4] / "backend/src/services/weather/weather_conditions_query_service.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("save_", ".add(", "delete", "commit"):
        assert forbidden not in source
