"""Unit tests for OperationsOverviewQueryService (Task A6, Part 30).

Uses REAL repositories/services backed by SQLite in-memory (matching
tests/repositories/*'s own convention, via the shared `sqlite_session_factory`
fixture) rather than a wall of hand-built fakes - this exercises genuine
delegation to FireDangerQueryService/ActiveFireEventsService/
FireSeverityAssessmentRepository.get_latest_for_events (all already tested
elsewhere), not a trivial pass-through of a fake. Only SimulationRunManager
is faked (constructing a real one pulls in production coordinator wiring
irrelevant to this suite). No FFWI/severity/detection/spread calculation,
agent invocation, or external I/O happens anywhere in this file.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import src.config.settings  # noqa: F401 - ensures sys.modules entry below exists

# `src/config/__init__.py` does `from src.config.settings import settings`,
# which shadows the package's own `settings` attribute with the Settings
# instance - a string-based monkeypatch target ("src.config.settings.settings")
# resolves through that same shadowed attribute chain and silently patches
# the wrong object. sys.modules keys by full dotted name and is set
# directly by the import system, immune to that shadowing (same fix as
# tests/services/simulation_control/test_simulation_run_manager.py).
_settings_module = sys.modules["src.config.settings"]

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.fire_report import WildfireReport
from src.models.operations_activity import OperationsActivityType
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService
from src.services.simulation_control.simulation_run_manager import (
    SimulationCurrentEventSnapshot,
    SimulationRunSnapshot,
    SimulationRunState,
    _idle_snapshot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

AS_OF = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


class FakeSimulationRunManager:
    def __init__(self, snapshot=None):
        self._snapshot = snapshot if snapshot is not None else _idle_snapshot()

    def get_current_snapshot(self):
        return self._snapshot


def make_service(*, sqlite_session_factory, simulation_run_manager=None) -> OperationsOverviewQueryService:
    return OperationsOverviewQueryService(
        fire_danger_query_service=FireDangerQueryService(
            fire_danger_assessment_repository=FireDangerAssessmentRepository(
                session_factory=sqlite_session_factory
            ),
            weather_repository=WeatherRepository(session_factory=sqlite_session_factory),
        ),
        active_fire_events_service=ActiveFireEventsService(
            fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(
                session_factory=sqlite_session_factory
            ),
        ),
        fire_danger_assessment_repository=FireDangerAssessmentRepository(session_factory=sqlite_session_factory),
        satellite_hotspot_repository=SatelliteHotspotRepository(session_factory=sqlite_session_factory),
        news_repository=NewsRepository(session_factory=sqlite_session_factory),
        weather_repository=WeatherRepository(session_factory=sqlite_session_factory),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(
            session_factory=sqlite_session_factory
        ),
        global_planning_run_repository=GlobalPlanningRunRepository(session_factory=sqlite_session_factory),
        simulation_run_manager=simulation_run_manager or FakeSimulationRunManager(),
    )


def persist_weather_observation(weather_repository: WeatherRepository, station_offset: int) -> tuple[int, int]:
    station = WeatherStation(
        external_station_id=920000 + station_offset,
        name=f"Station {station_offset}",
        latitude=32.7,
        longitude=35.0,
    )
    weather_repository.save_station(station)
    observation = WeatherObservation(
        station_external_id=station.external_station_id,
        timestamp=AS_OF - timedelta(minutes=5),
        temperature=30.0,
        relative_humidity=25.0,
        wind_speed=20.0,
    )
    weather_repository.save_observation(observation)
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.7, longitude=35.0, radius_km=5.0,
        start_time=AS_OF - timedelta(minutes=30), end_time=AS_OF,
    )
    record = next(c for c in candidates if c.observation.station_external_id == station.external_station_id)
    return record.observation_id, record.station_id


def make_fire_danger_assessment(**overrides) -> FireDangerAssessment:
    values = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=AS_OF,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    return FireDangerAssessment(**values)


def make_fire_event(**overrides) -> FireEvent:
    values = dict(
        latitude=32.7,
        longitude=35.0,
        detected_at=AS_OF,
        updated_at=AS_OF,
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.8,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    return FireEvent(**values)


def make_hotspot(**overrides) -> SatelliteHotspot:
    values = dict(latitude=32.7, longitude=35.0, detected_at=AS_OF, confidence="h", frp=50.0, satellite="N20")
    values.update(overrides)
    return SatelliteHotspot(**values)


def persist_hotspot(satellite_repository: SatelliteHotspotRepository, **overrides) -> int:
    satellite_repository.save_hotspot(make_hotspot(**overrides))
    return satellite_repository.get_recent(10)[0].id


def make_severity_assessment(fire_event_id: int, **overrides) -> FireSeverityAssessment:
    values = dict(
        fire_event_id=fire_event_id,
        assessed_at=AS_OF,
        status=FireSeverityAssessmentStatus.VALID,
        score=60.0,
        level=FireSeverityLevel.HIGH,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    return FireSeverityAssessment(**values)


# ---------------------------------------------------------------------------
# 1. Empty overview
# ---------------------------------------------------------------------------


def test_empty_overview_returns_all_empty_sections(sqlite_session_factory):
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.fire_danger_areas == ()
    assert snapshot.active_fires == ()
    assert snapshot.activity_feed.items == ()
    assert snapshot.generated_at == AS_OF


# ---------------------------------------------------------------------------
# 2. Fire Danger data delegated from A4
# ---------------------------------------------------------------------------


def test_fire_danger_areas_delegated_from_a4(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    fire_danger_repository.save_assessment(make_fire_danger_assessment(), (observation_id,), (station_id,))
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert len(snapshot.fire_danger_areas) == 1
    assert snapshot.fire_danger_areas[0].area_name == "Carmel"
    assert snapshot.fire_danger_areas[0].assessment.level is FireDangerLevel.VERY_HIGH


def test_fire_danger_no_assessment_is_not_fabricated_as_low(sqlite_session_factory):
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.fire_danger_areas == ()


# ---------------------------------------------------------------------------
# 3/4/5. Active SUSPECTED/CONFIRMED included, non-active excluded
# ---------------------------------------------------------------------------


def create_event(fire_event_repository, satellite_repository, hotspot_id, **overrides):
    return fire_event_repository.create_event(
        make_fire_event(**overrides), (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),)
    )


def test_suspected_and_confirmed_active_events_are_included_and_resolved_excluded(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)

    suspected = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.SUSPECTED)
    confirmed = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)
    resolved = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.SUSPECTED)
    fire_event_repository.update_event(resolved.id, make_fire_event(status=FireEventStatus.RESOLVED))

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    active_ids = {item.fire_event_id for item in snapshot.active_fires}
    assert suspected.id in active_ids
    assert confirmed.id in active_ids
    assert resolved.id not in active_ids


# ---------------------------------------------------------------------------
# 6/7/8/9. Latest Severity attachment via batched lookup
# ---------------------------------------------------------------------------


def test_active_fire_with_severity_gets_it_attached(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    observation_id, _ = persist_weather_observation(weather_repository, 1)
    event = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)
    severity_repository.save_assessment(
        make_severity_assessment(event.id, score=70.0, level=FireSeverityLevel.CRITICAL),
        weather_observation_ids=(observation_id,), satellite_hotspot_ids=(hotspot_id,), selected_frp_hotspot_id=hotspot_id,
    )

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    item = next(i for i in snapshot.active_fires if i.fire_event_id == event.id)
    assert item.severity is not None
    assert item.severity.level is FireSeverityLevel.CRITICAL


def test_active_fire_without_severity_is_null(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    event = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    item = next(i for i in snapshot.active_fires if i.fire_event_id == event.id)
    assert item.severity is None


def test_latest_persisted_severity_used_even_if_insufficient_data(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    observation_id, _ = persist_weather_observation(weather_repository, 1)
    event = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)
    severity_repository.save_assessment(
        make_severity_assessment(
            event.id, assessed_at=AS_OF - timedelta(minutes=10), score=70.0, level=FireSeverityLevel.CRITICAL
        ),
        weather_observation_ids=(observation_id,), satellite_hotspot_ids=(hotspot_id,), selected_frp_hotspot_id=hotspot_id,
    )
    severity_repository.save_assessment(
        make_severity_assessment(
            event.id, assessed_at=AS_OF, status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None
        ),
        weather_observation_ids=(), satellite_hotspot_ids=(), selected_frp_hotspot_id=None,
    )

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    item = next(i for i in snapshot.active_fires if i.fire_event_id == event.id)
    assert item.severity is not None
    assert item.severity.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert item.severity.score is None
    assert item.severity.level is None


def test_multiple_active_fires_use_one_batched_severity_lookup(sqlite_session_factory, monkeypatch):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    observation_id, _ = persist_weather_observation(weather_repository, 1)
    first = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)
    second = create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.SUSPECTED)
    for event in (first, second):
        severity_repository.save_assessment(
            make_severity_assessment(event.id), weather_observation_ids=(observation_id,),
            satellite_hotspot_ids=(hotspot_id,), selected_frp_hotspot_id=hotspot_id,
        )

    calls = []
    original = FireSeverityAssessmentRepository.get_latest_for_events

    def counted(self, fire_event_ids):
        # ActiveFireEventsService passes a one-shot generator expression -
        # materialize it once and reuse the tuple, or the real call below
        # would iterate an already-exhausted generator.
        ids = tuple(fire_event_ids)
        calls.append(ids)
        return original(self, ids)

    monkeypatch.setattr(FireSeverityAssessmentRepository, "get_latest_for_events", counted)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    assert len(calls) == 1
    assert set(calls[0]) == {first.id, second.id}
    assert all(item.severity is not None for item in snapshot.active_fires)


# ---------------------------------------------------------------------------
# 10. FireEvent has coordinates and only a resolved (never fabricated)
# location_name - no bare "area_name" field exists.
# ---------------------------------------------------------------------------


def test_active_fire_has_no_fabricated_area_name_field(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    for item in snapshot.active_fires:
        assert not hasattr(item, "area_name")
        assert item.latitude == 32.7
        assert item.longitude == 35.0


def test_active_fire_location_name_is_none_without_a_containing_fire_danger_area(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    assert len(snapshot.active_fires) == 1
    assert snapshot.active_fires[0].location_name is None


def test_active_fire_location_name_resolves_to_a_containing_fire_danger_area(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    # FireEvent (and its evidence hotspot, via persist_hotspot's default
    # coordinates) sit at (32.7, 35.0) - the same area this assessment covers.
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(area_latitude=32.7, area_longitude=35.0, area_radius_km=5.0),
        (observation_id,),
        (station_id,),
    )
    hotspot_id = persist_hotspot(satellite_repository)
    create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.active_fires[0].location_name == "Carmel"


# ---------------------------------------------------------------------------
# 11. Deterministic active-fire ordering
# ---------------------------------------------------------------------------


def test_active_fire_ordering_is_deterministic_across_calls(sqlite_session_factory):
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = persist_hotspot(satellite_repository)
    for _ in range(3):
        create_event(fire_event_repository, satellite_repository, hotspot_id, status=FireEventStatus.CONFIRMED)

    service = make_service(sqlite_session_factory=sqlite_session_factory)
    first = [item.fire_event_id for item in service.get_overview(as_of=AS_OF).active_fires]
    second = [item.fire_event_id for item in service.get_overview(as_of=AS_OF).active_fires]

    assert first == second


# ---------------------------------------------------------------------------
# 12/13/14/15. Simulation summary
# ---------------------------------------------------------------------------


def test_simulation_disabled_returns_enabled_false_and_null_run(sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=False),
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.simulation.enabled is False
    assert snapshot.simulation.run is None


def test_simulation_enabled_no_run_returns_null_run(sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=True),
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory, simulation_run_manager=FakeSimulationRunManager())

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.simulation.enabled is True
    assert snapshot.simulation.run is None


def test_simulation_running_exposes_progress(sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=True),
    )
    running_snapshot = SimulationRunSnapshot(
        run_id="run-1", state=SimulationRunState.RUNNING, preset_id="operations_demo", seed=42,
        mode="automatic", simulation_duration_seconds=120, events_total=18, events_completed=5,
        events_succeeded=5, events_failed=0,
        current_event=SimulationCurrentEventSnapshot(
            event_index=5, incident_id="incident-1", event_type="weather", timestamp_offset_sec=60
        ),
        started_at=AS_OF, completed_at=None, wall_clock_elapsed_seconds=12.5,
        last_message="Simulation running.", error=None,
    )
    service = make_service(
        sqlite_session_factory=sqlite_session_factory,
        simulation_run_manager=FakeSimulationRunManager(running_snapshot),
    )

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.simulation.enabled is True
    assert snapshot.simulation.run.state is SimulationRunState.RUNNING
    assert snapshot.simulation.run.events_completed == 5


def test_simulation_completed_exposes_final_state(sqlite_session_factory, monkeypatch):
    monkeypatch.setattr(
        _settings_module,
        "settings",
        SimpleNamespace(ENABLE_SIMULATION_CONTROL_API=True),
    )
    completed_snapshot = SimulationRunSnapshot(
        run_id="run-1", state=SimulationRunState.COMPLETED, preset_id="operations_demo", seed=42,
        mode="automatic", simulation_duration_seconds=120, events_total=18, events_completed=18,
        events_succeeded=18, events_failed=0, current_event=None,
        started_at=AS_OF, completed_at=AS_OF + timedelta(minutes=5), wall_clock_elapsed_seconds=300.0,
        last_message="Simulation run completed.", error=None,
    )
    service = make_service(
        sqlite_session_factory=sqlite_session_factory,
        simulation_run_manager=FakeSimulationRunManager(completed_snapshot),
    )

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.simulation.run.state is SimulationRunState.COMPLETED
    assert snapshot.simulation.run.completed_at == AS_OF + timedelta(minutes=5)


# ---------------------------------------------------------------------------
# 16. Overview GET does not mutate state
# ---------------------------------------------------------------------------


def test_get_overview_does_not_mutate_state(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    fire_danger_repository.save_assessment(make_fire_danger_assessment(), (observation_id,), (station_id,))
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    first = service.get_overview(as_of=AS_OF)
    second = service.get_overview(as_of=AS_OF)

    assert first.fire_danger_areas == second.fire_danger_areas
    assert len(fire_danger_repository.get_recent(10)) == 1


# ---------------------------------------------------------------------------
# Weather Activity Feed signal ("Risk-elevating weather conditions")
# ---------------------------------------------------------------------------


def _weather_feed_items(snapshot):
    from src.models.operations_activity import OperationsActivityType

    return [item for item in snapshot.activity_feed.items if item.activity_type is OperationsActivityType.WEATHER_CONDITIONS]


@pytest.mark.parametrize(
    "level",
    [
        FireDangerLevel.LOW,
        FireDangerLevel.MODERATE,
        FireDangerLevel.HIGH,
        FireDangerLevel.VERY_HIGH,
        FireDangerLevel.EXTREME,
    ],
)
def test_weather_signal_appears_for_every_fire_danger_level(sqlite_session_factory, level):
    """Part K: every successfully calculated (VALID) assessment gets a
    Weather Conditions signal now, not just HIGH+ - the wording distinction
    (neutral vs risk-elevating) is a frontend presentation concern only."""
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    fire_danger_repository.save_assessment(make_fire_danger_assessment(level=level), (observation_id,), (station_id,))
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    weather_items = _weather_feed_items(snapshot)
    assert len(weather_items) == 1
    assert weather_items[0].preview.fire_danger_level is level


def test_weather_signal_does_not_appear_for_insufficient_data(sqlite_session_factory):
    """Part T item 20: no valid weather input -> no Weather activity, never fabricated."""
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(
            status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None
        ),
        (),
        (),
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert _weather_feed_items(snapshot) == []


def test_weather_signal_uses_the_exact_persisted_weather_inputs_not_a_recalculation(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH), (observation_id,), (station_id,)
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    preview = _weather_feed_items(snapshot)[0].preview
    # persist_weather_observation's own fixture values (temperature=30.0,
    # relative_humidity=25.0, wind_speed=20.0) - a single contributing
    # station means the "mean" is exactly that one observation's values,
    # never a recomputed FFWI score.
    assert preview.temperature_c == pytest.approx(30.0)
    assert preview.relative_humidity_pct == pytest.approx(25.0)
    assert preview.wind_speed_kmh == pytest.approx(20.0)
    assert preview.wind_gust_kmh is None
    assert preview.area_name == "Carmel"
    assert preview.fire_danger_assessment_id > 0


def test_weather_signal_averages_multiple_contributing_stations(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id_1, station_id_1 = persist_weather_observation(weather_repository, 1)
    observation_id_2, station_id_2 = persist_weather_observation(weather_repository, 2)
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH),
        (observation_id_1, observation_id_2),
        (station_id_1, station_id_2),
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    weather_items = _weather_feed_items(snapshot)
    # One assessment -> at most one Weather Conditions activity, never one
    # row per contributing station (Part P/N item 22).
    assert len(weather_items) == 1
    preview = weather_items[0].preview
    # Both observations use identical fixture values - mean must equal that
    # same value, not double-count or otherwise distort it.
    assert preview.temperature_c == pytest.approx(30.0)


def test_weather_signal_occurred_at_uses_the_contributing_observations_own_timestamp(sqlite_session_factory):
    """occurred_at is the real domain/source timestamp of the traced weather
    observation(s) - the SAME kind of timestamp Fire Danger's own
    `assessed_at` is - never the assessment row's DB-insert `created_at`
    (mixing an insert-time member with a domain-time member in the same
    causal pair was reconsidered - see this task's report)."""
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    saved = fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH), (observation_id,), (station_id,)
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    weather_item = _weather_feed_items(snapshot)[0]
    # persist_weather_observation's fixture timestamps its one observation
    # at AS_OF - 5min - that real source timestamp, never the assessment's
    # own (later) DB-insert created_at.
    assert weather_item.occurred_at == AS_OF - timedelta(minutes=5)
    assert weather_item.occurred_at != saved.created_at


def test_weather_signal_entity_id_is_the_fire_danger_assessment_id(sqlite_session_factory):
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    saved = fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH), (observation_id,), (station_id,)
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    weather_item = _weather_feed_items(snapshot)[0]
    assert weather_item.entity_id == saved.assessment_id
    assert weather_item.activity_id == f"weather_conditions:{saved.assessment_id}"


def test_weather_conditions_sorts_before_its_own_fire_danger_result_on_an_exact_timestamp_tie(
    sqlite_session_factory,
):
    """Part I: a same-assessment causal pair must read cause (Weather) before
    effect (Fire Danger) even when their real timestamps are indistinguishable -
    never by mutating either timestamp, only a deterministic tie-break."""
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    # persist_weather_observation's fixture observation is timestamped at
    # exactly AS_OF - 5min - assess at that SAME instant to force a real tie.
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH, assessed_at=AS_OF - timedelta(minutes=5)),
        (observation_id,),
        (station_id,),
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    causal_pair = [item for item in items if item.activity_type in (OperationsActivityType.WEATHER_CONDITIONS, OperationsActivityType.FIRE_DANGER)]
    assert len(causal_pair) == 2
    assert causal_pair[0].occurred_at == causal_pair[1].occurred_at
    assert causal_pair[0].activity_type is OperationsActivityType.WEATHER_CONDITIONS
    assert causal_pair[1].activity_type is OperationsActivityType.FIRE_DANGER


def test_unrelated_activity_ordering_is_unaffected_by_the_causal_pair_tie_break(sqlite_session_factory):
    """The Weather/Fire Danger tie-break must not disrupt newest-first
    ordering (by available_at) for activities that are not part of a tied
    pair. News's available_at is its own `fetched_at` (test-controlled,
    domain data); Weather/Fire Danger's available_at is the assessment's
    real DB-insert `created_at` (server-assigned at save time, i.e. real
    "now") - fetched_at is set comfortably in the future so this comparison
    is deterministic regardless of when the test actually runs.
    """
    fire_danger_repository = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    news_repository = NewsRepository(session_factory=sqlite_session_factory)
    observation_id, station_id = persist_weather_observation(weather_repository, 1)
    fire_danger_repository.save_assessment(
        make_fire_danger_assessment(level=FireDangerLevel.HIGH, assessed_at=AS_OF), (observation_id,), (station_id,)
    )
    strictly_newer_available_at = datetime.now(timezone.utc) + timedelta(days=3650)
    news_repository.save_report(
        WildfireReport(
            source_url="https://example.com/newer", source_feed="Feed", title="Newer headline",
            summary="s", location_name=None, latitude=None, longitude=None,
            published_at=AS_OF + timedelta(minutes=1), fetched_at=strictly_newer_available_at,
        )
    )
    service = make_service(sqlite_session_factory=sqlite_session_factory)

    items = service.get_overview(as_of=strictly_newer_available_at).activity_feed.items

    # The strictly-newer, unrelated news item still sorts first by
    # available_at - the tie-break only matters for items sharing one instant.
    assert items[0].activity_type is OperationsActivityType.NEWS_REPORT


def test_weather_signal_never_calls_external_weather_provider(sqlite_session_factory):
    """Static guarantee: the feed-assembly method never imports/calls an
    IMS/live-weather client - it only reads already-persisted rows."""
    import inspect

    from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService

    source = inspect.getsource(OperationsOverviewQueryService._weather_conditions_candidates)
    for forbidden in ("requests.", "httpx.", "ims_client", "IMSClient", "fetch_weather"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Architecture guard (Task A6, Part 33)
# ---------------------------------------------------------------------------


def test_query_service_does_not_import_business_execution_components():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDetectionCalculator",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "SimulationRefreshCoordinator",
        "GlobalPlanningOrchestrator",
        "genetic_optimizer",
        "RoutePlanningAgent",
        "DemoSimulationRunner",
        "fastapi",
        "pydantic",
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "services"
        / "operations"
        / "operations_overview_query_service.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []
