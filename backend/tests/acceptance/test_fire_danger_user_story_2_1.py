"""Acceptance coverage for User Story 2.1 fire-danger assessment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.agents.analysis import FireDangerAssessmentAgent
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.models import AssessmentArea, FireDangerAssessmentStatus
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger import FireDangerInputService
from src.simulation import (
    CARMEL_LOCATION,
    ScenarioType,
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    build_scenario,
)
from src.simulation.analysis import SimulationFireDangerCoordinator
from src.simulation.simulation_event_executor import simulation_event_timestamp

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
LOW_AREA = AssessmentArea(id="acceptance-low", name="Acceptance Low", latitude=32.0, longitude=35.0, radius_km=5.0)
HIGH_AREA = AssessmentArea(id="acceptance-high", name="Acceptance High", latitude=33.0, longitude=35.0, radius_km=5.0)


def make_agent(sqlite_session_factory) -> FireDangerAssessmentAgent:
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    return FireDangerAssessmentAgent(
        input_service=FireDangerInputService(weather_repository=weather_repository),
        calculator=FFWICalculator(),
        repository=FireDangerAssessmentRepository(session_factory=sqlite_session_factory),
    )


def save_weather_observation(
    sqlite_session_factory,
    *,
    external_station_id: int,
    area: AssessmentArea,
    timestamp: datetime,
    temperature: float | None,
    relative_humidity: float | None,
    wind_speed: float | None,
    latitude_offset: float = 0.0,
) -> tuple[int, int]:
    repository = WeatherRepository(session_factory=sqlite_session_factory)
    repository.save_station(
        WeatherStation(
            external_station_id=external_station_id,
            name=f"Acceptance Station {external_station_id}",
            latitude=area.latitude + latitude_offset,
            longitude=area.longitude,
        )
    )
    repository.save_observation(
        WeatherObservation(
            station_external_id=external_station_id,
            timestamp=timestamp,
            temperature=temperature,
            relative_humidity=relative_humidity,
            wind_speed=wind_speed,
        )
    )
    session = sqlite_session_factory()
    row = (
        session.execute(
            select(WeatherObservationDB.id, WeatherStationDB.id)
            .join(WeatherStationDB, WeatherObservationDB.station_id == WeatherStationDB.id)
            .where(WeatherStationDB.external_station_id == external_station_id)
            .order_by(WeatherObservationDB.id.desc())
        )
        .first()
    )
    session.close()
    return row[0], row[1]


def test_real_pipeline_scores_hot_dry_windy_weather_higher_than_cool_humid_calm_weather(
    sqlite_session_factory,
):
    save_weather_observation(
        sqlite_session_factory,
        external_station_id=910001,
        area=LOW_AREA,
        timestamp=AS_OF,
        temperature=18.0,
        relative_humidity=75.0,
        wind_speed=3.0,
    )
    save_weather_observation(
        sqlite_session_factory,
        external_station_id=920001,
        area=HIGH_AREA,
        timestamp=AS_OF,
        temperature=41.0,
        relative_humidity=8.0,
        wind_speed=45.0,
    )
    agent = make_agent(sqlite_session_factory)

    low_result = agent.assess(area=LOW_AREA, as_of=AS_OF)
    high_result = agent.assess(area=HIGH_AREA, as_of=AS_OF)

    assert low_result.success is True
    assert high_result.success is True
    assert low_result.assessment.status is FireDangerAssessmentStatus.VALID
    assert high_result.assessment.status is FireDangerAssessmentStatus.VALID
    assert high_result.assessment.score > low_result.assessment.score
    assert _level_rank(high_result.assessment.level) >= _level_rank(low_result.assessment.level)


def test_traceability_records_only_exact_weather_observations_used(sqlite_session_factory):
    valid_observation_id, valid_station_id = save_weather_observation(
        sqlite_session_factory,
        external_station_id=930001,
        area=LOW_AREA,
        timestamp=AS_OF - timedelta(minutes=5),
        temperature=30.0,
        relative_humidity=25.0,
        wind_speed=12.0,
    )
    stale_observation_id, _ = save_weather_observation(
        sqlite_session_factory,
        external_station_id=930002,
        area=LOW_AREA,
        timestamp=AS_OF - timedelta(minutes=31),
        temperature=42.0,
        relative_humidity=5.0,
        wind_speed=50.0,
    )
    incomplete_observation_id, _ = save_weather_observation(
        sqlite_session_factory,
        external_station_id=930003,
        area=LOW_AREA,
        timestamp=AS_OF - timedelta(minutes=4),
        temperature=None,
        relative_humidity=10.0,
        wind_speed=35.0,
    )
    outside_observation_id, _ = save_weather_observation(
        sqlite_session_factory,
        external_station_id=930004,
        area=LOW_AREA,
        timestamp=AS_OF - timedelta(minutes=3),
        temperature=40.0,
        relative_humidity=8.0,
        wind_speed=45.0,
        latitude_offset=1.0,
    )

    result = make_agent(sqlite_session_factory).assess(area=LOW_AREA, as_of=AS_OF)

    assert result.success is True
    assert result.assessment.status is FireDangerAssessmentStatus.VALID
    session = sqlite_session_factory()
    trace_rows = (
        session.execute(
            select(FireDangerAssessmentWeatherInputDB)
            .where(FireDangerAssessmentWeatherInputDB.assessment_id == result.stored_assessment_id)
            .order_by(FireDangerAssessmentWeatherInputDB.id.asc())
        )
        .scalars()
        .all()
    )
    session.close()
    assert [row.weather_observation_id for row in trace_rows] == [valid_observation_id]
    assert [row.station_id for row in trace_rows] == [valid_station_id]
    assert stale_observation_id not in [row.weather_observation_id for row in trace_rows]
    assert incomplete_observation_id not in [row.weather_observation_id for row in trace_rows]
    assert outside_observation_id not in [row.weather_observation_id for row in trace_rows]


def test_simulation_low_risk_profile_scores_lower_than_high_risk_profile(sqlite_session_factory):
    low_scenario = build_scenario(
        scenario_type=ScenarioType.LOW_RISK_NO_FIRE,
        location=CARMEL_LOCATION,
        seed=42,
        incident_id="acceptance-low-risk-incident",
    )
    high_scenario = build_scenario(
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=CARMEL_LOCATION,
        seed=42,
        incident_id="acceptance-high-risk-incident",
    )
    weather_repository = WeatherRepository(session_factory=sqlite_session_factory)
    executor = SimulationEventExecutor(weather_repository=weather_repository)
    coordinator = SimulationFireDangerCoordinator(assessment_agent=make_agent(sqlite_session_factory))
    low_result = _execute_first_weather_event(low_scenario, executor, coordinator, AS_OF)
    high_result = _execute_first_weather_event(
        high_scenario,
        executor,
        coordinator,
        AS_OF + timedelta(hours=1),
    )

    assert low_result.assessment_result.success is True
    assert high_result.assessment_result.success is True
    assert low_result.assessment_result.assessment.status is FireDangerAssessmentStatus.VALID
    assert high_result.assessment_result.assessment.status is FireDangerAssessmentStatus.VALID
    assert high_result.assessment_result.assessment.score > low_result.assessment_result.assessment.score
    assert _level_rank(high_result.assessment_result.assessment.level) >= _level_rank(
        low_result.assessment_result.assessment.level
    )


def _execute_first_weather_event(
    scenario,
    executor: SimulationEventExecutor,
    coordinator: SimulationFireDangerCoordinator,
    scenario_started_at: datetime,
):
    event = scenario.events[0]
    event_timestamp = simulation_event_timestamp(scenario_started_at, event)
    execution_result: SimulationEventExecutionResult = executor.execute(
        scenario=scenario,
        event=event,
        event_timestamp=event_timestamp,
    )
    assert execution_result.saved_count > 0
    return coordinator.handle_event(
        scenario=scenario,
        event=event,
        execution_result=execution_result,
        event_timestamp=event_timestamp,
    )


def _level_rank(level) -> int:
    return {
        "low": 0,
        "moderate": 1,
        "high": 2,
        "very_high": 3,
        "extreme": 4,
    }[level.value]
