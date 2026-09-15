"""Acceptance tests for User Story 4.4: operational refresh orchestration."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from src.agents.analysis import ResponseTargetGenerationAgent
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.calculators.response_target import ResponseTargetCalculator
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadEffectiveState,
    FireSpreadFuelClass,
    FireSpreadInput,
    FireSpreadInputResult,
    FireSpreadInputStatus,
    FireSpreadPrediction,
    FireSpreadPredictionCell,
    FireSpreadPredictionStatus,
    OperationalRefreshTriggerType,
    ResponseTargetType,
    SatelliteHotspot,
)
from src.models.resource_status import ResourceStatus
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_spread import FireSpreadInputService
from src.services.operational_refresh import (
    FireSpreadRefreshOrchestrator,
    FireSpreadRefreshHorizonStatus,
    OperationalRefreshOrchestrator,
    OperationalRefreshStatus,
    ResourceStatusUpdateService,
)
from src.services.response_target import ResponseTargetInputService
from src.simulation import (
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventType,
    SimulationResourceStatusChange,
    build_active_fire_resource_refresh_scenario,
)
from src.simulation.analysis import SimulationRefreshCoordinator

AS_OF = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
CARMEL_LATITUDE = 32.731
CARMEL_LONGITUDE = 35.046
GOLAN_LATITUDE = 33.127
GOLAN_LONGITUDE = 35.781


@dataclass
class AcceptanceStack:
    fire_events: FireEventRepository
    severity: FireSeverityAssessmentRepository
    spread: FireSpreadPredictionRepository
    targets: ResponseTargetRepository
    weather: WeatherRepository
    satellite: SatelliteHotspotRepository
    resources: FirefightingResourceRepository
    severity_agent: "PersistingSeverityAgent"
    spread_input: FireSpreadInputService
    spread_agent: "CountingSpreadPredictionAgent"
    target_agent: ResponseTargetGenerationAgent
    orchestrator: OperationalRefreshOrchestrator


class PersistingSeverityAgent:
    """Small deterministic stand-in for the already-tested Severity agent.

    It persists real FireSeverityAssessment rows with exact weather/satellite
    traceability so the refresh stack can exercise the production spread input
    service and repositories without live Copernicus/IMS/FIRMS calls.
    """

    def __init__(self, repository: FireSeverityAssessmentRepository) -> None:
        self._repository = repository
        self.weather_ids_by_event: dict[int, tuple[int, ...]] = {}
        self.satellite_ids_by_event: dict[int, tuple[int, ...]] = {}
        self.score_by_event: dict[int, float] = {}
        self.calls: list[tuple[int, datetime]] = []

    def configure(
        self,
        fire_event_id: int,
        *,
        weather_observation_id: int,
        satellite_hotspot_id: int,
        score: float = 70.0,
    ) -> None:
        self.weather_ids_by_event[fire_event_id] = (weather_observation_id,)
        self.satellite_ids_by_event[fire_event_id] = (satellite_hotspot_id,)
        self.score_by_event[fire_event_id] = score

    def assess(self, fire_event_id: int, assessed_at: datetime):
        self.calls.append((fire_event_id, assessed_at))
        weather_ids = self.weather_ids_by_event[fire_event_id]
        satellite_ids = self.satellite_ids_by_event[fire_event_id]
        score = self.score_by_event.get(fire_event_id, 70.0)
        return self._repository.save_assessment(
            make_severity(fire_event_id=fire_event_id, assessed_at=assessed_at, score=score),
            weather_observation_ids=weather_ids,
            satellite_hotspot_ids=satellite_ids,
            selected_frp_hotspot_id=satellite_ids[0],
        )


class CountingSpreadPredictionAgent:
    def __init__(self, input_service: FireSpreadInputService, repository: FireSpreadPredictionRepository) -> None:
        self._input_service = input_service
        self._repository = repository
        self.calls: list[dict[str, object]] = []

    def predict_from_input_result(
        self,
        *,
        input_result: FireSpreadInputResult,
        as_of: datetime,
        horizon_minutes: int,
        effective_state_fingerprint: str | None = None,
    ):
        self.calls.append(
            {
                "input_result": input_result,
                "as_of": as_of,
                "horizon_minutes": horizon_minutes,
                "effective_state_fingerprint": effective_state_fingerprint,
            }
        )
        if input_result.status is FireSpreadInputStatus.READY:
            spread_input = input_result.input_data
            prediction = FireSpreadPrediction(
                fire_event_id=input_result.fire_event_id,
                severity_assessment_id=input_result.severity_assessment_id,
                predicted_at=as_of,
                horizon_minutes=horizon_minutes,
                status=FireSpreadPredictionStatus.VALID,
                methodology=METHODOLOGY_NAME,
                methodology_version=METHODOLOGY_VERSION,
                cells=(
                    make_cell(
                        spread_input.origin_latitude + (0.001 * (horizon_minutes // 30)),
                        spread_input.origin_longitude + (0.001 * (horizon_minutes // 30)),
                        85.0 if horizon_minutes == 30 else 72.0,
                    ),
                ),
                effective_state_fingerprint=effective_state_fingerprint,
            )
        else:
            status = {
                FireSpreadInputStatus.INSUFFICIENT_DATA: FireSpreadPredictionStatus.INSUFFICIENT_DATA,
                FireSpreadInputStatus.INACTIVE_EVENT: FireSpreadPredictionStatus.INACTIVE_EVENT,
            }[input_result.status]
            prediction = FireSpreadPrediction(
                fire_event_id=input_result.fire_event_id,
                severity_assessment_id=input_result.severity_assessment_id,
                predicted_at=as_of,
                horizon_minutes=horizon_minutes,
                status=status,
                methodology=METHODOLOGY_NAME,
                methodology_version=METHODOLOGY_VERSION,
                cells=(),
            )
        return self._repository.save_prediction(prediction, input_result.weather_observation_id)


class CountingInputService:
    def __init__(self, results: dict[int, FireSpreadInputResult]) -> None:
        self.results = results
        self.calls: list[int] = []

    def prepare_input(self, *, fire_event_id: int, as_of: datetime, horizon_minutes: int) -> FireSpreadInputResult:
        self.calls.append(horizon_minutes)
        return self.results[horizon_minutes]


@pytest.fixture
def stack(sqlite_session_factory) -> AcceptanceStack:
    fire_events = FireEventRepository(sqlite_session_factory)
    severity = FireSeverityAssessmentRepository(sqlite_session_factory)
    spread = FireSpreadPredictionRepository(sqlite_session_factory)
    targets = ResponseTargetRepository(sqlite_session_factory)
    weather = WeatherRepository(sqlite_session_factory)
    satellite = SatelliteHotspotRepository(sqlite_session_factory)
    resources = FirefightingResourceRepository(sqlite_session_factory)
    severity_agent = PersistingSeverityAgent(severity)
    spread_input = FireSpreadInputService(fire_events, severity, weather)
    spread_agent = CountingSpreadPredictionAgent(spread_input, spread)
    spread_refresh = FireSpreadRefreshOrchestrator(
        input_service=spread_input,
        prediction_agent=spread_agent,
        prediction_repository=spread,
    )
    target_agent = ResponseTargetGenerationAgent(
        input_service=ResponseTargetInputService(fire_events, severity, spread),
        calculator=ResponseTargetCalculator(),
        repository=targets,
    )
    orchestrator = OperationalRefreshOrchestrator(
        severity_agent=severity_agent,
        spread_refresh_orchestrator=spread_refresh,
        response_target_agent=target_agent,
        resource_status_service=ResourceStatusUpdateService(resources),
        resource_repository=resources,
        fire_event_repository=fire_events,
    )
    return AcceptanceStack(
        fire_events=fire_events,
        severity=severity,
        spread=spread,
        targets=targets,
        weather=weather,
        satellite=satellite,
        resources=resources,
        severity_agent=severity_agent,
        spread_input=spread_input,
        spread_agent=spread_agent,
        target_agent=target_agent,
        orchestrator=orchestrator,
    )


def test_at1_weather_update_creates_append_only_prediction_through_severity_trace(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_1 = persist_weather(stack, station_offset=1, wind_speed=4.0, humidity=34.0)
    weather_2 = persist_weather(stack, station_offset=2, wind_speed=9.0, humidity=34.0, timestamp=AS_OF + timedelta(minutes=1))
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_1, satellite_hotspot_id=satellite_id)

    first = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )
    first_ids = prediction_ids(sqlite_session_factory, fire_event_id)
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_2, satellite_hotspot_id=satellite_id)
    second = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )
    second_ids = prediction_ids(sqlite_session_factory, fire_event_id)

    assert first.success is True
    assert second.success is True
    assert len(first_ids) == 2
    assert len(second_ids) == 4
    assert set(first_ids).issubset(second_ids)
    latest_30 = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF + timedelta(minutes=1))
    latest_severity = stack.severity.get_latest_for_event(fire_event_id)
    assert latest_30.prediction.severity_assessment_id == latest_severity.assessment_id
    assert latest_30.weather_observation_id == weather_2
    assert latest_severity.weather_observation_ids == (weather_2,)


def test_at2_fire_event_location_change_refreshes_and_confidence_only_change_no_ops(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_id = persist_weather(stack, station_offset=10)
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_id, satellite_hotspot_id=satellite_id)
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )
    before_location_change = count_predictions(sqlite_session_factory, fire_event_id)
    update_fire_event(stack, fire_event_id, latitude=CARMEL_LATITUDE + 0.03, confidence=0.91)

    location_result = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )
    after_location_change = count_predictions(sqlite_session_factory, fire_event_id)
    update_fire_event(stack, fire_event_id, latitude=CARMEL_LATITUDE + 0.03, confidence=0.42)
    confidence_result = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF + timedelta(minutes=2),
    )

    assert location_result.spread_refresh_result.predictions_created == 2
    assert after_location_change == before_location_change + 2
    assert confidence_result.spread_refresh_result.no_ops == 2
    assert count_predictions(sqlite_session_factory, fire_event_id) == after_location_change


def test_at3_severity_update_uses_new_trace_and_score_only_no_op_still_refreshes_targets(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_1 = persist_weather(stack, station_offset=20, wind_speed=4.0)
    weather_2 = persist_weather(stack, station_offset=21, wind_speed=11.0, timestamp=AS_OF + timedelta(minutes=1))
    s1 = persist_severity(stack, fire_event_id, weather_1, satellite_id, assessed_at=AS_OF - timedelta(minutes=1))
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
    )
    s2 = persist_severity(stack, fire_event_id, weather_2, satellite_id, assessed_at=AS_OF + timedelta(minutes=1), score=80.0)

    result = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )
    latest = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF + timedelta(minutes=1))
    target_count = count_target_sets(sqlite_session_factory, fire_event_id)
    prediction_count = count_predictions(sqlite_session_factory, fire_event_id)
    s3 = persist_severity(stack, fire_event_id, weather_2, satellite_id, assessed_at=AS_OF + timedelta(minutes=2), score=95.0)
    score_only = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF + timedelta(minutes=2),
    )

    assert s1 != s2
    assert latest.prediction.severity_assessment_id == s2
    assert latest.weather_observation_id == weather_2
    assert result.spread_refresh_result.predictions_created == 2
    assert score_only.spread_refresh_result.no_ops == 2
    assert count_predictions(sqlite_session_factory, fire_event_id) == prediction_count
    assert count_target_sets(sqlite_session_factory, fire_event_id) == target_count + 1
    assert stack.severity.get_latest_for_event(fire_event_id).assessment_id == s3


def test_at4_at5_resource_status_simulation_removes_and_restores_same_truck(stack, sqlite_session_factory):
    insert_station_resources(sqlite_session_factory, status=ResourceStatus.AVAILABLE)
    coordinator = SimulationRefreshCoordinator(
        operational_refresh_orchestrator=stack.orchestrator,
        fire_event_repository=stack.fire_events,
    )
    assert available_resource_ids(stack) == ["TRUCK-A", "TRUCK-B"]

    unavailable = resource_event("TRUCK-A", ResourceStatus.UNAVAILABLE)
    first = coordinator.handle_event(
        scenario=build_active_fire_resource_refresh_scenario(),
        event=unavailable,
        execution_result=resource_execution(unavailable),
        event_timestamp=AS_OF,
    )
    available = resource_event("TRUCK-A", ResourceStatus.AVAILABLE)
    second = coordinator.handle_event(
        scenario=build_active_fire_resource_refresh_scenario(),
        event=available,
        execution_result=resource_execution(available),
        event_timestamp=AS_OF + timedelta(seconds=30),
    )

    assert first.refresh_results[0].status is OperationalRefreshStatus.RESOURCE_UPDATED
    assert stack.resources.get_by_id("TRUCK-A").status is ResourceStatus.AVAILABLE
    assert [result.resource_status_result.resource_id for result in (first.refresh_results[0], second.refresh_results[0])] == [
        "TRUCK-A",
        "TRUCK-A",
    ]
    assert available_resource_ids(stack) == ["TRUCK-A", "TRUCK-B"]
    assert count_resources(sqlite_session_factory) == 2


def test_at5_additional_assigned_available_transition_uses_authoritative_available_set(stack, sqlite_session_factory):
    insert_station_resources(sqlite_session_factory, status=ResourceStatus.AVAILABLE)

    assigned = stack.orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.ASSIGNED)
    restored = stack.orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.AVAILABLE)

    assert assigned.available_resources and [resource.id for resource in assigned.available_resources] == ["TRUCK-B"]
    assert restored.available_resources and [resource.id for resource in restored.available_resources] == [
        "TRUCK-A",
        "TRUCK-B",
    ]


def test_at6_resource_change_does_not_touch_detection_or_environmental_history(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack, confidence=0.77)
    insert_station_resources(sqlite_session_factory, status=ResourceStatus.AVAILABLE)
    before_event = stack.fire_events.get_by_id(fire_event_id).event
    before_counts = (
        count_predictions(sqlite_session_factory, fire_event_id),
        count_severity(sqlite_session_factory, fire_event_id),
        count_target_sets(sqlite_session_factory, fire_event_id),
    )

    result = stack.orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)
    after_event = stack.fire_events.get_by_id(fire_event_id).event

    assert result.status is OperationalRefreshStatus.RESOURCE_UPDATED
    assert after_event.detection_confidence == before_event.detection_confidence
    assert after_event.status is before_event.status
    assert (after_event.latitude, after_event.longitude) == (before_event.latitude, before_event.longitude)
    assert stack.severity_agent.calls == []
    assert stack.spread_agent.calls == []
    assert before_counts == (
        count_predictions(sqlite_session_factory, fire_event_id),
        count_severity(sqlite_session_factory, fire_event_id),
        count_target_sets(sqlite_session_factory, fire_event_id),
    )


def test_at7_previous_predictions_and_cells_remain_after_changed_refresh(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_1 = persist_weather(stack, station_offset=30, wind_speed=4.0)
    weather_2 = persist_weather(stack, station_offset=31, wind_speed=12.0, timestamp=AS_OF + timedelta(minutes=1))
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_1, satellite_hotspot_id=satellite_id)
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )
    p1 = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF)
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_2, satellite_hotspot_id=satellite_id)
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )
    p2 = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF + timedelta(minutes=1))

    assert p1.id != p2.id
    assert stack.spread.get_by_id(p1.id).id == p1.id
    assert prediction_cell_count(sqlite_session_factory, p1.id) > 0
    assert prediction_cell_count(sqlite_session_factory, p2.id) > 0
    assert prediction_ids(sqlite_session_factory, fire_event_id) == sorted(prediction_ids(sqlite_session_factory, fire_event_id))


def test_at8_resource_update_is_unrelated_to_spread_recalculation(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    insert_station_resources(sqlite_session_factory, status=ResourceStatus.AVAILABLE)
    prediction_count = count_predictions(sqlite_session_factory, fire_event_id)

    stack.orchestrator.refresh_resource(resource_id="TRUCK-A", new_status=ResourceStatus.UNAVAILABLE)

    assert count_predictions(sqlite_session_factory, fire_event_id) == prediction_count
    assert stack.spread_agent.calls == []


def test_at9_identical_effective_input_and_new_source_ids_do_not_duplicate_predictions(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    input_data = make_spread_input()
    fingerprint = FireSpreadEffectiveState.from_input(fire_event_id=fire_event_id, spread_input=input_data).fingerprint
    repository = FireSpreadPredictionRepository(sqlite_session_factory)
    weather_1 = persist_weather_with_repo(WeatherRepository(sqlite_session_factory), station_offset=40)
    weather_2 = persist_weather_with_repo(WeatherRepository(sqlite_session_factory), station_offset=41)
    repository.save_prediction(
        FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=persist_severity(
                stack,
                fire_event_id,
                weather_1,
                persist_hotspot(stack, fire_event_id),
                assessed_at=AS_OF - timedelta(minutes=1),
            ),
            predicted_at=AS_OF,
            horizon_minutes=30,
            status=FireSpreadPredictionStatus.VALID,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=(make_cell(32.75, 35.07, 75.0),),
            effective_state_fingerprint=fingerprint,
        ),
        weather_observation_id=weather_1,
    )
    input_service = CountingInputService(
        {
            30: FireSpreadInputResult(
                status=FireSpreadInputStatus.READY,
                input_data=input_data,
                fire_event_id=fire_event_id,
                severity_assessment_id=101,
                weather_observation_id=weather_2,
            )
        }
    )
    agent = CountingSpreadPredictionAgent(input_service, repository)
    refresh = FireSpreadRefreshOrchestrator(
        input_service=input_service,
        prediction_agent=agent,
        prediction_repository=repository,
    )

    result = refresh.refresh(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.NO_OP
    assert count_predictions(sqlite_session_factory, fire_event_id) == 1
    assert agent.calls == []


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_at10_resolved_or_dismissed_event_persists_inactive_state_without_active_targets(
    stack,
    sqlite_session_factory,
    status,
):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_id = persist_weather(stack, station_offset=50)
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_id, satellite_hotspot_id=satellite_id)
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )
    target_history_before = stack.targets.get_history_for_event(fire_event_id)
    update_fire_event(stack, fire_event_id, status=status)

    result = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )

    assert result.severity_result is None
    assert {item.status for item in result.spread_refresh_result.horizon_results} == {
        FireSpreadRefreshHorizonStatus.INACTIVE_EVENT
    }
    assert result.response_target_result.target_set_id is None
    assert stack.targets.get_history_for_event(fire_event_id) == target_history_before


def test_at11_response_targets_use_latest_valid_prediction_and_do_not_reuse_stale_cells(stack, sqlite_session_factory):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_1 = persist_weather(stack, station_offset=60, wind_speed=4.0)
    weather_2 = persist_weather(stack, station_offset=61, wind_speed=12.0, timestamp=AS_OF + timedelta(minutes=1))
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_1, satellite_hotspot_id=satellite_id)
    first = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
    )
    t1 = first.response_target_result.target_set_id
    p1 = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF)
    stack.severity_agent.configure(fire_event_id, weather_observation_id=weather_2, satellite_hotspot_id=satellite_id)
    second = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )
    t2 = second.response_target_result.target_set_id
    p2 = stack.spread.get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF + timedelta(minutes=1))
    latest_targets = stack.targets.get_by_id(t2).targets

    assert t1 != t2
    assert stack.targets.get_by_id(t1) is not None
    assert p1.id != p2.id
    predicted_targets = [target.target for target in latest_targets if target.target.target_type is ResponseTargetType.PREDICTED_RISK]
    latest_prediction_ids = {
        horizon.prediction_id
        for horizon in second.spread_refresh_result.horizon_results
        if horizon.prediction_id is not None
    }
    assert predicted_targets
    assert p2.id in latest_prediction_ids
    assert {target.spread_prediction_id for target in predicted_targets} == latest_prediction_ids
    assert p1.id not in {target.spread_prediction_id for target in predicted_targets}


def test_at11_transition_to_insufficient_does_not_reuse_stale_predicted_risk_targets(stack):
    fire_event_id = persist_fire_event(stack)
    satellite_id = persist_hotspot(stack, fire_event_id)
    weather_id = persist_weather(stack, station_offset=70)
    persist_severity(stack, fire_event_id, weather_id, satellite_id, assessed_at=AS_OF - timedelta(minutes=1))
    stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
    )
    persist_severity(
        stack,
        fire_event_id,
        weather_id,
        satellite_id,
        assessed_at=AS_OF + timedelta(minutes=1),
        status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    result = stack.orchestrator.refresh_fire_event(
        fire_event_id=fire_event_id,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF + timedelta(minutes=1),
    )

    assert result.response_target_result.target_count == 1
    assert result.response_target_result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert all(target.target_type is not ResponseTargetType.PREDICTED_RISK for target in result.response_target_result.targets)


def test_ac1_ac4_ac8_architecture_guardrails_and_documented_trigger_matrix():
    production_files = [
        Path("backend/src/services/operational_refresh/operational_refresh_orchestrator.py"),
        Path("backend/src/services/operational_refresh/fire_spread_refresh_orchestrator.py"),
        Path("backend/src/services/operational_refresh/operational_refresh_policy.py"),
    ]
    forbidden_imports = (
        "FireSpreadCalculator",
        "FireSeverityCalculator",
        "ResponseTargetCalculator",
        "RoadNetworkRepository",
        "GraphNode",
        "GraphEdge",
        "osmnx",
        "routing",
        "Dijkstra",
        "genetic",
        "ResponsePlan",
    )
    violations = []
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if any(fragment in module for fragment in forbidden_imports):
                violations.append((str(path), module))

    assert violations == []
    assert Path("backend/docs/operational_refresh.md").exists()


def test_ac5_resource_simulation_timing_and_selection_are_deterministic():
    first = build_active_fire_resource_refresh_scenario(seed=42)
    second = build_active_fire_resource_refresh_scenario(seed=42)
    first_resource_events = [event for event in first.events if event.event_type is SimulationEventType.RESOURCE_STATUS]
    second_resource_events = [event for event in second.events if event.event_type is SimulationEventType.RESOURCE_STATUS]

    assert [event.offset_seconds for event in first_resource_events] == [60, 90]
    assert [event.resource_status_change.new_status for event in first_resource_events] == [
        ResourceStatus.UNAVAILABLE,
        ResourceStatus.AVAILABLE,
    ]
    assert [event.resource_status_change.selection_key for event in first_resource_events] == [
        "incident-active-fire-resource-refresh-carmel-01:primary-resource",
        "incident-active-fire-resource-refresh-carmel-01:primary-resource",
    ]
    assert first_resource_events == second_resource_events


def test_ac6_multi_incident_dedupe_keeps_histories_independent(stack, sqlite_session_factory):
    carmel_id = persist_fire_event(stack, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    golan_id = persist_fire_event(stack, latitude=GOLAN_LATITUDE, longitude=GOLAN_LONGITUDE)
    satellite_carmel = persist_hotspot(stack, carmel_id, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    satellite_golan = persist_hotspot(stack, golan_id, latitude=GOLAN_LATITUDE, longitude=GOLAN_LONGITUDE)
    weather_carmel = persist_weather(stack, station_offset=80, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    weather_golan = persist_weather(stack, station_offset=81, latitude=GOLAN_LATITUDE, longitude=GOLAN_LONGITUDE)
    stack.severity_agent.configure(carmel_id, weather_observation_id=weather_carmel, satellite_hotspot_id=satellite_carmel)
    stack.severity_agent.configure(golan_id, weather_observation_id=weather_golan, satellite_hotspot_id=satellite_golan)
    coordinator = SimulationRefreshCoordinator(
        operational_refresh_orchestrator=stack.orchestrator,
        fire_event_repository=stack.fire_events,
    )

    result = coordinator.refresh_fire_events(
        fire_event_ids=(golan_id, carmel_id, golan_id),
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
    )

    assert result.fire_event_ids == tuple(sorted((carmel_id, golan_id)))
    assert count_predictions(sqlite_session_factory, carmel_id) == 2
    assert count_predictions(sqlite_session_factory, golan_id) == 2
    assert count_target_sets(sqlite_session_factory, carmel_id) == 1
    assert count_target_sets(sqlite_session_factory, golan_id) == 1


def persist_fire_event(
    stack: AcceptanceStack,
    *,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    confidence: float = 0.88,
) -> int:
    hotspot_id = persist_hotspot(stack, 0, latitude=latitude, longitude=longitude)
    return stack.fire_events.create_event(
        FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=AS_OF - timedelta(minutes=30),
            updated_at=AS_OF - timedelta(minutes=5),
            status=FireEventStatus.CONFIRMED,
            detection_confidence=confidence,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    ).id


def update_fire_event(
    stack: AcceptanceStack,
    fire_event_id: int,
    *,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    confidence: float = 0.88,
    status: FireEventStatus = FireEventStatus.CONFIRMED,
) -> None:
    stored = stack.fire_events.get_by_id(fire_event_id)
    stack.fire_events.update_event(
        fire_event_id,
        FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=stored.event.detected_at,
            updated_at=AS_OF,
            status=status,
            detection_confidence=confidence,
            methodology=stored.event.methodology,
            methodology_version=stored.event.methodology_version,
        ),
    )


def persist_hotspot(
    stack: AcceptanceStack,
    fire_event_id: int,
    *,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
) -> int:
    stack.satellite.save_hotspot(
        SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=AS_OF - timedelta(minutes=10),
            confidence="h",
            frp=80.0 + fire_event_id,
            satellite="N20",
        )
    )
    return max(record.id for record in stack.satellite.get_recent_hotspots(as_of=AS_OF, lookback_minutes=120))


def persist_weather(
    stack: AcceptanceStack,
    *,
    station_offset: int,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    wind_speed: float = 5.0,
    humidity: float = 35.0,
    timestamp: datetime = AS_OF,
) -> int:
    return persist_weather_with_repo(
        stack.weather,
        station_offset=station_offset,
        latitude=latitude,
        longitude=longitude,
        wind_speed=wind_speed,
        humidity=humidity,
        timestamp=timestamp,
    )


def persist_weather_with_repo(
    repository: WeatherRepository,
    *,
    station_offset: int,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    wind_speed: float = 5.0,
    humidity: float = 35.0,
    timestamp: datetime = AS_OF,
) -> int:
    station_id = 980000 + station_offset
    repository.save_station(
        WeatherStation(
            external_station_id=station_id,
            name=f"US44 Station {station_offset}",
            latitude=latitude,
            longitude=longitude,
        )
    )
    repository.save_observation(
        WeatherObservation(
            station_external_id=station_id,
            timestamp=timestamp,
            temperature=30.0,
            relative_humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=270.0,
        )
    )
    candidates = repository.get_recent_observations_for_area_candidates(
        latitude=latitude,
        longitude=longitude,
        radius_km=5.0,
        start_time=timestamp - timedelta(minutes=1),
        end_time=timestamp + timedelta(minutes=1),
    )
    return next(
        candidate.observation_id
        for candidate in candidates
        if candidate.observation.station_external_id == station_id
    )


def make_severity(
    *,
    fire_event_id: int,
    assessed_at: datetime,
    score: float | None = 70.0,
    status: FireSeverityAssessmentStatus = FireSeverityAssessmentStatus.VALID,
    level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
) -> FireSeverityAssessment:
    return FireSeverityAssessment(
        fire_event_id=fire_event_id,
        assessed_at=assessed_at,
        status=status,
        score=score,
        level=level,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API" if status is FireSeverityAssessmentStatus.VALID else None,
        vegetation_dataset_year=2019 if status is FireSeverityAssessmentStatus.VALID else None,
        vegetation_radius_km=1.0 if status is FireSeverityAssessmentStatus.VALID else None,
        vegetation_dominant_land_cover="Shrub cover" if status is FireSeverityAssessmentStatus.VALID else None,
        vegetation_fuel_score=0.8 if status is FireSeverityAssessmentStatus.VALID else None,
    )


def persist_severity(
    stack: AcceptanceStack,
    fire_event_id: int,
    weather_id: int,
    satellite_id: int,
    *,
    assessed_at: datetime,
    score: float | None = 70.0,
    status: FireSeverityAssessmentStatus = FireSeverityAssessmentStatus.VALID,
    level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
) -> int:
    weather_ids = (weather_id,) if status is FireSeverityAssessmentStatus.VALID else ()
    satellite_ids = (satellite_id,) if status is FireSeverityAssessmentStatus.VALID else ()
    return stack.severity.save_assessment(
        make_severity(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            score=score,
            status=status,
            level=level,
        ),
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=satellite_id if status is FireSeverityAssessmentStatus.VALID else None,
    ).assessment_id


def make_spread_input() -> FireSpreadInput:
    return FireSpreadInput(
        origin_latitude=CARMEL_LATITUDE,
        origin_longitude=CARMEL_LONGITUDE,
        wind_speed_kmh=18.0,
        wind_direction_deg=270.0,
        fuel_moisture_percent=12.0,
        fuel_class=FireSpreadFuelClass.SHRUBS,
        horizon_minutes=30,
    )


def make_cell(latitude: float, longitude: float, risk: float) -> FireSpreadPredictionCell:
    return FireSpreadPredictionCell(
        latitude=latitude,
        longitude=longitude,
        spread_probability=risk / 100.0,
        spread_risk_score=risk,
        reached_step=1,
        reached_minutes=5,
    )


def insert_station_resources(sqlite_session_factory, *, status: ResourceStatus) -> None:
    session = sqlite_session_factory()
    session.add(FireStationDB(id="STATION-1", name="Station 1", latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE))
    session.flush()
    session.add_all(
        [
            FirefightingResourceDB(id="TRUCK-A", station_id="STATION-1", status=status),
            FirefightingResourceDB(id="TRUCK-B", station_id="STATION-1", status=ResourceStatus.AVAILABLE),
        ]
    )
    session.commit()
    session.close()


def resource_event(resource_id: str, status: ResourceStatus) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=60,
        event_type=SimulationEventType.RESOURCE_STATUS,
        incident_id="incident-active-fire-resource-refresh-carmel-01",
        source_event_index=0,
        resource_status_change=SimulationResourceStatusChange(resource_id=resource_id, new_status=status),
    )


def resource_execution(event: SimulationEvent) -> SimulationEventExecutionResult:
    return SimulationEventExecutionResult(
        event=event,
        success=True,
        generated_count=1,
        saved_count=1,
        duplicates_skipped=0,
        failed_count=0,
    )


def available_resource_ids(stack: AcceptanceStack) -> list[str]:
    return [resource.id for resource in stack.resources.get_available_resources(["STATION-1"])]


def count_predictions(sqlite_session_factory, fire_event_id: int) -> int:
    session = sqlite_session_factory()
    try:
        return session.execute(
            select(func.count()).select_from(FireSpreadPredictionDB).where(FireSpreadPredictionDB.fire_event_id == fire_event_id)
        ).scalar_one()
    finally:
        session.close()


def prediction_ids(sqlite_session_factory, fire_event_id: int) -> list[int]:
    session = sqlite_session_factory()
    try:
        return list(
            session.execute(
                select(FireSpreadPredictionDB.id)
                .where(FireSpreadPredictionDB.fire_event_id == fire_event_id)
                .order_by(FireSpreadPredictionDB.id.asc())
            ).scalars()
        )
    finally:
        session.close()


def prediction_cell_count(sqlite_session_factory, prediction_id: int) -> int:
    session = sqlite_session_factory()
    try:
        return session.execute(
            select(func.count()).select_from(FireSpreadPredictionCellDB).where(FireSpreadPredictionCellDB.prediction_id == prediction_id)
        ).scalar_one()
    finally:
        session.close()


def count_target_sets(sqlite_session_factory, fire_event_id: int) -> int:
    session = sqlite_session_factory()
    try:
        return session.execute(
            select(func.count()).select_from(ResponseTargetSetDB).where(ResponseTargetSetDB.fire_event_id == fire_event_id)
        ).scalar_one()
    finally:
        session.close()


def count_severity(sqlite_session_factory, fire_event_id: int) -> int:
    from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB

    session = sqlite_session_factory()
    try:
        return session.execute(
            select(func.count()).select_from(FireSeverityAssessmentDB).where(FireSeverityAssessmentDB.fire_event_id == fire_event_id)
        ).scalar_one()
    finally:
        session.close()


def count_resources(sqlite_session_factory) -> int:
    session = sqlite_session_factory()
    try:
        return session.execute(select(func.count()).select_from(FirefightingResourceDB)).scalar_one()
    finally:
        session.close()
