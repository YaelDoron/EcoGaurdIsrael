"""Tests for ResponseTargetGenerationAgent orchestration."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.agents.analysis import (
    ResponseTargetGenerationAgent,
    ResponseTargetGenerationStatus,
)
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
from src.calculators.response_target.response_target_config import (
    MIN_PREDICTED_TARGET_RISK_SCORE,
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadPrediction,
    FireSpreadPredictionCell,
    FireSpreadPredictionStatus,
    PredictedRiskTargetCandidate,
    ResponseTarget,
    ResponseTargetInput,
    ResponseTargetInputResult,
    ResponseTargetInputStatus,
    ResponseTargetSet,
    ResponseTargetType,
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.response_target_repository import ResponseTargetRepository, StoredResponseTargetSet
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.response_target import ResponseTargetInputService

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42
FIRE_LAT = 32.731
FIRE_LON = 35.046


ACTIVE_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.ACTIVE_FIRE,
    latitude=FIRE_LAT,
    longitude=FIRE_LON,
    priority_score=150.0,
)
PREDICTED_TARGET = ResponseTarget(
    fire_event_id=FIRE_EVENT_ID,
    target_type=ResponseTargetType.PREDICTED_RISK,
    latitude=32.75,
    longitude=35.07,
    priority_score=80.0,
    prediction_horizon_minutes=30,
    spread_prediction_id=10,
    spread_prediction_cell_id=100,
)
CANDIDATE = PredictedRiskTargetCandidate(
    fire_event_id=FIRE_EVENT_ID,
    latitude=32.75,
    longitude=35.07,
    risk_score=80.0,
    prediction_horizon_minutes=30,
    spread_prediction_id=10,
    spread_prediction_cell_id=100,
)
INPUT_DATA = ResponseTargetInput(
    fire_event_id=FIRE_EVENT_ID,
    fire_latitude=FIRE_LAT,
    fire_longitude=FIRE_LON,
    severity_score=50.0,
    predicted_candidates=(CANDIDATE,),
)


class FakeInputService:
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls = []

    def prepare_input(self, fire_event_id, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeCalculator:
    def __init__(self, targets=(ACTIVE_TARGET, PREDICTED_TARGET), exc: Exception | None = None) -> None:
        self.targets = targets
        self.exc = exc
        self.calls = []

    def build_targets(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.targets


class FakeRepository:
    def __init__(self, stored_id=77, exc: Exception | None = None) -> None:
        self.stored_id = stored_id
        self.exc = exc
        self.calls = []

    def save_target_set(self, target_set):
        self.calls.append(target_set)
        if self.exc is not None:
            raise self.exc
        return StoredResponseTargetSet(
            id=self.stored_id,
            target_set=target_set,
            targets=(),
        )


def ready_result(input_data=INPUT_DATA) -> ResponseTargetInputResult:
    return ResponseTargetInputResult(
        status=ResponseTargetInputStatus.READY,
        input_data=input_data,
        fire_event_id=input_data.fire_event_id,
    )


def inactive_result() -> ResponseTargetInputResult:
    return ResponseTargetInputResult(
        status=ResponseTargetInputStatus.INACTIVE_EVENT,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
    )


def make_agent(input_service, calculator=None, repository=None) -> ResponseTargetGenerationAgent:
    return ResponseTargetGenerationAgent(
        input_service=input_service,
        calculator=calculator or FakeCalculator(),
        repository=repository or FakeRepository(),
    )


def test_ready_input_invokes_dependencies_once_and_returns_success():
    input_service = FakeInputService(ready_result())
    calculator = FakeCalculator()
    repository = FakeRepository()

    result = make_agent(input_service, calculator, repository).generate(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert input_service.calls == [{"fire_event_id": FIRE_EVENT_ID, "as_of": AS_OF}]
    assert len(calculator.calls) == 1
    assert len(repository.calls) == 1
    assert result.success is True
    assert result.status is ResponseTargetGenerationStatus.GENERATED
    assert result.target_set_id == 77


def test_calculator_receives_exact_prepared_input_fields():
    calculator = FakeCalculator()

    make_agent(FakeInputService(ready_result()), calculator).generate(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert calculator.calls == [
        {
            "fire_event_id": FIRE_EVENT_ID,
            "fire_latitude": FIRE_LAT,
            "fire_longitude": FIRE_LON,
            "severity_score": 50.0,
            "predicted_candidates": (CANDIDATE,),
        }
    ]


def test_target_set_uses_as_of_and_central_methodology_config():
    repository = FakeRepository()

    make_agent(FakeInputService(ready_result()), repository=repository).generate(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    target_set = repository.calls[0]
    assert target_set.generated_at is AS_OF
    assert target_set.methodology == RESPONSE_TARGET_METHODOLOGY_NAME
    assert target_set.methodology_version == RESPONSE_TARGET_METHODOLOGY_VERSION


def test_calculator_targets_passed_unchanged_and_in_order_to_repository():
    targets = (PREDICTED_TARGET, ACTIVE_TARGET)
    repository = FakeRepository()

    make_agent(FakeInputService(ready_result()), FakeCalculator(targets=targets), repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert repository.calls[0].targets == targets


def test_success_result_preserves_ordered_targets_and_count():
    targets = (ACTIVE_TARGET, PREDICTED_TARGET)

    result = make_agent(FakeInputService(ready_result()), FakeCalculator(targets=targets)).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.targets == targets
    assert result.target_count == 2


def test_ready_active_fire_only_persists_successfully():
    input_data = ResponseTargetInput(
        fire_event_id=FIRE_EVENT_ID,
        fire_latitude=FIRE_LAT,
        fire_longitude=FIRE_LON,
        severity_score=None,
        predicted_candidates=(),
    )
    repository = FakeRepository()

    result = make_agent(
        FakeInputService(ready_result(input_data)),
        FakeCalculator(targets=(ACTIVE_TARGET,)),
        repository,
    ).generate(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is ResponseTargetGenerationStatus.GENERATED
    assert repository.calls[0].targets == (ACTIVE_TARGET,)


def test_inactive_event_does_not_call_calculator_or_repository():
    calculator = FakeCalculator()
    repository = FakeRepository()

    result = make_agent(FakeInputService(inactive_result()), calculator, repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert calculator.calls == []
    assert repository.calls == []
    assert result.success is True
    assert result.status is ResponseTargetGenerationStatus.INACTIVE_EVENT
    assert result.target_set_id is None
    assert result.targets == ()
    assert result.target_count == 0


def test_input_service_exception_returns_failed_without_calculator_or_repository():
    calculator = FakeCalculator()
    repository = FakeRepository()

    result = make_agent(FakeInputService(exc=RuntimeError("input exploded")), calculator, repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.status is ResponseTargetGenerationStatus.FAILED
    assert result.success is False
    assert result.target_set_id is None
    assert calculator.calls == []
    assert repository.calls == []


def test_calculator_exception_returns_failed_without_repository_call():
    calculator = FakeCalculator(exc=RuntimeError("calculation exploded"))
    repository = FakeRepository()

    result = make_agent(FakeInputService(ready_result()), calculator, repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.status is ResponseTargetGenerationStatus.FAILED
    assert repository.calls == []


def test_repository_exception_returns_failed_without_fake_target_set_id():
    repository = FakeRepository(exc=RuntimeError("persistence exploded"))

    result = make_agent(FakeInputService(ready_result()), repository=repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.status is ResponseTargetGenerationStatus.FAILED
    assert result.target_set_id is None
    assert result.targets == ()
    assert len(repository.calls) == 1


def test_ready_result_without_input_data_returns_failed():
    input_result = SimpleNamespace(status=ResponseTargetInputStatus.READY, input_data=None, fire_event_id=FIRE_EVENT_ID)

    result = make_agent(FakeInputService(input_result)).generate(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF)

    assert result.status is ResponseTargetGenerationStatus.FAILED


def test_empty_calculator_output_returns_failed_and_is_not_persisted():
    repository = FakeRepository()

    result = make_agent(FakeInputService(ready_result()), FakeCalculator(targets=()), repository).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.status is ResponseTargetGenerationStatus.FAILED
    assert repository.calls == []


def test_failures_are_not_converted_to_inactive_event():
    result = make_agent(FakeInputService(exc=RuntimeError("input exploded"))).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert result.status is ResponseTargetGenerationStatus.FAILED


def test_identical_prepared_input_yields_equivalent_logical_targets():
    first = make_agent(FakeInputService(ready_result()), FakeCalculator(targets=(ACTIVE_TARGET,))).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )
    second = make_agent(FakeInputService(ready_result()), FakeCalculator(targets=(ACTIVE_TARGET,))).generate(
        fire_event_id=FIRE_EVENT_ID,
        as_of=AS_OF,
    )

    assert first.targets == second.targets


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_result())).generate(fire_event_id=invalid_fire_event_id, as_of=AS_OF)


def test_naive_as_of_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_result())).generate(
            fire_event_id=FIRE_EVENT_ID,
            as_of=datetime(2026, 9, 14, 12, 0),
        )


def test_result_model_invariants():
    with pytest.raises(ValueError):
        from src.agents.analysis.response_target_generation_result import ResponseTargetGenerationResult

        ResponseTargetGenerationResult(
            success=True,
            fire_event_id=FIRE_EVENT_ID,
            status=ResponseTargetGenerationStatus.GENERATED,
            target_set_id=None,
            targets=(ACTIVE_TARGET,),
            target_count=1,
        )


# ---------------------------------------------------------------------------
# SQLite-backed production-component orchestration
# ---------------------------------------------------------------------------


def persist_event(fire_event_repository, satellite_repository, status=FireEventStatus.CONFIRMED) -> int:
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=FIRE_LAT,
            longitude=FIRE_LON,
            detected_at=AS_OF - timedelta(minutes=30),
            confidence="h",
            frp=72.0,
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=AS_OF, lookback_minutes=360)[0].id
    return fire_event_repository.create_event(
        FireEvent(
            latitude=FIRE_LAT,
            longitude=FIRE_LON,
            detected_at=AS_OF - timedelta(minutes=30),
            updated_at=AS_OF - timedelta(minutes=5),
            status=status,
            detection_confidence=0.85,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    ).id


def persist_weather(weather_repository, station_offset=0) -> int:
    station = WeatherStation(
        external_station_id=940000 + station_offset,
        name=f"Station {station_offset}",
        latitude=FIRE_LAT,
        longitude=FIRE_LON,
    )
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=AS_OF - timedelta(minutes=5),
            temperature=28.0,
            relative_humidity=35.0,
            wind_speed=6.0,
            wind_direction=270.0,
        )
    )
    return next(
        record.observation_id
        for record in weather_repository.get_recent_observations_for_area_candidates(
            latitude=FIRE_LAT,
            longitude=FIRE_LON,
            radius_km=5.0,
            start_time=AS_OF - timedelta(minutes=30),
            end_time=AS_OF,
        )
        if record.observation.station_external_id == station.external_station_id
    )


def persist_severity(severity_repository, weather_repository, satellite_repository, fire_event_id: int) -> int:
    weather_id = persist_weather(weather_repository, station_offset=fire_event_id)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=FIRE_LAT,
            longitude=FIRE_LON,
            detected_at=AS_OF - timedelta(minutes=20),
            confidence="h",
            frp=55.0,
            satellite="N21",
        )
    )
    hotspot_id = max(record.id for record in satellite_repository.get_recent_hotspots(as_of=AS_OF, lookback_minutes=360))
    return severity_repository.save_assessment(
        FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=AS_OF - timedelta(minutes=10),
            status=FireSeverityAssessmentStatus.VALID,
            score=50.0,
            level=FireSeverityLevel.HIGH,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    ).assessment_id


def persist_spread_prediction(
    spread_repository,
    severity_repository,
    weather_repository,
    satellite_repository,
    fire_event_id: int,
    cells: tuple[FireSpreadPredictionCell, ...],
):
    assessment_id = persist_severity(severity_repository, weather_repository, satellite_repository, fire_event_id)
    weather_id = persist_weather(weather_repository, station_offset=100 + fire_event_id)
    return spread_repository.save_prediction(
        FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=assessment_id,
            predicted_at=AS_OF - timedelta(minutes=5),
            horizon_minutes=30,
            status=FireSpreadPredictionStatus.VALID,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=cells,
        ),
        weather_observation_id=weather_id,
    )


def make_real_agent(sqlite_session_factory) -> tuple[ResponseTargetGenerationAgent, ResponseTargetRepository]:
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    spread_repository = FireSpreadPredictionRepository(sqlite_session_factory)
    target_repository = ResponseTargetRepository(sqlite_session_factory)
    return (
        ResponseTargetGenerationAgent(
            input_service=ResponseTargetInputService(
                fire_event_repository=fire_event_repository,
                fire_severity_assessment_repository=severity_repository,
                fire_spread_prediction_repository=spread_repository,
            ),
            calculator=ResponseTargetCalculator(),
            repository=target_repository,
        ),
        target_repository,
    )


def test_real_agent_active_fire_without_spread_persists_active_target(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    fire_event_id = persist_event(fire_event_repository, satellite_repository)
    agent, repository = make_real_agent(sqlite_session_factory)

    result = agent.generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.status is ResponseTargetGenerationStatus.GENERATED
    assert result.target_count == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert repository.get_by_id(result.target_set_id).target_set.targets == result.targets


def test_real_agent_filters_low_risk_and_deduplicates_high_risk_predictions(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    weather_repository = WeatherRepository(sqlite_session_factory)
    spread_repository = FireSpreadPredictionRepository(sqlite_session_factory)
    fire_event_id = persist_event(fire_event_repository, satellite_repository)
    persist_spread_prediction(
        spread_repository,
        severity_repository,
        weather_repository,
        satellite_repository,
        fire_event_id,
        cells=(
            FireSpreadPredictionCell(
                latitude=32.75,
                longitude=35.07,
                spread_probability=0.8,
                spread_risk_score=80.0,
                reached_step=1,
                reached_minutes=5,
            ),
            FireSpreadPredictionCell(
                latitude=32.75001,
                longitude=35.07001,
                spread_probability=0.7,
                spread_risk_score=70.0,
                reached_step=2,
                reached_minutes=10,
            ),
            FireSpreadPredictionCell(
                latitude=32.76,
                longitude=35.08,
                spread_probability=0.55,
                spread_risk_score=55.0,
                reached_step=3,
                reached_minutes=15,
            ),
        ),
    )
    agent, repository = make_real_agent(sqlite_session_factory)

    result = agent.generate(fire_event_id=fire_event_id, as_of=AS_OF)
    stored = repository.get_by_id(result.target_set_id)

    assert [target.target_type for target in result.targets] == [
        ResponseTargetType.ACTIVE_FIRE,
        ResponseTargetType.PREDICTED_RISK,
    ]
    predicted = result.targets[1]
    assert predicted.risk_score if hasattr(predicted, "risk_score") else predicted.priority_score == 80.0
    assert stored.targets[0].target_order == 0
    assert stored.targets[1].target_order == 1
    assert stored.targets[1].target.spread_prediction_id is not None
    assert stored.targets[1].target.spread_prediction_cell_id is not None


def test_real_agent_does_not_attach_other_event_spread_by_geography(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    weather_repository = WeatherRepository(sqlite_session_factory)
    spread_repository = FireSpreadPredictionRepository(sqlite_session_factory)
    requested_event_id = persist_event(fire_event_repository, satellite_repository)
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    persist_spread_prediction(
        spread_repository,
        severity_repository,
        weather_repository,
        satellite_repository,
        other_event_id,
        cells=(
            FireSpreadPredictionCell(
                latitude=FIRE_LAT,
                longitude=FIRE_LON,
                spread_probability=0.9,
                spread_risk_score=90.0,
                reached_step=1,
                reached_minutes=5,
            ),
        ),
    )
    agent, _ = make_real_agent(sqlite_session_factory)

    result = agent.generate(fire_event_id=requested_event_id, as_of=AS_OF)

    assert result.target_count == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE


def test_real_agent_repeated_generation_is_append_only(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    fire_event_id = persist_event(fire_event_repository, satellite_repository)
    agent, repository = make_real_agent(sqlite_session_factory)

    first = agent.generate(fire_event_id=fire_event_id, as_of=AS_OF)
    second = agent.generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert first.targets == second.targets
    assert first.target_set_id != second.target_set_id
    assert len(repository.get_history_for_event(fire_event_id)) == 2
