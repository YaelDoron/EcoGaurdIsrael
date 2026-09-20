"""Tests for FireDangerAssessmentAgent orchestration."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.agents.analysis.fire_danger_assessment_agent import FireDangerAssessmentAgent
from src.agents.analysis.fire_danger_assessment_result import FireDangerAssessmentResult
from src.calculators.fire_danger.ffwi_config import (
    FFWI_METHODOLOGY_NAME,
    FFWI_METHODOLOGY_VERSION,
)
from src.models import (
    AssessmentArea,
    FireDangerAssessment,
    FireDangerAssessmentStatus,
    FireDangerCalculation,
    FireDangerInput,
    FireDangerInputResult,
    FireDangerInputStatus,
    FireDangerLevel,
)
from src.repositories.fire_danger_assessment_repository import StoredFireDangerAssessment

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
AREA = AssessmentArea(
    id="area-carmel",
    name="Carmel",
    latitude=32.731,
    longitude=35.046,
    radius_km=5,
)
INPUT_DATA = FireDangerInput(temperature_c=30, relative_humidity_pct=25, wind_speed_kmh=12)
CALCULATION = FireDangerCalculation(score=42.5, level=FireDangerLevel.VERY_HIGH)


class FakeInputService:
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls = []

    def build_input(self, area, as_of):
        self.calls.append({"area": area, "as_of": as_of})
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeCalculator:
    def __init__(self, calculation=CALCULATION, exc: Exception | None = None) -> None:
        self.calculation = calculation
        self.exc = exc
        self.calls = []

    def calculate(self, input_data):
        self.calls.append(input_data)
        if self.exc is not None:
            raise self.exc
        return self.calculation


class FakeRepository:
    def __init__(self, stored_id: int = 77, exc: Exception | None = None) -> None:
        self.stored_id = stored_id
        self.exc = exc
        self.calls = []

    def save_assessment(self, assessment, observation_ids, station_ids):
        self.calls.append(
            {
                "assessment": assessment,
                "observation_ids": observation_ids,
                "station_ids": station_ids,
            }
        )
        if self.exc is not None:
            raise self.exc
        return StoredFireDangerAssessment(
            assessment_id=self.stored_id,
            assessment=assessment,
            observation_ids=tuple(observation_ids),
            station_ids=tuple(station_ids),
            created_at=assessment.assessed_at,
        )


def ready_input_result(
    observation_ids=(101, 102),
    station_ids=(1, 2),
) -> FireDangerInputResult:
    return FireDangerInputResult(
        status=FireDangerInputStatus.READY,
        input_data=INPUT_DATA,
        observation_ids=observation_ids,
        station_ids=station_ids,
    )


def insufficient_input_result() -> FireDangerInputResult:
    return FireDangerInputResult(
        status=FireDangerInputStatus.INSUFFICIENT_DATA,
        input_data=None,
        observation_ids=(),
        station_ids=(),
    )


def make_agent(input_service, calculator=None, repository=None) -> FireDangerAssessmentAgent:
    return FireDangerAssessmentAgent(
        input_service=input_service,
        calculator=calculator or FakeCalculator(),
        repository=repository or FakeRepository(),
    )


def make_valid_assessment() -> FireDangerAssessment:
    return FireDangerAssessment(
        area_id=AREA.id,
        area_name=AREA.name,
        area_latitude=AREA.latitude,
        area_longitude=AREA.longitude,
        area_radius_km=AREA.radius_km,
        assessed_at=AS_OF,
        status=FireDangerAssessmentStatus.VALID,
        score=CALCULATION.score,
        level=CALCULATION.level,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )


def test_ready_input_calls_dependencies_once_and_returns_success():
    input_service = FakeInputService(ready_input_result())
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(input_service, calculator, repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert input_service.calls == [{"area": AREA, "as_of": AS_OF}]
    assert calculator.calls == [INPUT_DATA]
    assert len(repository.calls) == 1
    assert result.success is True
    assert result.stored_assessment_id == 77
    assert result.error_message is None


def test_ready_input_creates_valid_assessment():
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.assessment.status is FireDangerAssessmentStatus.VALID
    assert repository.calls[0]["assessment"].status is FireDangerAssessmentStatus.VALID


def test_valid_assessment_contains_area_calculation_time_and_methodology_fields():
    result = make_agent(FakeInputService(ready_input_result())).assess(area=AREA, as_of=AS_OF)

    assert result.assessment.area_id == AREA.id
    assert result.assessment.area_name == AREA.name
    assert result.assessment.area_latitude == AREA.latitude
    assert result.assessment.area_longitude == AREA.longitude
    assert result.assessment.area_radius_km == AREA.radius_km
    assert result.assessment.assessed_at == AS_OF
    assert result.assessment.score == CALCULATION.score
    assert result.assessment.level is CALCULATION.level
    assert result.assessment.methodology == FFWI_METHODOLOGY_NAME
    assert result.assessment.methodology_version == FFWI_METHODOLOGY_VERSION


def test_ready_flow_passes_exact_observation_ids_to_repository():
    observation_ids = (301, 201, 205)
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result(observation_ids=observation_ids)), repository=repository)

    agent.assess(area=AREA, as_of=AS_OF)

    assert repository.calls[0]["observation_ids"] == observation_ids


def test_ready_flow_passes_exact_station_ids_to_repository():
    station_ids = (9, 4, 7)
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result(station_ids=station_ids)), repository=repository)

    agent.assess(area=AREA, as_of=AS_OF)

    assert repository.calls[0]["station_ids"] == station_ids


def test_agent_does_not_perform_another_weather_lookup_after_input_service_result():
    input_service = FakeInputService(ready_input_result())
    agent = make_agent(input_service)

    agent.assess(area=AREA, as_of=AS_OF)

    assert len(input_service.calls) == 1


def test_insufficient_data_persists_business_result_without_calculation():
    input_service = FakeInputService(insufficient_input_result())
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(input_service, calculator, repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert calculator.calls == []
    assert len(repository.calls) == 1
    persisted = repository.calls[0]["assessment"]
    assert persisted.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA
    assert persisted.score is None
    assert persisted.level is None
    assert repository.calls[0]["observation_ids"] == ()
    assert repository.calls[0]["station_ids"] == ()
    assert result.success is True


def test_insufficient_data_is_not_operational_failure():
    result = make_agent(FakeInputService(insufficient_input_result())).assess(area=AREA, as_of=AS_OF)

    assert result.success is True
    assert result.error_message is None
    assert result.assessment.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA


def test_same_area_as_of_and_dependency_outputs_produce_equivalent_output():
    first = make_agent(FakeInputService(ready_input_result())).assess(area=AREA, as_of=AS_OF)
    second = make_agent(FakeInputService(ready_input_result())).assess(area=AREA, as_of=AS_OF)

    assert first == second


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_input_result())).assess(
            area=AREA,
            as_of=datetime(2026, 9, 14, 12, 0),
        )


def test_input_service_exception_returns_failed_result_without_persistence_or_calculation():
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(FakeInputService(exc=RuntimeError("input exploded")), calculator, repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.success is False
    assert result.assessment is None
    assert result.stored_assessment_id is None
    assert result.error_message == "Fire-danger input preparation failed."
    assert calculator.calls == []
    assert repository.calls == []


def test_calculator_exception_returns_failed_result_without_persistence():
    calculator = FakeCalculator(exc=RuntimeError("calculation exploded"))
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result()), calculator, repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.success is False
    assert result.assessment is None
    assert result.stored_assessment_id is None
    assert result.error_message == "Fire-danger calculation failed."
    assert repository.calls == []


def test_repository_exception_returns_failed_result():
    repository = FakeRepository(exc=RuntimeError("persistence exploded"))
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.success is False
    assert result.assessment.status is FireDangerAssessmentStatus.VALID
    assert result.stored_assessment_id is None
    assert result.error_message == "Fire-danger assessment persistence failed."


def test_operational_failure_is_not_persisted_as_insufficient_data():
    repository = FakeRepository(exc=RuntimeError("persistence exploded"))
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.assessment.status is FireDangerAssessmentStatus.VALID
    assert repository.calls[0]["assessment"].status is FireDangerAssessmentStatus.VALID


def test_repository_receives_exactly_one_save_request_per_successful_assessment_run():
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    result = agent.assess(area=AREA, as_of=AS_OF)

    assert result.success is True
    assert len(repository.calls) == 1


def test_successful_result_construction():
    result = FireDangerAssessmentResult(
        assessment=make_valid_assessment(),
        stored_assessment_id=5,
        success=True,
    )

    assert result.success is True
    assert result.stored_assessment_id == 5


def test_failed_result_construction():
    result = FireDangerAssessmentResult(
        assessment=None,
        stored_assessment_id=None,
        success=False,
        error_message="Fire-danger input preparation failed.",
    )

    assert result.success is False
    assert result.error_message == "Fire-danger input preparation failed."


def test_result_is_immutable():
    result = FireDangerAssessmentResult(
        assessment=make_valid_assessment(),
        stored_assessment_id=5,
        success=True,
    )

    with pytest.raises(FrozenInstanceError):
        result.success = False


def test_successful_result_rejects_error_message():
    with pytest.raises(ValueError):
        FireDangerAssessmentResult(
            assessment=make_valid_assessment(),
            stored_assessment_id=5,
            success=True,
            error_message="should not be present",
        )


def test_failed_result_rejects_stored_assessment_id():
    with pytest.raises(ValueError):
        FireDangerAssessmentResult(
            assessment=None,
            stored_assessment_id=5,
            success=False,
            error_message="failed",
        )
