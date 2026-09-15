"""Tests for FireSeverityAssessmentAgent orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.agents.analysis import FireSeverityAssessmentAgent
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models import (
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityCalculation,
    FireSeverityInput,
    FireSeverityInputResult,
    FireSeverityInputStatus,
    FireSeverityLevel,
    VegetationData,
)
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42
INPUT_DATA = FireSeverityInput(
    frp_mw=80,
    wind_speed_kmh=24,
    relative_humidity_pct=30,
    vegetation_fuel_score=0.7,
)
CALCULATION = FireSeverityCalculation(
    score=73.25,
    level=FireSeverityLevel.HIGH,
    frp_factor=0.8,
    wind_factor=0.48,
    dryness_factor=0.7,
    vegetation_factor=0.7,
    available_weight=1.0,
    methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
    methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
)
VEGETATION = VegetationData(
    fuel_score=0.7,
    dominant_land_cover="Shrubland",
    source="copernicus_land_cover",
    dataset_year=2024,
    radius_km=1.5,
    land_cover_distribution=(("Shrubland", 0.8), ("Urban", 0.2)),
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

    def save_assessment(
        self,
        assessment,
        weather_observation_ids,
        satellite_hotspot_ids,
        selected_frp_hotspot_id,
    ):
        self.calls.append(
            {
                "assessment": assessment,
                "weather_observation_ids": weather_observation_ids,
                "satellite_hotspot_ids": satellite_hotspot_ids,
                "selected_frp_hotspot_id": selected_frp_hotspot_id,
            }
        )
        if self.exc is not None:
            raise self.exc
        return StoredFireSeverityAssessment(
            assessment_id=self.stored_id,
            assessment=assessment,
            weather_observation_ids=tuple(weather_observation_ids),
            satellite_hotspot_ids=tuple(satellite_hotspot_ids),
            selected_frp_hotspot_id=selected_frp_hotspot_id,
        )


def ready_input_result(
    *,
    input_data=INPUT_DATA,
    weather_ids=(101, 102),
    satellite_ids=(201, 202),
    selected_frp_hotspot_id=202,
    vegetation_data=VEGETATION,
) -> FireSeverityInputResult:
    return FireSeverityInputResult(
        status=FireSeverityInputStatus.READY,
        input_data=input_data,
        fire_event_id=FIRE_EVENT_ID,
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
        vegetation_data=vegetation_data,
    )


def non_ready_input_result(
    status: FireSeverityInputStatus,
    *,
    weather_ids=(),
    satellite_ids=(),
    selected_frp_hotspot_id=None,
    vegetation_data=None,
) -> FireSeverityInputResult:
    return FireSeverityInputResult(
        status=status,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
        vegetation_data=vegetation_data,
    )


def make_agent(input_service, calculator=None, repository=None) -> FireSeverityAssessmentAgent:
    return FireSeverityAssessmentAgent(
        input_service=input_service,
        calculator=calculator or FakeCalculator(),
        repository=repository or FakeRepository(),
    )


def test_ready_input_calls_dependencies_once_and_returns_repository_result():
    input_service = FakeInputService(ready_input_result())
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(input_service, calculator, repository)

    result = agent.assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert input_service.calls == [{"fire_event_id": FIRE_EVENT_ID, "as_of": ASSESSED_AT}]
    assert calculator.calls == [INPUT_DATA]
    assert len(repository.calls) == 1
    assert isinstance(result, StoredFireSeverityAssessment)
    assert result.assessment_id == 77
    assert result.assessment == repository.calls[0]["assessment"]


def test_ready_input_creates_valid_assessment_from_calculator_result_and_methodology():
    repository = FakeRepository()
    result = make_agent(FakeInputService(ready_input_result()), repository=repository).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assessment = result.assessment
    assert assessment.fire_event_id == FIRE_EVENT_ID
    assert assessment.assessed_at == ASSESSED_AT
    assert assessment.status is FireSeverityAssessmentStatus.VALID
    assert assessment.score == CALCULATION.score
    assert assessment.level is CALCULATION.level
    assert assessment.methodology == FIRE_SEVERITY_METHODOLOGY_NAME
    assert assessment.methodology_version == FIRE_SEVERITY_METHODOLOGY_VERSION
    assert repository.calls[0]["assessment"] == assessment


def test_ready_input_copies_vegetation_snapshot():
    result = make_agent(FakeInputService(ready_input_result())).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert result.assessment.vegetation_source == VEGETATION.source
    assert result.assessment.vegetation_dataset_year == VEGETATION.dataset_year
    assert result.assessment.vegetation_radius_km == VEGETATION.radius_km
    assert result.assessment.vegetation_dominant_land_cover == VEGETATION.dominant_land_cover
    assert result.assessment.vegetation_fuel_score == VEGETATION.fuel_score


def test_ready_without_vegetation_still_persists_valid_assessment():
    result = make_agent(FakeInputService(ready_input_result(vegetation_data=None))).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert result.assessment.status is FireSeverityAssessmentStatus.VALID
    assert result.assessment.vegetation_source is None
    assert result.assessment.vegetation_dataset_year is None
    assert result.assessment.vegetation_radius_km is None
    assert result.assessment.vegetation_dominant_land_cover is None
    assert result.assessment.vegetation_fuel_score is None


def test_ready_flow_passes_exact_traceability_to_repository():
    repository = FakeRepository()
    weather_ids = (301, 101)
    satellite_ids = (902, 901)
    selected_frp_id = 901
    agent = make_agent(
        FakeInputService(
            ready_input_result(
                weather_ids=weather_ids,
                satellite_ids=satellite_ids,
                selected_frp_hotspot_id=selected_frp_id,
            )
        ),
        repository=repository,
    )

    agent.assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert repository.calls[0]["weather_observation_ids"] == tuple(sorted(weather_ids))
    assert repository.calls[0]["satellite_hotspot_ids"] == tuple(sorted(satellite_ids))
    assert repository.calls[0]["selected_frp_hotspot_id"] == selected_frp_id


def test_insufficient_data_persists_not_calculated_assessment_and_partial_traceability():
    input_result = non_ready_input_result(
        FireSeverityInputStatus.INSUFFICIENT_DATA,
        weather_ids=(11, 12),
        satellite_ids=(21,),
        selected_frp_hotspot_id=21,
        vegetation_data=VEGETATION,
    )
    calculator = FakeCalculator()
    repository = FakeRepository()
    result = make_agent(FakeInputService(input_result), calculator, repository).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert calculator.calls == []
    assert result.assessment.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA
    assert result.assessment.score is None
    assert result.assessment.level is None
    assert result.assessment.vegetation_source == VEGETATION.source
    assert repository.calls[0]["weather_observation_ids"] == (11, 12)
    assert repository.calls[0]["satellite_hotspot_ids"] == (21,)
    assert repository.calls[0]["selected_frp_hotspot_id"] == 21


def test_inactive_event_persists_inactive_assessment_without_calculation():
    input_result = non_ready_input_result(FireSeverityInputStatus.INACTIVE_EVENT)
    calculator = FakeCalculator()
    repository = FakeRepository()

    result = make_agent(FakeInputService(input_result), calculator, repository).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert calculator.calls == []
    assert result.assessment.status is FireSeverityAssessmentStatus.INACTIVE_EVENT
    assert result.assessment.score is None
    assert result.assessment.level is None
    assert len(repository.calls) == 1


@pytest.mark.parametrize(
    ("weather_ids", "satellite_ids", "selected_frp_hotspot_id"),
    [
        ((1, 2), (), None),
        ((), (7, 8), 8),
    ],
)
def test_non_ready_flow_forwards_partial_traceability_unchanged(
    weather_ids,
    satellite_ids,
    selected_frp_hotspot_id,
):
    repository = FakeRepository()
    input_result = non_ready_input_result(
        FireSeverityInputStatus.INSUFFICIENT_DATA,
        weather_ids=weather_ids,
        satellite_ids=satellite_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
    )

    make_agent(FakeInputService(input_result), repository=repository).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert repository.calls[0]["weather_observation_ids"] == tuple(sorted(weather_ids))
    assert repository.calls[0]["satellite_hotspot_ids"] == tuple(sorted(satellite_ids))
    assert repository.calls[0]["selected_frp_hotspot_id"] == selected_frp_hotspot_id


def test_assessed_at_is_passed_to_input_service_and_stored_exactly():
    input_service = FakeInputService(ready_input_result())

    result = make_agent(input_service).assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert input_service.calls[0]["as_of"] is ASSESSED_AT
    assert result.assessment.assessed_at is ASSESSED_AT


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_is_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_input_result())).assess(
            fire_event_id=invalid_fire_event_id,
            assessed_at=ASSESSED_AT,
        )


def test_naive_assessed_at_is_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_input_result())).assess(
            fire_event_id=FIRE_EVENT_ID,
            assessed_at=datetime(2026, 9, 14, 12, 0),
        )


def test_input_service_exception_propagates_without_calculation_or_persistence():
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(FakeInputService(exc=RuntimeError("input exploded")), calculator, repository)

    with pytest.raises(RuntimeError, match="input exploded"):
        agent.assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert calculator.calls == []
    assert repository.calls == []


def test_calculator_exception_propagates_without_persistence():
    calculator = FakeCalculator(exc=RuntimeError("calculation exploded"))
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result()), calculator, repository)

    with pytest.raises(RuntimeError, match="calculation exploded"):
        agent.assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert repository.calls == []


def test_repository_exception_propagates_after_building_domain_assessment():
    repository = FakeRepository(exc=RuntimeError("persistence exploded"))
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    with pytest.raises(RuntimeError, match="persistence exploded"):
        agent.assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert repository.calls[0]["assessment"].status is FireSeverityAssessmentStatus.VALID


@pytest.mark.parametrize(
    "status",
    [FireSeverityInputStatus.INSUFFICIENT_DATA, FireSeverityInputStatus.INACTIVE_EVENT],
)
def test_calculator_is_not_called_when_input_is_not_ready(status):
    calculator = FakeCalculator()

    make_agent(FakeInputService(non_ready_input_result(status)), calculator).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert calculator.calls == []


def test_agent_uses_calculator_output_without_recalculating_score_or_level():
    calculation = FireSeverityCalculation(
        score=1.23,
        level=FireSeverityLevel.CRITICAL,
        frp_factor=1,
        wind_factor=1,
        dryness_factor=1,
        vegetation_factor=None,
        available_weight=0.9,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
    )

    result = make_agent(
        FakeInputService(ready_input_result(input_data=FireSeverityInput(100, 50, 0))),
        calculator=FakeCalculator(calculation),
    ).assess(fire_event_id=FIRE_EVENT_ID, assessed_at=ASSESSED_AT)

    assert result.assessment.score == 1.23
    assert result.assessment.level is FireSeverityLevel.CRITICAL


def test_ready_result_without_input_data_fails_explicitly():
    input_result = SimpleNamespace(
        status=FireSeverityInputStatus.READY,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        weather_observation_ids=(1,),
        satellite_hotspot_ids=(2,),
        selected_frp_hotspot_id=2,
        vegetation_data=None,
    )

    with pytest.raises(ValueError, match="requires input_data"):
        make_agent(FakeInputService(input_result)).assess(
            fire_event_id=FIRE_EVENT_ID,
            assessed_at=ASSESSED_AT,
        )


def test_unknown_input_status_fails_explicitly_without_persistence():
    repository = FakeRepository()
    input_result = SimpleNamespace(
        status=SimpleNamespace(value="unknown"),
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        weather_observation_ids=(),
        satellite_hotspot_ids=(),
        selected_frp_hotspot_id=None,
        vegetation_data=None,
    )

    with pytest.raises(ValueError, match="Unsupported fire-severity input status"):
        make_agent(FakeInputService(input_result), repository=repository).assess(
            fire_event_id=FIRE_EVENT_ID,
            assessed_at=ASSESSED_AT,
        )

    assert repository.calls == []


def test_repository_return_type_contains_persisted_assessment_and_traceability():
    result = make_agent(FakeInputService(ready_input_result())).assess(
        fire_event_id=FIRE_EVENT_ID,
        assessed_at=ASSESSED_AT,
    )

    assert isinstance(result.assessment, FireSeverityAssessment)
    assert result.weather_observation_ids == (101, 102)
    assert result.satellite_hotspot_ids == (201, 202)
    assert result.selected_frp_hotspot_id == 202
