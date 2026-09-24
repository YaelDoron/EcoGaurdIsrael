"""Tests for FireSpreadPredictionAgent orchestration."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.agents.analysis import FireSpreadPredictionAgent
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models import (
    FireSpreadEffectiveState,
    FireSpreadCalculation,
    FireSpreadFuelClass,
    FireSpreadInput,
    FireSpreadInputResult,
    FireSpreadInputStatus,
    FireSpreadInsufficientDataReason,
    FireSpreadPredictionCell,
    FireSpreadPredictionStatus,
)
from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPrediction

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42
INPUT_DATA = FireSpreadInput(
    origin_latitude=32.731,
    origin_longitude=35.046,
    wind_speed_kmh=21.6,
    wind_direction_deg=270.0,
    fuel_moisture_percent=12.0,
    fuel_class=FireSpreadFuelClass.SHRUBS,
    horizon_minutes=30,
)
CELL = FireSpreadPredictionCell(
    latitude=32.735,
    longitude=35.05,
    spread_probability=0.6,
    spread_risk_score=60.0,
    reached_step=1,
    reached_minutes=5,
)
CALCULATION = FireSpreadCalculation(
    cells=(CELL,),
    horizon_minutes=30,
    methodology=METHODOLOGY_NAME,
    methodology_version=METHODOLOGY_VERSION,
)
EMPTY_CALCULATION = FireSpreadCalculation(
    cells=(),
    horizon_minutes=30,
    methodology=METHODOLOGY_NAME,
    methodology_version=METHODOLOGY_VERSION,
)


class FakeInputService:
    def __init__(self, result=None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls = []

    def prepare_input(self, fire_event_id, as_of, horizon_minutes):
        self.calls.append({"fire_event_id": fire_event_id, "as_of": as_of, "horizon_minutes": horizon_minutes})
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

    def save_prediction(self, prediction, weather_observation_id):
        self.calls.append({"prediction": prediction, "weather_observation_id": weather_observation_id})
        if self.exc is not None:
            raise self.exc
        return StoredFireSpreadPrediction(
            id=self.stored_id,
            prediction=prediction,
            weather_observation_id=weather_observation_id,
        )


def ready_input_result(
    *,
    input_data=INPUT_DATA,
    severity_assessment_id=500,
    weather_observation_id=101,
) -> FireSpreadInputResult:
    return FireSpreadInputResult(
        status=FireSpreadInputStatus.READY,
        input_data=input_data,
        fire_event_id=FIRE_EVENT_ID,
        severity_assessment_id=severity_assessment_id,
        weather_observation_id=weather_observation_id,
    )


def non_ready_input_result(
    status: FireSpreadInputStatus,
    *,
    severity_assessment_id=None,
    weather_observation_id=None,
    insufficient_data_reason=FireSpreadInsufficientDataReason.MISSING_VEGETATION,
) -> FireSpreadInputResult:
    return FireSpreadInputResult(
        status=status,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        severity_assessment_id=severity_assessment_id,
        weather_observation_id=weather_observation_id,
        insufficient_data_reason=(
            insufficient_data_reason if status is FireSpreadInputStatus.INSUFFICIENT_DATA else None
        ),
    )


def make_agent(input_service, calculator=None, repository=None) -> FireSpreadPredictionAgent:
    return FireSpreadPredictionAgent(
        input_service=input_service,
        calculator=calculator or FakeCalculator(),
        repository=repository or FakeRepository(),
    )


# ---------------------------------------------------------------------------
# READY
# ---------------------------------------------------------------------------


def test_ready_input_calls_dependencies_once_and_returns_repository_result():
    input_service = FakeInputService(ready_input_result())
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(input_service, calculator, repository)

    result = agent.predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert input_service.calls == [{"fire_event_id": FIRE_EVENT_ID, "as_of": AS_OF, "horizon_minutes": 30}]
    assert calculator.calls == [INPUT_DATA]
    assert len(repository.calls) == 1
    assert isinstance(result, StoredFireSpreadPrediction)
    assert result.id == 77
    assert result.prediction == repository.calls[0]["prediction"]


def test_ready_input_creates_valid_prediction_with_expected_fields():
    repository = FakeRepository()
    result = make_agent(FakeInputService(ready_input_result()), repository=repository).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    prediction = result.prediction
    assert prediction.fire_event_id == FIRE_EVENT_ID
    assert prediction.severity_assessment_id == 500
    assert prediction.predicted_at is AS_OF
    assert prediction.horizon_minutes == 30
    assert prediction.status is FireSpreadPredictionStatus.VALID
    assert prediction.methodology == METHODOLOGY_NAME
    assert prediction.methodology_version == METHODOLOGY_VERSION
    assert prediction.cells == (CELL,)
    assert prediction.effective_state_fingerprint == FireSpreadEffectiveState.from_input(
        fire_event_id=FIRE_EVENT_ID,
        spread_input=INPUT_DATA,
    ).fingerprint


def test_predict_from_prepared_input_uses_supplied_fingerprint_without_preparing_again():
    input_service = FakeInputService(ready_input_result())
    repository = FakeRepository()
    fingerprint = "a" * 64
    agent = make_agent(input_service, repository=repository)

    result = agent.predict_from_input_result(
        input_result=ready_input_result(),
        as_of=AS_OF,
        horizon_minutes=30,
        effective_state_fingerprint=fingerprint,
    )

    assert input_service.calls == []
    assert result.prediction.effective_state_fingerprint == fingerprint


def test_ready_flow_passes_selected_weather_observation_id_to_repository():
    repository = FakeRepository()
    make_agent(
        FakeInputService(ready_input_result(weather_observation_id=909)), repository=repository
    ).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert repository.calls[0]["weather_observation_id"] == 909


def test_ready_no_spread_calculation_still_produces_valid_status():
    repository = FakeRepository()
    calculator = FakeCalculator(calculation=EMPTY_CALCULATION)

    result = make_agent(FakeInputService(ready_input_result()), calculator, repository).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert result.prediction.status is FireSpreadPredictionStatus.VALID
    assert result.prediction.cells == ()
    assert repository.calls[0]["prediction"].status is FireSpreadPredictionStatus.VALID


@pytest.mark.parametrize("horizon_minutes", [30, 60])
def test_agent_supports_both_horizons(horizon_minutes):
    input_service = FakeInputService(ready_input_result())
    calculator = FakeCalculator(calculation=FireSpreadCalculation(
        cells=(CELL,), horizon_minutes=horizon_minutes, methodology=METHODOLOGY_NAME, methodology_version=METHODOLOGY_VERSION
    ))

    result = make_agent(input_service, calculator).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=horizon_minutes
    )

    assert input_service.calls[0]["horizon_minutes"] == horizon_minutes
    assert result.prediction.horizon_minutes == horizon_minutes


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA
# ---------------------------------------------------------------------------


def test_insufficient_data_does_not_call_calculator():
    calculator = FakeCalculator()

    make_agent(
        FakeInputService(non_ready_input_result(FireSpreadInputStatus.INSUFFICIENT_DATA)), calculator
    ).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert calculator.calls == []


def test_insufficient_data_persists_zero_cells_and_no_fake_severity_id():
    repository = FakeRepository()
    result = make_agent(
        FakeInputService(non_ready_input_result(FireSpreadInputStatus.INSUFFICIENT_DATA)), repository=repository
    ).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert result.prediction.status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
    assert result.prediction.cells == ()
    assert result.prediction.severity_assessment_id is None
    assert repository.calls[0]["weather_observation_id"] is None


def test_insufficient_data_preserves_known_severity_id_when_available():
    input_result = non_ready_input_result(FireSpreadInputStatus.INSUFFICIENT_DATA, severity_assessment_id=500)
    result = make_agent(FakeInputService(input_result)).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert result.prediction.severity_assessment_id == 500


@pytest.mark.parametrize(
    "reason", [FireSpreadInsufficientDataReason.STALE_WEATHER, FireSpreadInsufficientDataReason.UNSUPPORTED_VEGETATION]
)
def test_insufficient_data_reason_is_passed_through_to_the_persisted_prediction(reason):
    repository = FakeRepository()
    input_result = non_ready_input_result(FireSpreadInputStatus.INSUFFICIENT_DATA, insufficient_data_reason=reason)

    result = make_agent(FakeInputService(input_result), repository=repository).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert result.prediction.insufficient_data_reason is reason
    assert repository.calls[0]["prediction"].insufficient_data_reason is reason


def test_valid_and_inactive_predictions_have_no_reason():
    valid = make_agent(FakeInputService(ready_input_result())).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )
    inactive = make_agent(FakeInputService(non_ready_input_result(FireSpreadInputStatus.INACTIVE_EVENT))).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert valid.prediction.insufficient_data_reason is None
    assert inactive.prediction.insufficient_data_reason is None


# ---------------------------------------------------------------------------
# INACTIVE_EVENT
# ---------------------------------------------------------------------------


def test_inactive_event_does_not_call_calculator():
    calculator = FakeCalculator()

    make_agent(
        FakeInputService(non_ready_input_result(FireSpreadInputStatus.INACTIVE_EVENT)), calculator
    ).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert calculator.calls == []


def test_inactive_event_persists_zero_cells_with_inactive_status():
    repository = FakeRepository()
    result = make_agent(
        FakeInputService(non_ready_input_result(FireSpreadInputStatus.INACTIVE_EVENT)), repository=repository
    ).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert result.prediction.status is FireSpreadPredictionStatus.INACTIVE_EVENT
    assert result.prediction.cells == ()
    assert result.prediction.severity_assessment_id is None
    assert len(repository.calls) == 1


# ---------------------------------------------------------------------------
# Determinism, validation, error propagation
# ---------------------------------------------------------------------------


def test_same_inputs_produce_equivalent_prediction():
    first = make_agent(FakeInputService(ready_input_result())).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )
    second = make_agent(FakeInputService(ready_input_result())).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert first.prediction == second.prediction


def test_predicted_at_equals_as_of_exactly():
    result = make_agent(FakeInputService(ready_input_result())).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert result.prediction.predicted_at is AS_OF


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_is_rejected(invalid_fire_event_id):
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_input_result())).predict(
            fire_event_id=invalid_fire_event_id, as_of=AS_OF, horizon_minutes=30
        )


def test_naive_as_of_is_rejected():
    with pytest.raises(ValueError):
        make_agent(FakeInputService(ready_input_result())).predict(
            fire_event_id=FIRE_EVENT_ID, as_of=datetime(2026, 9, 14, 12, 0), horizon_minutes=30
        )


def test_input_service_exception_propagates_without_calculation_or_persistence():
    calculator = FakeCalculator()
    repository = FakeRepository()
    agent = make_agent(FakeInputService(exc=RuntimeError("input exploded")), calculator, repository)

    with pytest.raises(RuntimeError, match="input exploded"):
        agent.predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert calculator.calls == []
    assert repository.calls == []


def test_calculator_exception_propagates_without_persistence():
    calculator = FakeCalculator(exc=RuntimeError("calculation exploded"))
    repository = FakeRepository()
    agent = make_agent(FakeInputService(ready_input_result()), calculator, repository)

    with pytest.raises(RuntimeError, match="calculation exploded"):
        agent.predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert repository.calls == []


def test_repository_exception_propagates_after_building_domain_prediction():
    repository = FakeRepository(exc=RuntimeError("persistence exploded"))
    agent = make_agent(FakeInputService(ready_input_result()), repository=repository)

    with pytest.raises(RuntimeError, match="persistence exploded"):
        agent.predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)

    assert repository.calls[0]["prediction"].status is FireSpreadPredictionStatus.VALID


def test_ready_result_without_input_data_fails_explicitly():
    input_result = SimpleNamespace(
        status=FireSpreadInputStatus.READY,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        severity_assessment_id=500,
        weather_observation_id=101,
    )

    with pytest.raises(ValueError, match="requires input_data"):
        make_agent(FakeInputService(input_result)).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)


def test_ready_result_without_severity_assessment_id_fails_explicitly():
    input_result = SimpleNamespace(
        status=FireSpreadInputStatus.READY,
        input_data=INPUT_DATA,
        fire_event_id=FIRE_EVENT_ID,
        severity_assessment_id=None,
        weather_observation_id=101,
    )

    with pytest.raises(ValueError, match="requires severity_assessment_id"):
        make_agent(FakeInputService(input_result)).predict(fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30)


def test_unknown_input_status_fails_explicitly_without_persistence():
    repository = FakeRepository()
    input_result = SimpleNamespace(
        status=SimpleNamespace(value="unknown"),
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        severity_assessment_id=None,
        weather_observation_id=None,
    )

    with pytest.raises(ValueError, match="Unsupported fire-spread input status"):
        make_agent(FakeInputService(input_result), repository=repository).predict(
            fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
        )

    assert repository.calls == []


def test_repository_return_type_contains_stored_prediction():
    result = make_agent(FakeInputService(ready_input_result())).predict(
        fire_event_id=FIRE_EVENT_ID, as_of=AS_OF, horizon_minutes=30
    )

    assert isinstance(result, StoredFireSpreadPrediction)
    assert result.weather_observation_id == 101
