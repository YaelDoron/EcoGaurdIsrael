"""Tests for FireSpreadRefreshOrchestrator no-op and refresh behavior."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ast
from pathlib import Path

import pytest

from src.models import (
    FireSpreadEffectiveState,
    FireSpreadFuelClass,
    FireSpreadInput,
    FireSpreadInputResult,
    FireSpreadInputStatus,
    FireSpreadInsufficientDataReason,
    FireSpreadPrediction,
    FireSpreadPredictionStatus,
    OperationalRefreshTriggerType,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPredictionWithCells, StoredFireSpreadPrediction
from src.services.operational_refresh import FireSpreadRefreshHorizonStatus, FireSpreadRefreshOrchestrator

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
FIRE_EVENT_ID = 42


class FakeInputService:
    def __init__(self, results_by_horizon) -> None:
        self.results_by_horizon = dict(results_by_horizon)
        self.calls = []
        self.shared_context_calls = []
        self.shared_context_for_event_calls = []

    def prepare_shared_context(self, fire_event_id, as_of):
        self.shared_context_calls.append({"fire_event_id": fire_event_id, "as_of": as_of})
        return {"fire_event_id": fire_event_id, "as_of": as_of}

    def prepare_shared_context_for_event(self, stored_event, as_of, *, resolved_severity=None):
        self.shared_context_for_event_calls.append(
            {"fire_event_id": stored_event.id, "as_of": as_of, "resolved_severity": resolved_severity}
        )
        return {"fire_event_id": stored_event.id, "as_of": as_of}

    def build_input_for_horizon(self, shared_context, horizon_minutes):
        self.calls.append(
            {
                "fire_event_id": shared_context["fire_event_id"],
                "as_of": shared_context["as_of"],
                "horizon_minutes": horizon_minutes,
            }
        )
        return self.results_by_horizon[horizon_minutes]

    def prepare_input(self, fire_event_id, as_of, horizon_minutes):
        context = self.prepare_shared_context(fire_event_id, as_of)
        return self.build_input_for_horizon(context, horizon_minutes)


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []
        self.next_id = 100

    def predict_from_input_result(self, *, input_result, as_of, horizon_minutes, effective_state_fingerprint=None):
        self.calls.append(
            {
                "input_result": input_result,
                "as_of": as_of,
                "horizon_minutes": horizon_minutes,
                "effective_state_fingerprint": effective_state_fingerprint,
            }
        )
        self.next_id += 1
        status = {
            FireSpreadInputStatus.READY: FireSpreadPredictionStatus.VALID,
            FireSpreadInputStatus.INSUFFICIENT_DATA: FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            FireSpreadInputStatus.INACTIVE_EVENT: FireSpreadPredictionStatus.INACTIVE_EVENT,
        }[input_result.status]
        return StoredFireSpreadPrediction(
            id=self.next_id,
            prediction=FireSpreadPrediction(
                fire_event_id=input_result.fire_event_id,
                severity_assessment_id=input_result.severity_assessment_id,
                predicted_at=as_of,
                horizon_minutes=horizon_minutes,
                status=status,
                methodology=METHODOLOGY_NAME,
                methodology_version=METHODOLOGY_VERSION,
                cells=(),
                effective_state_fingerprint=effective_state_fingerprint,
            ),
            weather_observation_id=input_result.weather_observation_id,
        )


class FakeRepository:
    def __init__(self, latest_by_horizon=None) -> None:
        self.latest_by_horizon = dict(latest_by_horizon or {})
        self.calls = []
        self.batch_calls = []

    def get_latest_for_event_and_horizon_as_of(self, fire_event_id, horizon_minutes, as_of):
        self.calls.append({"fire_event_id": fire_event_id, "horizon_minutes": horizon_minutes, "as_of": as_of})
        return self.latest_by_horizon.get(horizon_minutes)

    def get_latest_for_event_and_horizons_as_of(self, fire_event_id, horizons, as_of):
        """Mirrors the production repository's batched method - the
        orchestrator now prefetches "latest per horizon" ONCE per refresh
        (instead of once per horizon) via this method."""
        self.batch_calls.append({"fire_event_id": fire_event_id, "horizons": tuple(horizons), "as_of": as_of})
        return {
            horizon_minutes: self.latest_by_horizon[horizon_minutes]
            for horizon_minutes in horizons
            if horizon_minutes in self.latest_by_horizon
        }


def ready_result(input_data=None, *, fire_event_id=FIRE_EVENT_ID) -> FireSpreadInputResult:
    return FireSpreadInputResult(
        status=FireSpreadInputStatus.READY,
        input_data=input_data or make_input(),
        fire_event_id=fire_event_id,
        severity_assessment_id=500,
        weather_observation_id=101,
    )


def non_ready_result(
    status: FireSpreadInputStatus,
    insufficient_data_reason: FireSpreadInsufficientDataReason = FireSpreadInsufficientDataReason.MISSING_VEGETATION,
) -> FireSpreadInputResult:
    return FireSpreadInputResult(
        status=status,
        input_data=None,
        fire_event_id=FIRE_EVENT_ID,
        insufficient_data_reason=(
            insufficient_data_reason if status is FireSpreadInputStatus.INSUFFICIENT_DATA else None
        ),
    )


def make_input(**overrides) -> FireSpreadInput:
    defaults = dict(
        origin_latitude=32.731,
        origin_longitude=35.046,
        wind_speed_kmh=21.6,
        wind_direction_deg=270.0,
        fuel_moisture_percent=12.0,
        fuel_class=FireSpreadFuelClass.SHRUBS,
        horizon_minutes=30,
    )
    defaults.update(overrides)
    return FireSpreadInput(**defaults)


def stored_prediction(
    *,
    prediction_id=10,
    fire_event_id=FIRE_EVENT_ID,
    horizon_minutes=30,
    status=FireSpreadPredictionStatus.VALID,
    fingerprint=None,
    predicted_at=AS_OF,
    insufficient_data_reason=None,
) -> StoredFireSpreadPredictionWithCells:
    return StoredFireSpreadPredictionWithCells(
        id=prediction_id,
        prediction=FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=500 if status is FireSpreadPredictionStatus.VALID else None,
            predicted_at=predicted_at,
            horizon_minutes=horizon_minutes,
            status=status,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=(),
            effective_state_fingerprint=fingerprint,
            insufficient_data_reason=insufficient_data_reason,
        ),
        cells=(),
        weather_observation_id=101 if status is FireSpreadPredictionStatus.VALID else None,
    )


def make_orchestrator(input_service, agent=None, repository=None):
    return FireSpreadRefreshOrchestrator(
        input_service=input_service,
        prediction_agent=agent or FakeAgent(),
        prediction_repository=repository or FakeRepository(),
    )


def test_resource_status_update_causes_zero_input_or_agent_calls():
    input_service = FakeInputService({30: ready_result()})
    agent = FakeAgent()
    repository = FakeRepository()
    orchestrator = make_orchestrator(input_service, agent, repository)

    result = orchestrator.refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.RESOURCE_STATUS_UPDATE,
        as_of=AS_OF,
    )

    assert result.reevaluation_required is False
    assert input_service.calls == []
    assert agent.calls == []
    assert repository.calls == []


@pytest.mark.parametrize(
    "trigger_type",
    [
        OperationalRefreshTriggerType.WEATHER_UPDATE,
        OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        OperationalRefreshTriggerType.SEVERITY_UPDATE,
    ],
)
def test_environmental_triggers_reevaluate(trigger_type):
    input_service = FakeInputService({30: ready_result()})
    orchestrator = make_orchestrator(input_service)

    result = orchestrator.refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=trigger_type,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.reevaluation_required is True
    assert input_service.calls == [{"fire_event_id": FIRE_EVENT_ID, "as_of": AS_OF, "horizon_minutes": 30}]


def test_both_horizons_prefetch_latest_predictions_in_one_batched_call():
    """Performance pass: the "latest per horizon" read used to happen once
    per horizon (2 separate repository calls for 30m/60m). It must now be
    exactly one batched call covering both requested horizons."""
    input_service = FakeInputService({30: ready_result(), 60: ready_result()})
    repository = FakeRepository()

    make_orchestrator(input_service, repository=repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30, 60),
    )

    assert repository.batch_calls == [{"fire_event_id": FIRE_EVENT_ID, "horizons": (30, 60), "as_of": AS_OF}]
    # Neither horizon pays its own singular "latest" lookup for the decision -
    # only REFRESHED's per-horizon follow-up read would ever add to .calls.
    assert len(repository.calls) == 2  # both horizons recompute (nothing pre-seeded) -> 1 follow-up read each


def test_identical_ready_fingerprint_is_no_op_without_agent_call():
    input_data = make_input()
    fingerprint = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=input_data).fingerprint
    input_service = FakeInputService({30: ready_result(input_data)})
    agent = FakeAgent()
    repository = FakeRepository({30: stored_prediction(fingerprint=fingerprint)})

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.NO_OP
    assert result.horizon_results[0].previous_prediction_id == 10
    assert result.horizon_results[0].effective_state_fingerprint == fingerprint
    assert agent.calls == []


def test_repeated_identical_refresh_remains_no_op():
    input_data = make_input()
    fingerprint = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=input_data).fingerprint
    input_service = FakeInputService({30: ready_result(input_data)})
    agent = FakeAgent()
    repository = FakeRepository({30: stored_prediction(fingerprint=fingerprint)})
    orchestrator = make_orchestrator(input_service, agent, repository)

    first = orchestrator.refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )
    second = orchestrator.refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert first.no_ops == 1
    assert second.no_ops == 1
    assert agent.calls == []


@pytest.mark.parametrize(
    "input_data",
    [
        make_input(wind_speed_kmh=35.0),
        make_input(wind_direction_deg=280.0),
        make_input(fuel_moisture_percent=20.0),
        make_input(fuel_class=FireSpreadFuelClass.GRASSLAND),
        make_input(origin_latitude=32.8),
        make_input(origin_longitude=35.1),
    ],
)
def test_changed_ready_state_invokes_existing_agent_with_exact_fingerprint(input_data):
    old_fingerprint = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=make_input()).fingerprint
    current_fingerprint = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=input_data).fingerprint
    input_service = FakeInputService({30: ready_result(input_data)})
    agent = FakeAgent()
    repository = FakeRepository({30: stored_prediction(fingerprint=old_fingerprint)})

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.REFRESHED
    assert result.horizon_results[0].effective_state_fingerprint == current_fingerprint
    assert agent.calls[0]["effective_state_fingerprint"] == current_fingerprint
    assert len(input_service.calls) == 1
    assert len(agent.calls) == 1


def test_legacy_null_fingerprint_causes_one_refresh_instead_of_false_no_op():
    input_service = FakeInputService({30: ready_result()})
    agent = FakeAgent()
    repository = FakeRepository({30: stored_prediction(fingerprint=None)})

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.REFRESHED
    assert len(agent.calls) == 1


def test_horizon_comparison_is_independent():
    input30 = make_input(horizon_minutes=30)
    input60 = make_input(horizon_minutes=60)
    fp30 = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=input30).fingerprint
    wrong_horizon_fp = FireSpreadEffectiveState.from_input(fire_event_id=FIRE_EVENT_ID, spread_input=input60).fingerprint
    input_service = FakeInputService({30: ready_result(input30), 60: ready_result(input60)})
    agent = FakeAgent()
    repository = FakeRepository(
        {
            30: stored_prediction(horizon_minutes=30, fingerprint=wrong_horizon_fp),
            60: stored_prediction(prediction_id=60, horizon_minutes=60, fingerprint=wrong_horizon_fp),
        }
    )

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
        horizons=(60, 30, 30),
    )

    assert [item.horizon_minutes for item in result.horizon_results] == [30, 60]
    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.REFRESHED
    assert result.horizon_results[0].effective_state_fingerprint == fp30
    assert result.horizon_results[1].status is FireSpreadRefreshHorizonStatus.NO_OP


# --- performance pass: shared-context loaded once, not once per horizon ---
#
# Root cause this guards against: prepare_input() used to be called
# independently per horizon, re-loading the identical FireEvent/severity/
# weather each time - a pure N+1 that profiling showed doubled Spread's
# input-loading cost for no benefit, since none of that loaded data depends
# on horizon_minutes.


def test_shared_context_is_loaded_exactly_once_regardless_of_horizon_count():
    input_service = FakeInputService({30: ready_result(make_input(horizon_minutes=30)), 60: ready_result(make_input(horizon_minutes=60))})
    agent = FakeAgent()
    repository = FakeRepository({})

    make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
        horizons=(30, 60),
    )

    assert len(input_service.shared_context_calls) == 1
    assert len(input_service.calls) == 2  # still one build_input_for_horizon() per horizon


def test_both_horizons_still_produce_correct_independent_results_from_one_shared_load():
    input30 = make_input(horizon_minutes=30, wind_speed_kmh=10.0)
    input60 = make_input(horizon_minutes=60, wind_speed_kmh=25.0)
    input_service = FakeInputService({30: ready_result(input30), 60: ready_result(input60)})
    agent = FakeAgent()
    repository = FakeRepository({})

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
        horizons=(30, 60),
    )

    assert [item.horizon_minutes for item in result.horizon_results] == [30, 60]
    assert agent.calls[0]["input_result"].input_data.wind_speed_kmh == 10.0
    assert agent.calls[1]["input_result"].input_data.wind_speed_kmh == 25.0


def test_shared_context_load_failure_marks_every_horizon_failed_not_just_one():
    class _FailingInputService:
        def prepare_shared_context(self, fire_event_id, as_of):
            raise RuntimeError("simulated shared-load DB failure")

    orchestrator = make_orchestrator(_FailingInputService(), FakeAgent(), FakeRepository({}))

    result = orchestrator.refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.FIRE_EVENT_UPDATE,
        as_of=AS_OF,
        horizons=(30, 60),
    )

    assert [item.horizon_minutes for item in result.horizon_results] == [30, 60]
    assert all(item.status is FireSpreadRefreshHorizonStatus.FAILED for item in result.horizon_results)


# --- performance pass: refresh_for_event() (FireEvent/Severity handoff) --


class _FakeStoredEvent:
    def __init__(self, event_id):
        self.id = event_id


def test_refresh_for_event_uses_shared_context_for_event_not_by_id():
    input_service = FakeInputService({30: ready_result()})
    stored_event = _FakeStoredEvent(FIRE_EVENT_ID)

    make_orchestrator(input_service).refresh_for_event(
        stored_event=stored_event,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert input_service.shared_context_calls == []  # the by-id path is never used
    assert len(input_service.shared_context_for_event_calls) == 1
    assert input_service.shared_context_for_event_calls[0]["fire_event_id"] == FIRE_EVENT_ID


def test_refresh_for_event_forwards_resolved_severity_to_input_service():
    input_service = FakeInputService({30: ready_result()})
    stored_event = _FakeStoredEvent(FIRE_EVENT_ID)
    sentinel_severity = object()

    make_orchestrator(input_service).refresh_for_event(
        stored_event=stored_event,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        resolved_severity=sentinel_severity,
        horizons=(30,),
    )

    assert input_service.shared_context_for_event_calls[0]["resolved_severity"] is sentinel_severity


def test_refresh_for_event_no_op_populates_resolved_prediction_from_latest_without_extra_reads():
    input_service = FakeInputService({30: ready_result()})
    stored_event = _FakeStoredEvent(FIRE_EVENT_ID)
    existing_fingerprint = FireSpreadEffectiveState.from_input(
        fire_event_id=FIRE_EVENT_ID, spread_input=make_input()
    ).fingerprint
    latest = stored_prediction(fingerprint=existing_fingerprint)
    repository = FakeRepository({30: latest})

    result = make_orchestrator(input_service, repository=repository).refresh_for_event(
        stored_event=stored_event,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.NO_OP
    assert result.horizon_results[0].resolved_prediction is latest
    # The "latest per horizon" read is now one batched call up front; NO_OP
    # never pays a second (singular, follow-up) read for the handoff object.
    assert len(repository.batch_calls) == 1
    assert repository.calls == []


def test_refresh_for_event_refreshed_populates_resolved_prediction_via_one_follow_up_read():
    input_service = FakeInputService({30: ready_result()})
    stored_event = _FakeStoredEvent(FIRE_EVENT_ID)
    repository = FakeRepository({})

    result = make_orchestrator(input_service, repository=repository).refresh_for_event(
        stored_event=stored_event,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.REFRESHED
    # repository fake returns None for both reads here (nothing pre-seeded),
    # so resolved_prediction is None - the structural point is the batched
    # initial read (1 batch call) plus the one singular follow-up read,
    # covered by test_future_prediction_is_not_used_to_suppress_historical_refresh.
    assert len(repository.batch_calls) == 1
    assert len(repository.calls) == 1


@pytest.mark.parametrize(
    ("input_status", "prediction_status", "refresh_status"),
    [
        (
            FireSpreadInputStatus.INSUFFICIENT_DATA,
            FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            FireSpreadRefreshHorizonStatus.INSUFFICIENT_DATA,
        ),
        (
            FireSpreadInputStatus.INACTIVE_EVENT,
            FireSpreadPredictionStatus.INACTIVE_EVENT,
            FireSpreadRefreshHorizonStatus.INACTIVE_EVENT,
        ),
    ],
)
def test_non_ready_state_persisted_when_latest_was_valid(input_status, prediction_status, refresh_status):
    input_service = FakeInputService({30: non_ready_result(input_status)})
    agent = FakeAgent()
    repository = FakeRepository({30: stored_prediction(status=FireSpreadPredictionStatus.VALID, fingerprint="a" * 64)})

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is refresh_status
    assert agent.calls[0]["effective_state_fingerprint"] is None


@pytest.mark.parametrize(
    ("input_status", "prediction_status"),
    [
        (FireSpreadInputStatus.INSUFFICIENT_DATA, FireSpreadPredictionStatus.INSUFFICIENT_DATA),
        (FireSpreadInputStatus.INACTIVE_EVENT, FireSpreadPredictionStatus.INACTIVE_EVENT),
    ],
)
def test_repeated_same_non_ready_state_is_no_op(input_status, prediction_status):
    # Same status AND same reason (missing_vegetation / None for inactive).
    stored_reason = (
        FireSpreadInsufficientDataReason.MISSING_VEGETATION
        if prediction_status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
        else None
    )
    input_service = FakeInputService({30: non_ready_result(input_status)})
    agent = FakeAgent()
    repository = FakeRepository(
        {30: stored_prediction(status=prediction_status, insufficient_data_reason=stored_reason)}
    )

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.NO_OP
    assert agent.calls == []


@pytest.mark.parametrize(
    ("stored_reason", "new_reason"),
    [
        # A. historical row without a reason -> known reason
        (None, FireSpreadInsufficientDataReason.MISSING_VEGETATION),
        # B. reason changed while the status stayed insufficient_data
        (FireSpreadInsufficientDataReason.MISSING_VEGETATION, FireSpreadInsufficientDataReason.STALE_WEATHER),
    ],
)
def test_insufficient_data_with_changed_reason_persists_a_new_row(stored_reason, new_reason):
    input_service = FakeInputService({30: non_ready_result(FireSpreadInputStatus.INSUFFICIENT_DATA, new_reason)})
    agent = FakeAgent()
    repository = FakeRepository(
        {30: stored_prediction(status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, insufficient_data_reason=stored_reason)}
    )

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.SEVERITY_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    horizon_result = result.horizon_results[0]
    assert horizon_result.status is FireSpreadRefreshHorizonStatus.INSUFFICIENT_DATA
    assert horizon_result.previous_prediction_id == 10
    assert len(agent.calls) == 1
    assert agent.calls[0]["input_result"].insufficient_data_reason is new_reason


def test_future_prediction_is_not_used_to_suppress_historical_refresh():
    input_service = FakeInputService({30: ready_result()})
    agent = FakeAgent()
    repository = FakeRepository()

    result = make_orchestrator(input_service, agent, repository).refresh(
        fire_event_id=FIRE_EVENT_ID,
        trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
        as_of=AS_OF,
        horizons=(30,),
    )

    # The initial "latest per horizon" read (used for the NO_OP/fingerprint
    # decision) is now a single batched call covering all requested horizons.
    # REFRESHED then performs one more (singular, per-horizon) follow-up read
    # to obtain the WithCells handoff object for Response Targets (see
    # resolved_prediction's docstring) - the NO_OP path does not pay this cost.
    assert repository.batch_calls == [{"fire_event_id": FIRE_EVENT_ID, "horizons": (30,), "as_of": AS_OF}]
    assert repository.calls == [{"fire_event_id": FIRE_EVENT_ID, "horizon_minutes": 30, "as_of": AS_OF}]
    assert result.horizon_results[0].status is FireSpreadRefreshHorizonStatus.REFRESHED


def test_naive_as_of_rejected_before_input_service_call():
    input_service = FakeInputService({30: ready_result()})

    with pytest.raises(ValueError):
        make_orchestrator(input_service).refresh(
            fire_event_id=FIRE_EVENT_ID,
            trigger_type=OperationalRefreshTriggerType.WEATHER_UPDATE,
            as_of=datetime(2026, 9, 14, 12, 0),
            horizons=(30,),
        )

    assert input_service.calls == []


def test_refresh_modules_do_not_import_out_of_scope_systems():
    forbidden_fragments = (
        "simulation",
        "scripts",
        "road_network",
        "RoadNetworkRepository",
        "OperationalContextService",
        "ResponseTarget",
        "response_target",
        "routing",
        "Dijkstra",
        "genetic",
        "external",
        "fire_spread_calculator",
    )
    production_files = [
        (Path(__file__).resolve().parents[4] / "backend/src/services/operational_refresh/fire_spread_refresh_orchestrator.py"),
        (Path(__file__).resolve().parents[4] / "backend/src/services/operational_refresh/fire_spread_refresh_result.py"),
    ]

    violations = []
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if module and any(fragment in module for fragment in forbidden_fragments):
                violations.append((str(path), module))

    assert violations == []
