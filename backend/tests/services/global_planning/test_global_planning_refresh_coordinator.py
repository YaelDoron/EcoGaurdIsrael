"""Unit tests for GlobalPlanningRefreshCoordinator using fakes (Stage 6 of
the Global Multi-Incident Optimizer refactor, Tasks 18-19, 33) - the bounded
stale-retry exhaustion path and NO_ACTIVE_EVENTS are exercised precisely
here; the full real-DB ACTIVATED/NO_OP/dynamic-scenario paths are covered by
tests/integration/test_global_planning_refresh_coordinator_integration.py.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.services.global_planning.global_planning_refresh_coordinator import (
    GlobalPlanningRefreshCoordinator,
    GlobalPlanningRefreshStatus,
    MAX_GLOBAL_STALE_RETRIES,
)
from src.services.global_planning.global_response_plan_activation_service import GlobalPlanningStaleInput

AS_OF = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)


class _FakeFireEventRepository:
    def __init__(self, active_ids_sequence):
        self._sequence = list(active_ids_sequence)
        self.calls = 0

    def get_active_fire_event_ids(self):
        index = min(self.calls, len(self._sequence) - 1)
        self.calls += 1
        return self._sequence[index]


class _FakeGlobalPlanningRunRepository:
    def __init__(self):
        self._next_id = 1
        self.created_runs = []
        self.completed = {}
        self.fingerprints = {}
        self.member_results = []

    def create_run(self, *, started_at, trigger, methodology, methodology_version, input_fingerprint, fire_event_ids):
        run_id = self._next_id
        self._next_id += 1
        self.created_runs.append(run_id)

        class _Stored:
            id = run_id

        return _Stored()

    def set_input_fingerprint(self, run_id, fingerprint):
        self.fingerprints[run_id] = fingerprint

    def record_member_result(self, run_id, fire_event_id, *, result_status, response_plan_id, local_state_fingerprint, error_code):
        self.member_results.append((run_id, fire_event_id, result_status))

    def complete_run(self, run_id, *, status, completed_at):
        self.completed[run_id] = status

    def get_latest_activated(self, *, exclude_run_id=None):
        return None


class _FakeInputBuilder:
    def __init__(self):
        self.calls = 0

    def build(self, *, global_planning_run_id, as_of):
        self.calls += 1

        class _FakeInput:
            input_fingerprint = "f" * 64
            current_assignments = ()

        return _FakeInput()


class _AlwaysStaleActivationService:
    """Every activate() call raises GlobalPlanningStaleInput."""

    def __init__(self):
        self.calls = 0

    def activate(self, *, global_planning_input, global_optimization_result, as_of, run_history=None):
        self.calls += 1
        raise GlobalPlanningStaleInput("simulated staleness")


class _FakeOptimizationService:
    def __init__(self):
        self.calls = 0

    def optimize(self, global_planning_input, config, demand_scoring_policy, severity_demand_policy, stability_policy):
        self.calls += 1

        class _FakeResult:
            actions = ()
            assignment_changes = ()
            shortage = None

        return _FakeResult()


def _make_coordinator(fire_event_ids_sequence, activation_service=None, run_repository=None):
    return GlobalPlanningRefreshCoordinator(
        fire_event_repository=_FakeFireEventRepository(fire_event_ids_sequence),
        global_planning_run_repository=run_repository or _FakeGlobalPlanningRunRepository(),
        input_builder=_FakeInputBuilder(),
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service or _AlwaysStaleActivationService(),
    )


def test_no_active_events_returns_immediately_without_creating_a_run():
    run_repository = _FakeGlobalPlanningRunRepository()
    coordinator = _make_coordinator(((),), run_repository=run_repository)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS
    assert run_repository.created_runs == []


def test_stale_retries_are_bounded_and_exhausted(monkeypatch):
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysStaleActivationService()
    coordinator = _make_coordinator(((1,), (1,), (1,)), activation_service=activation_service, run_repository=run_repository)

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.STALE_RETRY_EXHAUSTED
    assert result.retry_count == MAX_GLOBAL_STALE_RETRIES + 1
    # Exactly one full cycle attempt per retry - never partial/per-event patching.
    assert activation_service.calls == MAX_GLOBAL_STALE_RETRIES + 1
    assert len(run_repository.created_runs) == MAX_GLOBAL_STALE_RETRIES + 1
    for run_id in run_repository.created_runs:
        assert run_repository.completed[run_id].value == "failed"


def test_stale_retry_recaptures_active_set_between_attempts():
    """If the active set shrinks to empty between retry attempts (e.g. the
    only FireEvent resolved), the coordinator must report NO_ACTIVE_EVENTS
    rather than attempting a doomed cycle with an empty set."""
    run_repository = _FakeGlobalPlanningRunRepository()
    activation_service = _AlwaysStaleActivationService()
    fire_event_repository = _FakeFireEventRepository(((1,), ()))
    coordinator = GlobalPlanningRefreshCoordinator(
        fire_event_repository=fire_event_repository,
        global_planning_run_repository=run_repository,
        input_builder=_FakeInputBuilder(),
        optimization_service=_FakeOptimizationService(),
        activation_service=activation_service,
    )

    result = coordinator.refresh(trigger="manual", as_of=AS_OF)

    assert result.status is GlobalPlanningRefreshStatus.NO_ACTIVE_EVENTS
    # Only the first attempt actually ran a cycle before the set emptied out.
    assert activation_service.calls == 1


def test_invalid_trigger_rejected():
    coordinator = _make_coordinator(((1,),))
    with pytest.raises(ValueError):
        coordinator.refresh(trigger="", as_of=AS_OF)


def test_invalid_as_of_rejected():
    coordinator = _make_coordinator(((1,),))
    with pytest.raises(ValueError):
        coordinator.refresh(trigger="manual", as_of=datetime(2026, 1, 1))
