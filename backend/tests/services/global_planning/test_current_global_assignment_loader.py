"""Tests for CurrentGlobalAssignmentLoader (Stage 6 of the Global
Multi-Incident Optimizer refactor, Task 14), using fakes - mirrors
test_current_response_plan_resolver.py's fake-based pattern so the join
logic (ResponsePlan actions + ResourceCommitment.dispatch_state ->
CurrentGlobalAssignment) is exercised without a real database.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.models import ResponseAction, ResponsePlan, ResponsePlanStatus
from src.models.dispatch_state import DispatchState
from src.models.resource_commitment import ResourceCommitment
from src.repositories.response_plan_repository import StoredResponsePlan
from src.services.global_planning.current_global_assignment_loader import CurrentGlobalAssignmentLoader

AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


class FakeResolver:
    def __init__(self, plans_by_event: dict[int, StoredResponsePlan]) -> None:
        self._plans_by_event = plans_by_event
        self.calls: list[int] = []

    def resolve(self, *, fire_event_id: int):
        self.calls.append(fire_event_id)
        return self._plans_by_event.get(fire_event_id)


class FakeCommitmentRepository:
    def __init__(self, commitments_by_event: dict[int, tuple[ResourceCommitment, ...]]) -> None:
        self._commitments_by_event = commitments_by_event
        self.calls: list[int] = []

    def get_for_fire_event(self, fire_event_id: int) -> tuple[ResourceCommitment, ...]:
        self.calls.append(fire_event_id)
        return self._commitments_by_event.get(fire_event_id, ())


def make_plan(plan_id: int, fire_event_id: int, actions: tuple[ResponseAction, ...]) -> StoredResponsePlan:
    plan = ResponsePlan(
        fire_event_id=fire_event_id,
        response_target_set_id=1,
        route_planning_run_id=601,
        generated_at=AS_OF,
        status=ResponsePlanStatus.COMPLETE if actions else ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
        methodology="global_genetic_resource_allocation",
        methodology_version="1.0",
        random_seed=42,
        actions=actions,
        uncovered_target_ids=(),
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=200.0,
    )
    return StoredResponsePlan(id=plan_id, plan=plan)


def make_commitment(resource_id: str, fire_event_id: int, dispatch_state: DispatchState) -> ResourceCommitment:
    return ResourceCommitment(
        resource_id=resource_id, fire_event_id=fire_event_id, response_plan_id=1,
        committed_at=AS_OF, dispatch_state=dispatch_state,
    )


def test_loads_one_assignment_per_committed_resource_on_the_current_plan():
    plan = make_plan(1, fire_event_id=10, actions=(ResponseAction("R1", 100, 500),))
    resolver = FakeResolver({10: plan})
    commitments = FakeCommitmentRepository({10: (make_commitment("R1", 10, DispatchState.DISPATCHED),)})
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    (assignment,) = loader.load((10,))

    assert assignment.resource_id == "R1"
    assert assignment.fire_event_id == 10
    assert assignment.response_target_id == 100
    assert assignment.response_plan_id == 1
    assert assignment.dispatch_state is DispatchState.DISPATCHED


def test_no_current_plan_yields_no_assignments_for_that_event():
    resolver = FakeResolver({})
    commitments = FakeCommitmentRepository({})
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    assert loader.load((10,)) == ()


def test_action_without_a_matching_commitment_is_skipped_defensively():
    plan = make_plan(1, fire_event_id=10, actions=(ResponseAction("R1", 100, 500),))
    resolver = FakeResolver({10: plan})
    commitments = FakeCommitmentRepository({10: ()})  # no commitment at all for R1
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    assert loader.load((10,)) == ()


def test_multiple_events_are_each_loaded_independently():
    plan_a = make_plan(1, fire_event_id=10, actions=(ResponseAction("R1", 100, 500),))
    plan_b = make_plan(2, fire_event_id=20, actions=(ResponseAction("R2", 200, 600),))
    resolver = FakeResolver({10: plan_a, 20: plan_b})
    commitments = FakeCommitmentRepository(
        {
            10: (make_commitment("R1", 10, DispatchState.DISPATCHED),),
            20: (make_commitment("R2", 20, DispatchState.PLANNED),),
        }
    )
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    assignments = loader.load((10, 20))

    by_resource = {assignment.resource_id: assignment for assignment in assignments}
    assert by_resource["R1"].fire_event_id == 10
    assert by_resource["R1"].dispatch_state is DispatchState.DISPATCHED
    assert by_resource["R2"].fire_event_id == 20
    assert by_resource["R2"].dispatch_state is DispatchState.PLANNED


def test_multiple_actions_on_one_plan_each_produce_an_assignment():
    plan = make_plan(1, fire_event_id=10, actions=(ResponseAction("R1", 100, 500), ResponseAction("R2", 100, 501)))
    resolver = FakeResolver({10: plan})
    commitments = FakeCommitmentRepository(
        {10: (make_commitment("R1", 10, DispatchState.DISPATCHED), make_commitment("R2", 10, DispatchState.DISPATCHED))}
    )
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    assignments = loader.load((10,))

    assert {a.resource_id for a in assignments} == {"R1", "R2"}


def test_load_queries_only_the_given_fire_event_ids():
    resolver = FakeResolver({})
    commitments = FakeCommitmentRepository({})
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    loader.load((5, 6))

    assert resolver.calls == [5, 6]


def test_empty_fire_event_ids_yields_no_assignments_and_no_calls():
    resolver = FakeResolver({})
    commitments = FakeCommitmentRepository({})
    loader = CurrentGlobalAssignmentLoader(resolver, commitments)

    assert loader.load(()) == ()
    assert resolver.calls == []
