"""Tests for GlobalResponsePlanActivationService (Stage 6 of the Global
Multi-Incident Optimizer refactor, Tasks 26-29, 51), against real SQLite -
one atomic transaction persisting RoutePlanningRun/ResponsePlan/
ResourceCommitment(DISPATCHED)/sidecar for every FireEvent the Global GA
result touches, or nothing at all.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_service import (
    GlobalResponseOptimizationService,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.models.dispatch_state import DispatchState
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.resource_status import ResourceStatus
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.global_planning.global_response_plan_activation_service import (
    GlobalPlanningStaleInput,
    GlobalResponsePlanActivationService,
    GlobalRunHistoryContext,
)
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict
from tests.calculators.global_response_optimization.helpers import make_demand, make_input, make_resource, make_route, make_target

AS_OF = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
RELIABLE_CONFIG = GlobalResponseOptimizationConfig(population_size=20, generation_count=20, random_seed=42)


def _persist_fire_event(session, latitude=32.7, longitude=35.0, status="confirmed") -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF, updated_at=AS_OF,
        status=status, detection_confidence=0.9, methodology="m", methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id, latitude=32.7, longitude=35.0) -> tuple[int, int]:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    target = ResponseTargetDB(
        response_target_set_id=target_set.id, fire_event_id=fire_event_id, target_order=1,
        target_type="active_fire", latitude=latitude, longitude=longitude, priority_score=100.0,
    )
    session.add(target)
    session.flush()
    return target_set.id, target.id


def _persist_graph_nodes(session) -> None:
    """Persists the two GraphNode ids make_route()'s default node_path=(1, 2)
    refers to - RouteResultDB.source_node_id/target_node_id are real FKs."""
    for node_id in (1, 2):
        if session.get(GraphNodeDB, node_id) is None:
            session.add(GraphNodeDB(id=node_id, latitude=32.7, longitude=35.0))
    session.flush()


def _persist_resource(session, resource_id, station_id="S1", status=ResourceStatus.AVAILABLE) -> None:
    if session.get(FireStationDB, station_id) is None:
        session.add(FireStationDB(id=station_id, name=station_id, latitude=32.7, longitude=35.0))
        session.flush()
    session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=status))
    _persist_graph_nodes(session)


def _make_service(sqlite_session_factory) -> GlobalResponsePlanActivationService:
    return GlobalResponsePlanActivationService(
        response_plan_repository=ResponsePlanRepository(sqlite_session_factory),
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(sqlite_session_factory),
        route_planning_repository=RoutePlanningRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        session_factory=sqlite_session_factory,
    )


def _create_global_run(sqlite_session_factory, fire_event_ids) -> int:
    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.create_run(
        started_at=AS_OF, trigger="manual", methodology="global_genetic_resource_allocation",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=fire_event_ids,
    )
    return stored_run.id


def _optimize(global_input):
    return GlobalResponseOptimizationService().optimize(global_input, RELIABLE_CONFIG)


# ---------------------------------------------------------------------------
# Happy path - single event
# ---------------------------------------------------------------------------


def test_single_event_activation_persists_plan_route_run_and_dispatched_commitment(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    activation = _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
    )

    assert activation.response_plan_ids_by_event[fire_event_id] > 0
    plan_id = activation.response_plan_ids_by_event[fire_event_id]

    session = sqlite_session_factory()
    db_plan = session.get(ResponsePlanDB, plan_id)
    assert db_plan.global_planning_run_id == run_id
    assert db_plan.fire_event_id == fire_event_id
    route_run = session.get(RoutePlanningRunDB, db_plan.route_planning_run_id)
    assert route_run.fire_event_id == fire_event_id
    commitment = session.get(ResourceCommitmentDB, "R1")
    assert commitment.fire_event_id == fire_event_id
    assert commitment.response_plan_id == plan_id
    assert commitment.dispatch_state == DispatchState.DISPATCHED.value
    session.close()


def test_activation_makes_the_plan_current_via_the_planning_state_sidecar(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)
    activation = _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
    )
    plan_id = activation.response_plan_ids_by_event[fire_event_id]

    from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver

    resolver = CurrentResponsePlanResolver(
        ResponsePlanRepository(sqlite_session_factory), ResponsePlanPlanningStateRepository(sqlite_session_factory)
    )
    current = resolver.resolve(fire_event_id=fire_event_id)
    assert current is not None
    assert current.id == plan_id


# ---------------------------------------------------------------------------
# Multi-event atomicity
# ---------------------------------------------------------------------------


def test_two_events_activate_atomically_in_one_call(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    event_b = _persist_fire_event(session, 32.90, 35.20)
    target_set_a, target_a = _persist_target_set(session, event_a, 32.70, 35.00)
    target_set_b, target_b = _persist_target_set(session, event_b, 32.90, 35.20)
    _persist_resource(session, "R1")
    _persist_resource(session, "R2")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (event_a, event_b))

    demand_a = make_demand(event_a, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(event_b, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(event_a, event_b),
        targets=(make_target(event_a, target_a), make_target(event_b, target_b)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", event_a, target_a, eta_seconds=10.0),
            make_route("R2", event_b, target_b, eta_seconds=10.0),
        ),
        event_target_set_ids={event_a: target_set_a, event_b: target_set_b},
        incident_demands=(demand_a, demand_b),
    )
    result = _optimize(global_input)

    activation = _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
    )

    assert set(activation.response_plan_ids_by_event) == {event_a, event_b}
    session = sqlite_session_factory()
    all_plans = session.execute(select(ResponsePlanDB)).scalars().all()
    assert len(all_plans) == 2
    all_commitments = session.execute(select(ResourceCommitmentDB)).scalars().all()
    assert len(all_commitments) == 2
    session.close()


# ---------------------------------------------------------------------------
# Stale-input protection (Task 29/30/32)
# ---------------------------------------------------------------------------


def test_rejects_activation_when_fire_event_is_no_longer_active(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    # The FireEvent resolves BEFORE activation runs.
    session = sqlite_session_factory()
    db_event = session.get(FireEventDB, fire_event_id)
    db_event.status = "resolved"
    session.commit()
    session.close()

    with pytest.raises(GlobalPlanningStaleInput):
        _make_service(sqlite_session_factory).activate(
            global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
        )

    session = sqlite_session_factory()
    assert session.execute(select(ResponsePlanDB)).scalars().all() == []
    assert session.execute(select(ResourceCommitmentDB)).scalars().all() == []
    session.close()


def test_rejects_activation_when_a_new_fire_event_became_active(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    # A brand-new FireEvent becomes active BEFORE activation runs.
    session = sqlite_session_factory()
    _persist_fire_event(session, 40.0, 45.0)
    session.commit()
    session.close()

    with pytest.raises(GlobalPlanningStaleInput):
        _make_service(sqlite_session_factory).activate(
            global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
        )


def test_rejects_activation_when_a_competing_cycle_already_committed_a_resource_elsewhere(sqlite_session_factory):
    """Two independent global cycles cover the SAME two-event world (as the
    real architecture requires - one global problem, not per-event
    fragments) but disagree on who gets R1. Whichever activates SECOND must
    be rejected as stale rather than double-committing R1 - Task 52's
    concurrent-activation guarantee, exercised without real thread
    concurrency."""
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    event_b = _persist_fire_event(session, 32.90, 35.20)
    target_set_a, target_a = _persist_target_set(session, event_a, 32.70, 35.00)
    target_set_b, target_b = _persist_target_set(session, event_b, 32.90, 35.20)
    _persist_resource(session, "R1")
    session.commit()
    session.close()

    def _build(run_fire_event_ids, r1_route_target):
        run_id = _create_global_run(sqlite_session_factory, run_fire_event_ids)
        demand_a = make_demand(event_a, minimum_resources=0, desired_resources=1)
        demand_b = make_demand(event_b, minimum_resources=0, desired_resources=1)
        routes = [make_route("R1", event_a, target_a, eta_seconds=10.0)] if r1_route_target == "a" else [
            make_route("R1", event_b, target_b, eta_seconds=10.0)
        ]
        global_input = make_input(
            global_planning_run_id=run_id,
            active_fire_event_ids=(event_a, event_b),
            targets=(make_target(event_a, target_a), make_target(event_b, target_b)),
            resources=(make_resource("R1"),),
            routes=tuple(routes),
            event_target_set_ids={event_a: target_set_a, event_b: target_set_b},
            incident_demands=(demand_a, demand_b),
        )
        return global_input, _optimize(global_input)

    input_giving_r1_to_a, result_giving_r1_to_a = _build((event_a, event_b), "a")
    input_giving_r1_to_b, result_giving_r1_to_b = _build((event_a, event_b), "b")

    # The competing cycle (R1 -> event_b) activates first and wins.
    _make_service(sqlite_session_factory).activate(
        global_planning_input=input_giving_r1_to_b, global_optimization_result=result_giving_r1_to_b, as_of=AS_OF
    )

    # This cycle's own input still says R1 -> event_a - now stale.
    with pytest.raises(GlobalPlanningStaleInput):
        _make_service(sqlite_session_factory).activate(
            global_planning_input=input_giving_r1_to_a, global_optimization_result=result_giving_r1_to_a, as_of=AS_OF
        )

    session = sqlite_session_factory()
    # Exactly one plan for event_a exists - the winning cycle's own
    # (R1-less) plan for it, never a second one from the rejected cycle.
    plans_for_event_a = session.execute(select(ResponsePlanDB).where(ResponsePlanDB.fire_event_id == event_a)).scalars().all()
    assert len(plans_for_event_a) == 1
    commitment = session.get(ResourceCommitmentDB, "R1")
    assert commitment.fire_event_id == event_b  # untouched - still owned by the winning cycle
    session.close()


def test_rejects_activation_when_a_resource_went_unavailable(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    session = sqlite_session_factory()
    db_resource = session.get(FirefightingResourceDB, "R1")
    db_resource.status = ResourceStatus.UNAVAILABLE
    session.commit()
    session.close()

    with pytest.raises(GlobalPlanningStaleInput):
        _make_service(sqlite_session_factory).activate(
            global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
        )


# ---------------------------------------------------------------------------
# Atomic failure / rollback (Task 51)
# ---------------------------------------------------------------------------


def test_failure_mid_activation_leaves_nothing_written(sqlite_session_factory):
    session = sqlite_session_factory()
    event_a = _persist_fire_event(session, 32.70, 35.00)
    event_b = _persist_fire_event(session, 32.90, 35.20)
    target_set_a, target_a = _persist_target_set(session, event_a, 32.70, 35.00)
    target_set_b, target_b = _persist_target_set(session, event_b, 32.90, 35.20)
    _persist_resource(session, "R1")
    _persist_resource(session, "R2")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (event_a, event_b))

    demand_a = make_demand(event_a, minimum_resources=1, desired_resources=1)
    demand_b = make_demand(event_b, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(event_a, event_b),
        targets=(make_target(event_a, target_a), make_target(event_b, target_b)),
        resources=(make_resource("R1"), make_resource("R2")),
        routes=(
            make_route("R1", event_a, target_a, eta_seconds=10.0),
            make_route("R2", event_b, target_b, eta_seconds=10.0),
        ),
        event_target_set_ids={event_a: target_set_a, event_b: target_set_b},
        incident_demands=(demand_a, demand_b),
    )
    result = _optimize(global_input)

    service = _make_service(sqlite_session_factory)

    class _ExplodingRoutePlanningRepository(RoutePlanningRepository):
        def save_run_in_session(self, session, run):
            if run.fire_event_id == event_b:
                raise RuntimeError("simulated failure persisting the second event")
            return super().save_run_in_session(session, run)

    service._route_planning_repository = _ExplodingRoutePlanningRepository(sqlite_session_factory)

    with pytest.raises(RuntimeError):
        service.activate(global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF)

    session = sqlite_session_factory()
    assert session.execute(select(ResponsePlanDB)).scalars().all() == []
    assert session.execute(select(RoutePlanningRunDB)).scalars().all() == []
    assert session.execute(select(ResourceCommitmentDB)).scalars().all() == []
    session.close()


# ---------------------------------------------------------------------------
# Zero-resource event (Task 22/43)
# ---------------------------------------------------------------------------


def test_zero_action_event_still_gets_a_truthful_current_plan(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(),  # no resources at all - genuinely nothing to assign
        routes=(),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)
    assert result.actions == ()

    activation = _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
    )

    plan_id = activation.response_plan_ids_by_event[fire_event_id]
    session = sqlite_session_factory()
    db_plan = session.get(ResponsePlanDB, plan_id)
    assert db_plan.status == "no_feasible_assignments"
    uncovered = session.execute(
        select(ResponseTargetDB.id).where(ResponseTargetDB.id == target_id)
    ).scalar_one()
    assert uncovered == target_id
    session.close()


# ---------------------------------------------------------------------------
# Final-closure atomicity fix (audit item 2): GlobalPlanningRunEvent member
# results, GlobalPlanningRun optimization metadata, and its final status are
# written by activate() itself, in the SAME transaction as the ResponsePlan/
# ResourceCommitment writes.
# ---------------------------------------------------------------------------


def _make_run_history() -> GlobalRunHistoryContext:
    return GlobalRunHistoryContext(
        combined_fingerprint="c" * 64,
        optimization_policy_fingerprint="p" * 64,
        demand_scoring_policy_methodology="test-demand-scoring",
        demand_scoring_policy_version="1.0",
        severity_demand_policy_methodology="test-severity-demand",
        severity_demand_policy_version="1.0",
        stability_policy_methodology="test-stability",
        stability_policy_version="1.0",
    )


def test_activate_with_run_history_persists_member_results_and_run_status_atomically(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    activation = _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF,
        run_history=_make_run_history(),
    )
    plan_id = activation.response_plan_ids_by_event[fire_event_id]

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    stored_run = run_repository.get_by_id(run_id)
    assert stored_run.run.status is GlobalPlanningRunStatus.COMPLETED
    assert stored_run.run.fitness_score is not None
    assert stored_run.run.shortage_total_required == 1
    (member,) = run_repository.get_members(run_id)
    assert member.member.response_plan_id == plan_id
    assert member.member.minimum_resources == 1
    assert member.member.assigned_resources == 1


def test_activate_without_run_history_never_touches_global_planning_run_bookkeeping(sqlite_session_factory):
    """Backward compatibility: existing direct callers of activate() that
    do not pass run_history (unit tests exercising activation in isolation)
    are unaffected - the run stays exactly RUNNING, untouched."""
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    _make_service(sqlite_session_factory).activate(
        global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF
    )

    stored_run = GlobalPlanningRunRepository(sqlite_session_factory).get_by_id(run_id)
    assert stored_run.run.status is GlobalPlanningRunStatus.RUNNING
    assert stored_run.run.fitness_score is None


class _PoisonedGlobalPlanningRunRepository(GlobalPlanningRunRepository):
    """Fails partway through the run-history write - after the member
    result for the ONE event in this test has already been staged in the
    session, but before the run is marked COMPLETED - to prove the whole
    activation transaction (ResponsePlan/ResourceCommitment included) rolls
    back together with it rather than leaving a "partially historied"
    authoritative generation."""

    def record_global_optimization_metadata_in_session(self, *args, **kwargs):
        raise RuntimeError("simulated crash between plan/commitment activation and history persistence")


def test_failure_injected_between_activation_and_run_history_rolls_back_everything(sqlite_session_factory):
    """Failure-injection test (audit item 2): a fault injected exactly at
    the former crash-window boundary - after ResponsePlan/ResourceCommitment
    writes, mid-way through writing GlobalPlanningRunEvent/GlobalPlanningRun
    history - must roll back the ENTIRE transaction. No authoritative
    ResponsePlan/ResourceCommitment may exist while required history is
    missing: with the fix, the only two possible outcomes are "both
    committed" or "neither committed", never a mix."""
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    target_set_id, target_id = _persist_target_set(session, fire_event_id)
    _persist_resource(session, "R1")
    session.commit()
    session.close()
    run_id = _create_global_run(sqlite_session_factory, (fire_event_id,))

    demand = make_demand(fire_event_id, minimum_resources=1, desired_resources=1)
    global_input = make_input(
        global_planning_run_id=run_id,
        active_fire_event_ids=(fire_event_id,),
        targets=(make_target(fire_event_id, target_id),),
        resources=(make_resource("R1"),),
        routes=(make_route("R1", fire_event_id, target_id, eta_seconds=10.0),),
        event_target_set_ids={fire_event_id: target_set_id},
        incident_demands=(demand,),
    )
    result = _optimize(global_input)

    poisoned_repository = _PoisonedGlobalPlanningRunRepository(sqlite_session_factory)
    service = GlobalResponsePlanActivationService(
        response_plan_repository=ResponsePlanRepository(sqlite_session_factory),
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(sqlite_session_factory),
        route_planning_repository=RoutePlanningRepository(sqlite_session_factory),
        resource_commitment_repository=ResourceCommitmentRepository(sqlite_session_factory),
        global_planning_run_repository=poisoned_repository,
        session_factory=sqlite_session_factory,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        service.activate(
            global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF,
            run_history=_make_run_history(),
        )

    session = sqlite_session_factory()
    assert session.execute(select(ResponsePlanDB)).scalars().all() == []
    assert session.execute(select(ResourceCommitmentDB)).scalars().all() == []
    assert session.execute(select(RoutePlanningRunDB)).scalars().all() == []
    db_run = session.get(GlobalPlanningRunDB, run_id)
    assert db_run.status == GlobalPlanningRunStatus.RUNNING.value  # never advanced to COMPLETED
    session.close()

    run_repository = GlobalPlanningRunRepository(sqlite_session_factory)
    (member,) = run_repository.get_members(run_id)
    assert member.member.result_status is None  # the member write was rolled back too, not left half-applied
    assert member.member.response_plan_id is None
