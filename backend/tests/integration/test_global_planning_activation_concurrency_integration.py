"""Live concurrency integration tests for GlobalResponsePlanActivationService
(Stage 6 of the Global Multi-Incident Optimizer refactor: Blocker D - real
concurrent-session Neon tests).

Mirrors test_resource_commitment_concurrency_integration.py's proven
approach (Stage 1) one level up: real OS threads plus a barrier force two
INDEPENDENTLY-BUILT "competing global cycles" - each a real, self-consistent
GlobalPlanningInput/GlobalOptimizationResult pair claiming the SAME scarce
resource for a DIFFERENT FireEvent - to race for real, inside real
overlapping DB transactions against Neon/PostgreSQL, exercising the exact
row-locking (SELECT ... FOR UPDATE on FireEvent + resource rows) and
stale-input revalidation that is the actual concurrency safety net
GlobalPlanningRefreshCoordinator relies on (see
GlobalResponsePlanActivationService.activate()'s docstring). A sequential
call pattern cannot exercise this for real, only its logic in isolation.

Why two DIRECTLY-CONSTRUCTED conflicting results rather than two real
GlobalResponseOptimizationService.optimize() calls racing end-to-end: the
Global GA is a deterministic pure function of its input - two concurrent
optimize() calls over the IDENTICAL pre-activation world state converge to
the IDENTICAL assignment regardless of thread scheduling, so an "organic"
GA-level race is not reproducible. Constructing two independently-decided,
individually-valid GlobalOptimizationResult objects (exactly mirroring
test_resource_commitment_concurrency_integration.py's own precedent of
directly building two conflicting ResponsePlan objects, not racing two
optimizer runs) is what actually exercises the activation-time concurrency
guarantee this blocker cares about.

No FIRMS/IMS/Copernicus/agent access is required - only this project's own
Neon/PostgreSQL database. No DB mutation survives past this test's cleanup.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.config.settings import settings
from src.database.connection import get_engine, get_session_factory, init_db
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.graph_node_db import GraphNodeDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_event_resource_demand_result import GlobalEventResourceDemandResult
from src.models.global_optimization_result import GlobalOptimizationResult
from src.models.fire_event_status import FireEventStatus
from src.models.global_response_action import GlobalResponseAction
from src.models.global_resource_shortage import GlobalResourceShortage
from src.models.response_target_type import ResponseTargetType
from src.models.resource_status import ResourceStatus
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.models import GraphEdge, GraphNode
from src.models.dispatch_state import DispatchState
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.fire_event_lifecycle.fire_event_lifecycle_service import FireEventLifecycleService
from src.services.global_planning.global_planning_refresh_coordinator import GlobalPlanningRefreshStatus
from src.services.global_planning.global_planning_refresh_production_factory import (
    build_global_planning_refresh_coordinator,
)
from src.services.global_planning.global_response_plan_activation_service import (
    GlobalPlanningStaleInput,
    GlobalResponsePlanActivationService,
)
from src.services.operational_refresh.resource_status_update_service import ResourceStatusUpdateService
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver
from tests.calculators.global_response_optimization.helpers import make_demand, make_input, make_resource, make_target

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage6-global-activation-concurrency-integration"
TRIGGER_PREFIX = "stage6-cc-"
AS_OF = datetime(2026, 9, 24, 7, 0, tzinfo=timezone.utc)
STATION_ID = "STAGE6-CC-STATION"
SOURCE_NODE_ID = 910060001
TARGET_NODE_ID = 910060002

# Additional fixture identifiers for the mid-run new-fire concurrency test
# (two stations, four nodes - distinct from the ids above to avoid any
# cross-test collision).
STATION_A_ID_NF = "STAGE6-CC-STATION-A-NF"
STATION_B_ID_NF = "STAGE6-CC-STATION-B-NF"
NODE_STATION_A_NF = 910060011
NODE_TARGET_A_NF = 910060012
NODE_STATION_B_NF = 910060013
NODE_TARGET_B_NF = 910060014


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url):
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM response_plan_planning_states WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_actions WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_plan_uncovered_targets WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM resource_commitments WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        # A real coordinator refresh() (used by the mid-run new-fire test)
        # plans for the FULL active FireEvent set, which on this shared Neon
        # dev DB may include pre-existing FireEvents this test never
        # created (not tagged with METHODOLOGY_VERSION). Any ResponsePlan
        # created for such a stray event still belongs to THIS test's own
        # GlobalPlanningRun - scope this cleanup by global_planning_run_id,
        # not just fire_event_id, so it is fully removed without touching
        # that stray event's own unrelated history.
        connection.execute(
            text(
                "DELETE FROM response_plan_planning_states WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix))"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM response_actions WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix))"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM response_plan_uncovered_targets WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix))"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM global_planning_run_events WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix)"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM resource_commitments WHERE response_plan_id IN ("
                "SELECT id FROM response_plans WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix))"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM response_plans WHERE global_planning_run_id IN ("
                "SELECT id FROM global_planning_runs WHERE trigger LIKE :prefix)"
            ),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM global_planning_runs WHERE trigger LIKE :prefix"),
            {"prefix": f"{TRIGGER_PREFIX}%"},
        )
        connection.execute(
            text(
                "DELETE FROM route_results WHERE route_planning_run_id IN ("
                "SELECT id FROM route_planning_runs WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                "))"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM route_planning_runs WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_targets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text(
                "DELETE FROM response_target_sets WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology_version = :methodology_version"),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        connection.execute(
            text("DELETE FROM firefighting_resources WHERE station_id = :station_id"),
            {"station_id": STATION_ID},
        )
        connection.execute(text("DELETE FROM fire_stations WHERE id = :station_id"), {"station_id": STATION_ID})
        connection.execute(
            text("DELETE FROM graph_nodes WHERE id IN (:source_id, :target_id)"),
            {"source_id": SOURCE_NODE_ID, "target_id": TARGET_NODE_ID},
        )
        connection.execute(
            text("DELETE FROM firefighting_resources WHERE station_id IN (:a, :b)"),
            {"a": STATION_A_ID_NF, "b": STATION_B_ID_NF},
        )
        connection.execute(
            text("DELETE FROM fire_stations WHERE id IN (:a, :b)"),
            {"a": STATION_A_ID_NF, "b": STATION_B_ID_NF},
        )
        connection.execute(
            text(
                "DELETE FROM graph_edges WHERE source_node_id IN (:a1, :a2, :b1, :b2) "
                "OR target_node_id IN (:a1, :a2, :b1, :b2)"
            ),
            {
                "a1": NODE_STATION_A_NF, "a2": NODE_TARGET_A_NF,
                "b1": NODE_STATION_B_NF, "b2": NODE_TARGET_B_NF,
            },
        )
        connection.execute(
            text("DELETE FROM graph_nodes WHERE id IN (:a1, :a2, :b1, :b2)"),
            {
                "a1": NODE_STATION_A_NF, "a2": NODE_TARGET_A_NF,
                "b1": NODE_STATION_B_NF, "b2": NODE_TARGET_B_NF,
            },
        )


def _persist_fire_event(session, latitude: float, longitude: float) -> int:
    event = FireEventDB(
        latitude=latitude, longitude=longitude, detected_at=AS_OF, updated_at=AS_OF, status="confirmed",
        detection_confidence=0.9, methodology="STAGE6_CC_DETECTION", methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_target_set(session, fire_event_id: int, latitude: float, longitude: float) -> tuple[int, int]:
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


def _create_global_run(fire_event_ids: tuple[int, ...], trigger: str) -> int:
    stored_run = GlobalPlanningRunRepository().create_run(
        started_at=AS_OF, trigger=trigger, methodology="global_genetic_resource_allocation",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=fire_event_ids,
    )
    return stored_run.id


def _make_service() -> GlobalResponsePlanActivationService:
    return GlobalResponsePlanActivationService(
        response_plan_repository=ResponsePlanRepository(),
        response_plan_planning_state_repository=ResponsePlanPlanningStateRepository(),
        route_planning_repository=RoutePlanningRepository(),
        resource_commitment_repository=ResourceCommitmentRepository(),
    )


def _build_competing_result(
    *, global_planning_run_id: int, event_a: int, event_b: int, target_a: int, target_b: int,
    winner_event_id: int, cycle_tag: str,
) -> GlobalOptimizationResult:
    """One independently-decided candidate generation: assigns the sole
    resource R1 to `winner_event_id`, leaves the other event uncovered -
    both events are still part of the input's active set (this cycle
    genuinely considered both), it simply decided differently about where
    the one scarce resource should go."""
    loser_event_id = event_b if winner_event_id == event_a else event_a
    loser_target_id = target_b if winner_event_id == event_a else target_a
    winner_target_id = target_a if winner_event_id == event_a else target_b

    action = GlobalResponseAction(
        resource_id="STAGE6-CC-R1", station_id=STATION_ID, fire_event_id=winner_event_id,
        response_target_id=winner_target_id, target_type=ResponseTargetType.ACTIVE_FIRE,
        target_priority=100.0, eta_seconds=30.0, route_distance_meters=200.0, node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
    )
    winner_demand = GlobalEventResourceDemandResult(
        fire_event_id=winner_event_id, minimum_resources=1, desired_resources=1,
        suppression_resources_assigned=1, required_slots_covered=1, required_slots_uncovered=0,
        desired_slots_covered=0, desired_slots_uncovered=0, predicted_risk_slots_covered=0,
    )
    loser_demand = GlobalEventResourceDemandResult(
        fire_event_id=loser_event_id, minimum_resources=1, desired_resources=1,
        suppression_resources_assigned=0, required_slots_covered=0, required_slots_uncovered=1,
        desired_slots_covered=0, desired_slots_uncovered=0, predicted_risk_slots_covered=0,
    )
    event_results = (
        GlobalEventOptimizationResult(
            fire_event_id=winner_event_id, actions=(action,), covered_slot_ids=(f"{winner_target_id}#0",),
            uncovered_slot_ids=(), coverage_score=100.0, average_eta_seconds=30.0, demand_result=winner_demand,
        ),
        GlobalEventOptimizationResult(
            fire_event_id=loser_event_id, actions=(), covered_slot_ids=(),
            uncovered_slot_ids=(f"{loser_target_id}#0",), coverage_score=0.0, average_eta_seconds=None,
            demand_result=loser_demand,
        ),
    )
    shortage = GlobalResourceShortage(
        total_required=2, total_desired=2, total_assigned=1, unmet_required=1, unmet_desired=1,
        candidate_assignable_resource_count=1, committed_resource_count=0, unavailable_resource_count=0,
    )
    return GlobalOptimizationResult(
        global_planning_run_id=global_planning_run_id, input_fingerprint="f" * 64,
        optimization_methodology="global_genetic_resource_allocation", optimization_methodology_version="1.0",
        random_seed=42, fitness_score=10.0, coverage_score=50.0, average_eta_seconds=30.0,
        actions=(action,), uncovered_slot_ids=(f"{loser_target_id}#0",), event_results=event_results,
        shortage=shortage, config=GlobalResponseOptimizationConfig(),
    )


def _run_concurrently(fn_a, fn_b):
    """Release both callables at (as close to) the same instant via a
    barrier, each on its own thread, and return their (result, exception)
    outcomes."""
    barrier = threading.Barrier(2)
    outcomes = {}

    def _call(name, fn):
        barrier.wait()
        try:
            outcomes[name] = ("ok", fn())
        except Exception as exc:  # noqa: BLE001 - capturing for assertion, not swallowing
            outcomes[name] = ("error", exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_call, "a", fn_a)
        future_b = executor.submit(_call, "b", fn_b)
        future_a.result(timeout=120)
        future_b.result(timeout=120)
    return outcomes


def test_two_competing_global_cycles_cannot_both_commit_the_shared_resource_neon():
    """Blocker D's central question: "Can concurrent global cycles leave
    mixed current generations?" - Answer: NO. Exactly one of the two
    competing cycles wins the shared resource; the other is rejected as
    stale (never partially written); the ResourceCommitment PK invariant
    holds; no resource ends up referenced by two FireEvents."""
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 6 CC Station", latitude=31.6, longitude=34.6)
    session.add(station)
    session.add(FirefightingResourceDB(id="STAGE6-CC-R1", station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    for node_id, lat, lon in ((SOURCE_NODE_ID, 31.6, 34.6), (TARGET_NODE_ID, 31.6, 34.6)):
        session.add(GraphNodeDB(id=node_id, latitude=lat, longitude=lon))
    event_a = _persist_fire_event(session, 31.6, 34.6)
    event_b = _persist_fire_event(session, 31.61, 34.61)
    session.commit()
    target_set_a, target_a = _persist_target_set(session, event_a, 31.6, 34.6)
    target_set_b, target_b = _persist_target_set(session, event_b, 31.61, 34.61)
    session.commit()
    session.close()

    # This is a live, shared Neon dev database - other confirmed FireEvents
    # may already exist from unrelated work. _validate_not_stale requires
    # active_fire_event_ids to cover the FULL currently-active set (never a
    # subset), exactly like a real GlobalPlanningRefreshCoordinator.refresh()
    # would build it - so the input honestly reflects "every active FireEvent
    # this cycle considered," not just the two this test cares about.
    full_active_ids = FireEventRepository().get_active_fire_event_ids()
    assert event_a in full_active_ids and event_b in full_active_ids
    stray_ids = tuple(fid for fid in full_active_ids if fid not in (event_a, event_b))

    run_1_id = _create_global_run(full_active_ids, trigger=f"{TRIGGER_PREFIX}cycle-1")
    run_2_id = _create_global_run(full_active_ids, trigger=f"{TRIGGER_PREFIX}cycle-2")

    shared_kwargs = dict(
        active_fire_event_ids=full_active_ids,
        targets=(make_target(event_a, target_a), make_target(event_b, target_b)),
        resources=(make_resource("STAGE6-CC-R1", station_id=STATION_ID),),
        event_target_set_ids={event_a: target_set_a, event_b: target_set_b},
        incident_demands=(
            make_demand(event_a, minimum_resources=1, desired_resources=1),
            make_demand(event_b, minimum_resources=1, desired_resources=1),
        )
        + tuple(make_demand(stray_id) for stray_id in stray_ids),
    )
    input_1 = make_input(global_planning_run_id=run_1_id, routes=(), **shared_kwargs)
    input_2 = make_input(global_planning_run_id=run_2_id, routes=(), **shared_kwargs)
    result_1 = _build_competing_result(
        global_planning_run_id=run_1_id, event_a=event_a, event_b=event_b, target_a=target_a, target_b=target_b,
        winner_event_id=event_a, cycle_tag="cycle-1",
    )
    result_2 = _build_competing_result(
        global_planning_run_id=run_2_id, event_a=event_a, event_b=event_b, target_a=target_a, target_b=target_b,
        winner_event_id=event_b, cycle_tag="cycle-2",
    )

    service = _make_service()
    outcomes = _run_concurrently(
        lambda: service.activate(global_planning_input=input_1, global_optimization_result=result_1, as_of=AS_OF),
        lambda: service.activate(global_planning_input=input_2, global_optimization_result=result_2, as_of=AS_OF),
    )

    statuses = {name: kind for name, (kind, _) in outcomes.items()}
    assert sorted(statuses.values()) == ["error", "ok"], (
        f"expected exactly one winning cycle and one rejected (stale) cycle, got {outcomes}"
    )
    (loser_name,) = [name for name, kind in statuses.items() if kind == "error"]
    loser_exception = outcomes[loser_name][1]
    assert isinstance(loser_exception, (GlobalPlanningStaleInput, ResourceCommitmentConflict)), (
        f"losing cycle must fail cleanly (stale-input or PK conflict), never a mystery exception; got {loser_exception!r}"
    )

    # -- ResourceCommitment PK invariant: exactly one row for the shared resource. --
    engine = get_engine()
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM resource_commitments WHERE resource_id = 'STAGE6-CC-R1'")
        ).scalar()
    assert count == 1

    resource_commitment_repository = ResourceCommitmentRepository()
    committed = resource_commitment_repository.get_by_resource_id("STAGE6-CC-R1")
    winner_event_id = event_a if statuses["a"] == "ok" else event_b
    assert committed.fire_event_id == winner_event_id  # never split, never assigned to the loser's event

    # -- No resource appears committed to two FireEvents; exactly one
    # coherent current generation per event, never a mix of both cycles'
    # decisions. --
    winner_name = "a" if statuses["a"] == "ok" else "b"
    winner_activation = outcomes[winner_name][1]
    assert winner_activation.response_plan_ids_by_event[winner_event_id] > 0

    resolver = CurrentResponsePlanResolver(ResponsePlanRepository(), ResponsePlanPlanningStateRepository())
    current_winner_plan = resolver.resolve(fire_event_id=winner_event_id)
    assert current_winner_plan is not None
    assert current_winner_plan.id == winner_activation.response_plan_ids_by_event[winner_event_id]
    winner_resource_ids = {action.resource_id for action in current_winner_plan.plan.actions}
    assert winner_resource_ids == {"STAGE6-CC-R1"}

    other_event_id = event_b if winner_event_id == event_a else event_a
    current_other_plan = resolver.resolve(fire_event_id=other_event_id)
    if current_other_plan is not None:
        assert "STAGE6-CC-R1" not in {action.resource_id for action in current_other_plan.plan.actions}


def test_activation_racing_a_concurrent_resolve_never_leaves_a_dangling_commitment_neon():
    """"Resolve-during-run" (Blocker D, Task 15/19): a global activation for
    FireEvent A races against FireEventLifecycleService.resolve_event(A) on
    a separate thread - both genuinely contend for the SAME FireEventDB row
    lock (activate()'s _lock_fire_events and resolve_event's own
    with_for_update select), so whichever wins is real, not simulated.

    Whichever order wins, the invariant Task 10 requires must hold: an
    inactive (RESOLVED) FireEvent owns zero ResourceCommitments -
    - resolve wins first: activation then sees the FireEvent is no longer
      active and is rejected as stale (GlobalPlanningStaleInput) - nothing
      is written.
    - activation wins first: it commits R1 -> A, then resolve_event (Task
      1.1's own atomic transition) releases every commitment A owns as
      part of the SAME resolve transaction - R1 ends up free again.
    """
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 6 CC Station", latitude=31.6, longitude=34.6)
    session.add(station)
    session.add(FirefightingResourceDB(id="STAGE6-CC-R1", station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    for node_id, lat, lon in ((SOURCE_NODE_ID, 31.6, 34.6), (TARGET_NODE_ID, 31.6, 34.6)):
        session.add(GraphNodeDB(id=node_id, latitude=lat, longitude=lon))
    event_a = _persist_fire_event(session, 31.6, 34.6)
    session.commit()
    target_set_a, target_a = _persist_target_set(session, event_a, 31.6, 34.6)
    session.commit()
    session.close()

    full_active_ids = FireEventRepository().get_active_fire_event_ids()
    assert event_a in full_active_ids
    stray_ids = tuple(fid for fid in full_active_ids if fid != event_a)
    run_id = _create_global_run(full_active_ids, trigger=f"{TRIGGER_PREFIX}resolve-race")

    action = GlobalResponseAction(
        resource_id="STAGE6-CC-R1", station_id=STATION_ID, fire_event_id=event_a, response_target_id=target_a,
        target_type=ResponseTargetType.ACTIVE_FIRE, target_priority=100.0, eta_seconds=30.0,
        route_distance_meters=200.0, node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
    )
    demand = GlobalEventResourceDemandResult(
        fire_event_id=event_a, minimum_resources=1, desired_resources=1, suppression_resources_assigned=1,
        required_slots_covered=1, required_slots_uncovered=0, desired_slots_covered=0, desired_slots_uncovered=0,
        predicted_risk_slots_covered=0,
    )
    event_results = (
        GlobalEventOptimizationResult(
            fire_event_id=event_a, actions=(action,), covered_slot_ids=(f"{target_a}#0",), uncovered_slot_ids=(),
            coverage_score=100.0, average_eta_seconds=30.0, demand_result=demand,
        ),
    )
    shortage = GlobalResourceShortage(
        total_required=1, total_desired=1, total_assigned=1, unmet_required=0, unmet_desired=0,
        candidate_assignable_resource_count=1, committed_resource_count=0, unavailable_resource_count=0,
    )
    result = GlobalOptimizationResult(
        global_planning_run_id=run_id, input_fingerprint="f" * 64,
        optimization_methodology="global_genetic_resource_allocation", optimization_methodology_version="1.0",
        random_seed=42, fitness_score=10.0, coverage_score=100.0, average_eta_seconds=30.0, actions=(action,),
        uncovered_slot_ids=(), event_results=event_results, shortage=shortage,
        config=GlobalResponseOptimizationConfig(),
    )
    global_input = make_input(
        global_planning_run_id=run_id, active_fire_event_ids=full_active_ids,
        targets=(make_target(event_a, target_a),), resources=(make_resource("STAGE6-CC-R1", station_id=STATION_ID),),
        routes=(), event_target_set_ids={event_a: target_set_a},
        incident_demands=(make_demand(event_a, minimum_resources=1, desired_resources=1),)
        + tuple(make_demand(stray_id) for stray_id in stray_ids),
    )

    service = _make_service()
    lifecycle_service = FireEventLifecycleService()
    outcomes = _run_concurrently(
        lambda: service.activate(global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF),
        lambda: lifecycle_service.resolve_event(event_a, as_of=AS_OF),
    )

    statuses = {name: kind for name, (kind, _) in outcomes.items()}
    assert statuses["b"] == "ok", f"resolve_event must never fail in this scenario, got {outcomes['b']}"
    if statuses["a"] == "error":
        assert isinstance(outcomes["a"][1], GlobalPlanningStaleInput), (
            f"a rejected activation must fail cleanly as stale, got {outcomes['a'][1]!r}"
        )

    stored_event = FireEventRepository().get_by_id(event_a)
    assert stored_event.event.status is FireEventStatus.RESOLVED

    engine = get_engine()
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM resource_commitments WHERE fire_event_id = :event_a"),
            {"event_a": event_a},
        ).scalar()
    assert count == 0, "an inactive (RESOLVED) FireEvent must own zero commitments, regardless of race order"


def test_activation_racing_a_concurrent_resource_status_change_never_commits_an_unavailable_resource_neon():
    """"Resource-change-during-run" (Blocker D, Task 15/17): a global
    activation claiming R1 for FireEvent A races against
    ResourceStatusUpdateService marking R1 UNAVAILABLE on a separate
    thread. Postgres row-locking still serializes them for real: activate()
    takes SELECT ... FOR UPDATE on R1's row, and any plain UPDATE from the
    status-change transaction blocks on that same row until the lock is
    released - so this genuinely races, even though the status-change path
    itself issues no explicit FOR UPDATE.

    Whichever order wins, the invariant is: R1 never ends up committed to A
    while ALSO having been UNAVAILABLE at the moment the winning decision
    was actually made -
    - the status change wins first: activation's own stale check sees R1
      UNAVAILABLE and is rejected (GlobalPlanningStaleInput) - no commitment
      is ever written for a resource this cycle should never have used.
    - activation wins first: it legitimately committed R1 to A before the
      status change happened - a resource going UNAVAILABLE immediately
      after being dispatched is a real, later, independent operational fact,
      not a corrupted read.
    """
    session = get_session_factory()()
    station = FireStationDB(id=STATION_ID, name="Stage 6 CC Station", latitude=31.6, longitude=34.6)
    session.add(station)
    session.add(FirefightingResourceDB(id="STAGE6-CC-R1", station_id=STATION_ID, status=ResourceStatus.AVAILABLE))
    for node_id, lat, lon in ((SOURCE_NODE_ID, 31.6, 34.6), (TARGET_NODE_ID, 31.6, 34.6)):
        session.add(GraphNodeDB(id=node_id, latitude=lat, longitude=lon))
    event_a = _persist_fire_event(session, 31.6, 34.6)
    session.commit()
    target_set_a, target_a = _persist_target_set(session, event_a, 31.6, 34.6)
    session.commit()
    session.close()

    full_active_ids = FireEventRepository().get_active_fire_event_ids()
    assert event_a in full_active_ids
    stray_ids = tuple(fid for fid in full_active_ids if fid != event_a)
    run_id = _create_global_run(full_active_ids, trigger=f"{TRIGGER_PREFIX}resource-race")

    action = GlobalResponseAction(
        resource_id="STAGE6-CC-R1", station_id=STATION_ID, fire_event_id=event_a, response_target_id=target_a,
        target_type=ResponseTargetType.ACTIVE_FIRE, target_priority=100.0, eta_seconds=30.0,
        route_distance_meters=200.0, node_path=(SOURCE_NODE_ID, TARGET_NODE_ID),
    )
    demand = GlobalEventResourceDemandResult(
        fire_event_id=event_a, minimum_resources=1, desired_resources=1, suppression_resources_assigned=1,
        required_slots_covered=1, required_slots_uncovered=0, desired_slots_covered=0, desired_slots_uncovered=0,
        predicted_risk_slots_covered=0,
    )
    event_results = (
        GlobalEventOptimizationResult(
            fire_event_id=event_a, actions=(action,), covered_slot_ids=(f"{target_a}#0",), uncovered_slot_ids=(),
            coverage_score=100.0, average_eta_seconds=30.0, demand_result=demand,
        ),
    )
    shortage = GlobalResourceShortage(
        total_required=1, total_desired=1, total_assigned=1, unmet_required=0, unmet_desired=0,
        candidate_assignable_resource_count=1, committed_resource_count=0, unavailable_resource_count=0,
    )
    result = GlobalOptimizationResult(
        global_planning_run_id=run_id, input_fingerprint="f" * 64,
        optimization_methodology="global_genetic_resource_allocation", optimization_methodology_version="1.0",
        random_seed=42, fitness_score=10.0, coverage_score=100.0, average_eta_seconds=30.0, actions=(action,),
        uncovered_slot_ids=(), event_results=event_results, shortage=shortage,
        config=GlobalResponseOptimizationConfig(),
    )
    global_input = make_input(
        global_planning_run_id=run_id, active_fire_event_ids=full_active_ids,
        targets=(make_target(event_a, target_a),), resources=(make_resource("STAGE6-CC-R1", station_id=STATION_ID),),
        routes=(), event_target_set_ids={event_a: target_set_a},
        incident_demands=(make_demand(event_a, minimum_resources=1, desired_resources=1),)
        + tuple(make_demand(stray_id) for stray_id in stray_ids),
    )

    service = _make_service()
    status_service = ResourceStatusUpdateService(FirefightingResourceRepository())
    outcomes = _run_concurrently(
        lambda: service.activate(global_planning_input=global_input, global_optimization_result=result, as_of=AS_OF),
        lambda: status_service.update_status(resource_id="STAGE6-CC-R1", new_status=ResourceStatus.UNAVAILABLE),
    )

    statuses = {name: kind for name, (kind, _) in outcomes.items()}
    assert statuses["b"] == "ok"
    status_result = outcomes["b"][1]
    assert status_result.status.name in ("UPDATED", "NO_OP")

    final_resource = FirefightingResourceRepository().get_by_id("STAGE6-CC-R1")
    assert final_resource.status is ResourceStatus.UNAVAILABLE

    engine = get_engine()
    with engine.connect() as connection:
        count = connection.execute(
            text("SELECT COUNT(*) FROM resource_commitments WHERE resource_id = 'STAGE6-CC-R1'")
        ).scalar()

    if statuses["a"] == "ok":
        # Activation legitimately won the race before UNAVAILABLE landed -
        # the resource is committed to A even though it is now UNAVAILABLE,
        # which is a real, later, independent operational fact.
        assert count == 1
        committed = ResourceCommitmentRepository().get_by_resource_id("STAGE6-CC-R1")
        assert committed.fire_event_id == event_a
    else:
        assert isinstance(outcomes["a"][1], GlobalPlanningStaleInput), (
            f"a rejected activation must fail cleanly as stale, got {outcomes['a'][1]!r}"
        )
        assert count == 0, "an activation rejected for an UNAVAILABLE resource must never leave a commitment"


class _PauseAfterFirstOptimize:
    """Wraps the real GlobalResponseOptimizationService so the FIRST optimize()
    call - the one whose result will be handed to activate() for the
    {A}-only cycle - blocks (via threading.Event, not a fixed sleep) right
    after computing its real result, until a second, independent DB session
    has inserted and committed FireEvent B. This makes the race between
    "activate() locks FireEvent A" and "Fire B becomes active" deterministic
    instead of a timing gamble, while every subsequent optimize() call (the
    coordinator's own bounded retry) passes straight through unmodified."""

    def __init__(self, real_service, b_inserted_event: threading.Event, ready_event: threading.Event) -> None:
        self._real_service = real_service
        self._b_inserted_event = b_inserted_event
        self._ready_event = ready_event
        self.call_count = 0

    def optimize(self, *args, **kwargs):
        self.call_count += 1
        result = self._real_service.optimize(*args, **kwargs)
        if self.call_count == 1:
            self._ready_event.set()  # "GA for {A} is done - go insert B now"
            if not self._b_inserted_event.wait(timeout=60):
                raise TimeoutError("FireEvent B was not inserted within 60s - test setup problem, not a real result.")
        return result


def _persist_road_network_for_new_fire_scenario() -> None:
    session = get_session_factory()()
    RoadNetworkRepository().save_network(
        session,
        nodes=[
            GraphNode(id=NODE_STATION_A_NF, latitude=31.70, longitude=34.70),
            GraphNode(id=NODE_TARGET_A_NF, latitude=31.701, longitude=34.701),
            GraphNode(id=NODE_STATION_B_NF, latitude=31.90, longitude=34.90),
            GraphNode(id=NODE_TARGET_B_NF, latitude=31.901, longitude=34.901),
        ],
        edges=[
            GraphEdge(source_node_id=NODE_STATION_A_NF, target_node_id=NODE_TARGET_A_NF, distance_meters=200.0, travel_time_seconds=30.0),
            GraphEdge(source_node_id=NODE_STATION_B_NF, target_node_id=NODE_TARGET_B_NF, distance_meters=200.0, travel_time_seconds=30.0),
        ],
    )
    session.commit()
    session.close()


def test_mid_run_new_fire_forces_a_real_retry_that_preserves_as_dispatched_lock_and_covers_both_events_neon():
    """Audit item 1: the one true-concurrency scenario the existing three
    Blocker D tests do NOT cover - a brand-new FireEvent becoming active
    strictly BETWEEN one planning cycle's optimize() and its activate(),
    on a genuinely separate DB session/thread, driven through the REAL
    GlobalPlanningRefreshCoordinator.refresh() (not a hand-built
    GlobalOptimizationResult).

    Sequence, deterministic via threading.Event (not a Barrier - the two
    sides are not "release at the same instant", they are "B must land
    strictly between optimize() and activate()"):
      1. A separate, prior refresh() cycle dispatches R1 -> A (A already has
         a DISPATCHED resource before the cycle under test, per the audit's
         explicit requirement).
      2. Planner thread calls refresh() again. Its FIRST optimize() call
         computes a real result for active set {A} alone, then blocks.
      3. Main thread's helper inserts + commits FireEvent B (with a real
         target set) on its OWN session - genuinely independent of the
         planner's session/transaction - then signals the planner to resume.
      4. Planner's activate() call for the {A}-only result is rejected as
         stale (active set grew to include B) - GlobalPlanningStaleInput,
         caught by _run_one_cycle, which returns None.
      5. refresh()'s own bounded-retry loop (MAX_GLOBAL_STALE_RETRIES=1)
         re-fetches the active set (now {A, B}), builds a FRESH
         GlobalPlanningInput, runs ONE new GA optimization (call_count==2,
         passes straight through), and atomically activates both A and B.
      6. R1 must still be DISPATCHED to A - never yanked by the retry.
    """
    engine = get_engine()
    session = get_session_factory()()
    station_a = FireStationDB(id=STATION_A_ID_NF, name=STATION_A_ID_NF, latitude=31.70, longitude=34.70)
    station_b = FireStationDB(id=STATION_B_ID_NF, name=STATION_B_ID_NF, latitude=31.90, longitude=34.90)
    session.add(station_a)
    session.add(station_b)
    session.add(FirefightingResourceDB(id="STAGE6-CC-R1-NF", station_id=STATION_A_ID_NF, status=ResourceStatus.AVAILABLE))
    session.add(FirefightingResourceDB(id="STAGE6-CC-R2-NF", station_id=STATION_B_ID_NF, status=ResourceStatus.AVAILABLE))
    event_a = _persist_fire_event(session, 31.701, 34.701)
    session.commit()
    session.close()
    _persist_road_network_for_new_fire_scenario()
    session = get_session_factory()()
    _persist_target_set(session, event_a, 31.701, 34.701)
    session.commit()
    session.close()

    config = GlobalResponseOptimizationConfig(population_size=16, generation_count=12, random_seed=42)

    # -- Step 1: a normal, prior cycle dispatches R1 -> A. --
    setup_coordinator = build_global_planning_refresh_coordinator(config=config)
    initial = setup_coordinator.refresh(trigger=f"{TRIGGER_PREFIX}new-fire-setup", as_of=AS_OF)
    assert initial.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert event_a in initial.response_plan_ids_by_event
    with engine.connect() as connection:
        r1_fire_event_id = connection.execute(
            text("SELECT fire_event_id FROM resource_commitments WHERE resource_id = 'STAGE6-CC-R1-NF'")
        ).scalar()
    assert r1_fire_event_id == event_a

    # -- Step 2-5: the paused planner cycle, racing a genuinely independent insert of Fire B. --
    planner_coordinator = build_global_planning_refresh_coordinator(config=config)
    ready_event = threading.Event()
    b_inserted_event = threading.Event()
    planner_coordinator._optimization_service = _PauseAfterFirstOptimize(  # noqa: SLF001 - test-only seam injection
        planner_coordinator._optimization_service, b_inserted_event, ready_event  # noqa: SLF001
    )

    event_b_holder: dict[str, int] = {}

    def _insert_fire_b_on_a_separate_session() -> None:
        assert ready_event.wait(timeout=60), "planner never reached its first optimize() pause point"
        independent_session = get_session_factory()()  # genuinely separate SQLAlchemy session
        try:
            event_b_holder["id"] = _persist_fire_event(independent_session, 31.901, 34.901)
            _persist_target_set(independent_session, event_b_holder["id"], 31.901, 34.901)
            independent_session.commit()
        finally:
            independent_session.close()
        b_inserted_event.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        planner_future = executor.submit(
            planner_coordinator.refresh, trigger=f"{TRIGGER_PREFIX}new-fire-retry", as_of=AS_OF + timedelta(minutes=5)
        )
        inserter_future = executor.submit(_insert_fire_b_on_a_separate_session)
        inserter_future.result(timeout=90)
        result = planner_future.result(timeout=90)

    event_b = event_b_holder["id"]

    # -- Required behavior: retry happened, both events covered, R1's lock preserved. --
    assert result.status is GlobalPlanningRefreshStatus.ACTIVATED
    assert result.retry_count == 1, "exactly one bounded retry - the {A}-only cycle was rejected as stale, once"
    assert planner_coordinator._optimization_service.call_count == 2  # noqa: SLF001 - one per attempt
    assert set(result.response_plan_ids_by_event) == {event_a, event_b}

    with engine.connect() as connection:
        r1_fire_event_id_after = connection.execute(
            text("SELECT fire_event_id, dispatch_state FROM resource_commitments WHERE resource_id = 'STAGE6-CC-R1-NF'")
        ).one()
    assert r1_fire_event_id_after[0] == event_a  # never yanked by the retry
    assert r1_fire_event_id_after[1] == DispatchState.DISPATCHED.value

    resolver = CurrentResponsePlanResolver(ResponsePlanRepository(), ResponsePlanPlanningStateRepository())
    plan_b = resolver.resolve(fire_event_id=event_b)
    assert plan_b is not None  # Fire B, which arrived mid-run, was genuinely planned for in the retry

    # -- The rejected {A}-only run and its retry are BOTH visible in
    # GlobalPlanningRun history - the first as FAILED, never silently
    # dropped. --
    with engine.connect() as connection:
        statuses_for_trigger = connection.execute(
            text("SELECT status FROM global_planning_runs WHERE trigger = :trigger ORDER BY id"),
            {"trigger": f"{TRIGGER_PREFIX}new-fire-retry"},
        ).scalars().all()
    assert statuses_for_trigger == ["failed", "completed"], (
        f"expected [failed, completed] (stale rejection then successful retry), got {statuses_for_trigger}"
    )
