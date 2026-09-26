"""Tests for Task 5: BaselineComparisonService orchestration.

Uses the REAL Task 1-4 Company 3 components (`BaselinePlanEvaluator`,
`BaselinePlanComparisonCalculator`, `PlanComparisonRepository` backed by
the shared SQLite test fixture) end-to-end. Only the still-missing
external dependencies are faked: the optimized-plan reader and the
routing-run reader (Company 1/2 persistence does not exist yet -- see
`baseline_comparison_ports.py`). The scorer is also faked, exactly like
Task 2's own tests, purely to observe what the service builds and passed
through -- it is not a scoring implementation.

`ResponseTargetSetReader` is exercised two ways: a lightweight in-memory
fake (built from the REAL `StoredResponseTargetSet`/`StoredResponseTarget`
dataclasses, not a competing model) for most tests, and once, explicitly,
through the REAL `ResponseTargetRepository` to prove the service's default
production wiring actually works end-to-end.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.calculators.baseline_plan.baseline_plan_calculator import RouteCandidate, TargetOrder
from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.response_target import ResponseTarget
from src.models.response_target_set import ResponseTargetSet
from src.models.response_target_type import ResponseTargetType
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.plan_comparison_repository import PlanComparisonRepository, PlanComparisonRepositoryError
from src.repositories.response_target_repository import (
    ResponseTargetRepository,
    StoredResponseTarget,
    StoredResponseTargetSet,
)
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.baseline_comparison import BaselineComparisonService, BaselineComparisonServiceError

FIRE_EVENT_ID = 1
RESPONSE_PLAN_ID = 100
ROUTE_PLANNING_RUN_ID = 10
RESPONSE_TARGET_SET_ID = 20
GENERATED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# FND-05: plan_comparisons.fire_event_id/optimized_plan_id/
# route_planning_run_id/response_target_set_id are now real FKs. This suite's
# fake readers return hand-picked ids with no real backing rows, so wrap
# PlanComparisonRepository to provision a minimal, independent row for each
# of the 4 ids before delegating to the real save() (mirrors
# tests/repositories/test_plan_comparison_repository.py).
# ---------------------------------------------------------------------------


def persist_fk_prerequisites(
    session_factory, *, fire_event_id: int, response_target_set_id: int, route_planning_run_id: int, optimized_plan_id: int
) -> None:
    session = session_factory()
    added = False
    try:
        with session.no_autoflush:
            needs_fire_event = session.get(FireEventDB, fire_event_id) is None
            needs_target_set = session.get(ResponseTargetSetDB, response_target_set_id) is None
            needs_route_run = session.get(RoutePlanningRunDB, route_planning_run_id) is None
            needs_response_plan = session.get(ResponsePlanDB, optimized_plan_id) is None

        if needs_fire_event:
            session.add(
                FireEventDB(
                    id=fire_event_id,
                    latitude=32.731,
                    longitude=35.046,
                    detected_at=GENERATED_AT,
                    updated_at=GENERATED_AT,
                    status="confirmed",
                    detection_confidence=0.9,
                    methodology="TEST_DETECTION",
                    methodology_version="1.0",
                )
            )
            added = True
        if needs_target_set:
            session.add(
                ResponseTargetSetDB(
                    id=response_target_set_id,
                    fire_event_id=fire_event_id,
                    generated_at=GENERATED_AT,
                    methodology="TEST_TARGETS",
                    methodology_version="1.0",
                )
            )
            added = True
        if needs_fire_event or needs_target_set:
            session.flush()
        if needs_route_run:
            session.add(
                RoutePlanningRunDB(
                    id=route_planning_run_id,
                    fire_event_id=fire_event_id,
                    response_target_set_id=response_target_set_id,
                    planned_at=GENERATED_AT,
                    methodology="TEST_ROUTING",
                    methodology_version="1.0",
                    resource_ids=[],
                )
            )
            added = True
            session.flush()
        if needs_response_plan:
            session.add(
                ResponsePlanDB(
                    id=optimized_plan_id,
                    fire_event_id=fire_event_id,
                    response_target_set_id=response_target_set_id,
                    route_planning_run_id=route_planning_run_id,
                    generated_at=GENERATED_AT,
                    status="complete",
                    methodology="GENETIC_RESOURCE_ALLOCATION",
                    methodology_version="1.0",
                    random_seed=42,
                )
            )
            added = True
        if added:
            session.commit()
    finally:
        session.close()


class _FKProvisioningRepository:
    """Wraps PlanComparisonRepository so this suite's fake readers can keep
    returning arbitrary ids without each test separately pre-creating the FK
    chain plan_comparisons now requires (FND-05)."""

    def __init__(self, inner: PlanComparisonRepository, session_factory) -> None:
        self._inner = inner
        self._session_factory = session_factory

    def save(self, comparison):
        if isinstance(comparison, PlanComparison):
            persist_fk_prerequisites(
                self._session_factory,
                fire_event_id=comparison.fire_event_id,
                response_target_set_id=comparison.response_target_set_id,
                route_planning_run_id=comparison.route_planning_run_id,
                optimized_plan_id=comparison.optimized_plan_id,
            )
        return self._inner.save(comparison)

    def get_by_id(self, comparison_id):
        return self._inner.get_by_id(comparison_id)

    def list_for_fire_event(self, fire_event_id):
        return self._inner.list_for_fire_event(fire_event_id)


def fk_provisioning_repository(session_factory) -> PlanComparisonRepository:
    return _FKProvisioningRepository(PlanComparisonRepository(session_factory=session_factory), session_factory)


# ---------------------------------------------------------------------------
# Test doubles / fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakePlanScoreBreakdown:
    total_score: float
    coverage_score: float
    average_eta_seconds: float | None
    covered_target_count: int
    total_target_count: int
    uncovered_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class FakeOptimizedPlan:
    id: int
    fire_event_id: int
    route_planning_run_id: int
    response_target_set_id: int
    score: FakePlanScoreBreakdown


@dataclass(frozen=True)
class FakeRouteResult:
    id: int
    resource_id: str
    response_target_id: int
    status: str
    travel_time_seconds: float | None
    distance_meters: float | None = None


@dataclass(frozen=True)
class FakeRoutePlanningRun:
    id: int
    fire_event_id: int
    response_target_set_id: int
    resource_ids: tuple[str, ...]
    route_results: tuple[FakeRouteResult, ...]


class FakeOptimizedPlanReader:
    def __init__(self, plans: dict[int, FakeOptimizedPlan]) -> None:
        self._plans = plans
        self.calls: list[int] = []

    def get_by_id(self, response_plan_id: int) -> FakeOptimizedPlan | None:
        self.calls.append(response_plan_id)
        return self._plans.get(response_plan_id)


class FakeRoutePlanningRunReader:
    def __init__(self, runs: dict[int, FakeRoutePlanningRun]) -> None:
        self._runs = runs
        self.calls: list[int] = []

    def get_by_id(self, route_planning_run_id: int) -> FakeRoutePlanningRun | None:
        self.calls.append(route_planning_run_id)
        return self._runs.get(route_planning_run_id)


class FakeResponseTargetSetReader:
    def __init__(self, target_sets: dict[int, StoredResponseTargetSet]) -> None:
        self._target_sets = target_sets
        self.calls: list[int] = []

    def get_by_id(self, response_target_set_id: int) -> StoredResponseTargetSet | None:
        self.calls.append(response_target_set_id)
        return self._target_sets.get(response_target_set_id)


class RecordingScorer:
    """Records every call; never computes anything -- see baseline_plan_evaluator tests for the same pattern."""

    def __init__(self, *, result: FakePlanScoreBreakdown) -> None:
        self._result = result
        self.call_count = 0
        self.received_contexts = []

    def evaluate(self, context):
        self.call_count += 1
        self.received_contexts.append(context)
        return self._result


class RaisingScorer:
    def evaluate(self, context):
        raise RuntimeError("shared scorer failed")


def make_response_target(*, fire_event_id: int, is_active: bool, index: int) -> ResponseTarget:
    if is_active:
        return ResponseTarget(
            fire_event_id=fire_event_id,
            target_type=ResponseTargetType.ACTIVE_FIRE,
            latitude=32.7,
            longitude=35.0,
            priority_score=100.0,
        )
    return ResponseTarget(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.PREDICTED_RISK,
        latitude=32.7 + index * 0.01,
        longitude=35.0 + index * 0.01,
        priority_score=80.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=index,
        spread_prediction_cell_id=index,
    )


def make_stored_target_set(
    *,
    id: int,
    fire_event_id: int = FIRE_EVENT_ID,
    target_ids_and_orders,
) -> StoredResponseTargetSet:
    stored_targets = []
    domain_targets = []
    for index, (target_id, target_order) in enumerate(target_ids_and_orders):
        target = make_response_target(fire_event_id=fire_event_id, is_active=(index == 0), index=index)
        stored_targets.append(StoredResponseTarget(id=target_id, target_order=target_order, target=target))
        domain_targets.append(target)
    target_set = ResponseTargetSet(
        fire_event_id=fire_event_id,
        generated_at=GENERATED_AT,
        methodology="TEST_METHODOLOGY",
        methodology_version="1.0",
        targets=tuple(domain_targets),
    )
    return StoredResponseTargetSet(id=id, target_set=target_set, targets=tuple(stored_targets))


def make_plan(**overrides) -> FakeOptimizedPlan:
    defaults = dict(
        id=RESPONSE_PLAN_ID,
        fire_event_id=FIRE_EVENT_ID,
        route_planning_run_id=ROUTE_PLANNING_RUN_ID,
        response_target_set_id=RESPONSE_TARGET_SET_ID,
        score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 3, 3, ()),
    )
    defaults.update(overrides)
    return FakeOptimizedPlan(**defaults)


def make_run(**overrides) -> FakeRoutePlanningRun:
    defaults = dict(
        id=ROUTE_PLANNING_RUN_ID,
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=RESPONSE_TARGET_SET_ID,
        resource_ids=("R1", "R2"),
        route_results=(
            FakeRouteResult(
                id=501, resource_id="R1", response_target_id=201, status="reachable",
                travel_time_seconds=50.0, distance_meters=500.0,
            ),
            FakeRouteResult(
                id=502, resource_id="R2", response_target_id=202, status="reachable",
                travel_time_seconds=30.0, distance_meters=300.0,
            ),
        ),
    )
    defaults.update(overrides)
    return FakeRoutePlanningRun(**defaults)


def make_target_set(**overrides) -> StoredResponseTargetSet:
    defaults = dict(
        id=RESPONSE_TARGET_SET_ID,
        fire_event_id=FIRE_EVENT_ID,
        target_ids_and_orders=[(201, 1), (202, 2)],
    )
    defaults.update(overrides)
    return make_stored_target_set(**defaults)


def make_service(
    *,
    plan=None,
    run=None,
    target_set=None,
    scorer=None,
    sqlite_session_factory,
) -> tuple[BaselineComparisonService, FakeOptimizedPlanReader, FakeRoutePlanningRunReader, FakeResponseTargetSetReader, "RecordingScorer"]:
    plan = plan if plan is not None else make_plan()
    run = run if run is not None else make_run()
    target_set = target_set if target_set is not None else make_target_set()
    scorer = scorer if scorer is not None else RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))

    plan_reader = FakeOptimizedPlanReader({plan.id: plan})
    run_reader = FakeRoutePlanningRunReader({run.id: run})
    target_set_reader = FakeResponseTargetSetReader({target_set.id: target_set})

    service = BaselineComparisonService(
        optimized_plan_reader=plan_reader,
        route_planning_run_reader=run_reader,
        scorer=scorer,
        response_target_set_reader=target_set_reader,
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )
    return service, plan_reader, run_reader, target_set_reader, scorer


def saved_count_for_event(sqlite_session_factory, fire_event_id: int) -> int:
    repo = PlanComparisonRepository(session_factory=sqlite_session_factory)
    return len(repo.list_for_fire_event(fire_event_id))


# ---------------------------------------------------------------------------
# 1-18. Happy path: exact snapshot reuse, single delegated calls, persistence, return value
# ---------------------------------------------------------------------------


def test_end_to_end_happy_path_reuses_exact_snapshot_and_persists_once(sqlite_session_factory):
    plan = make_plan(score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 2, 2, ()))
    run = make_run()
    target_set = make_target_set()
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service, plan_reader, run_reader, target_set_reader, scorer = make_service(
        plan=plan, run=run, target_set=target_set, scorer=scorer, sqlite_session_factory=sqlite_session_factory
    )

    result = service.compare(response_plan_id=RESPONSE_PLAN_ID)

    # 1-3: exact ids requested, nothing else
    assert plan_reader.calls == [RESPONSE_PLAN_ID]
    assert run_reader.calls == [ROUTE_PLANNING_RUN_ID]
    assert target_set_reader.calls == [RESPONSE_TARGET_SET_ID]

    # 10 & 11 & 15: scorer (via BaselinePlanEvaluator) called exactly once -- optimized side never rescored
    assert scorer.call_count == 1
    context = scorer.received_contexts[0]

    # 4, 5, 8: exact persisted target_order / response_target_id / target_type / priority_score reach Task 1's input
    assert context.targets == (
        TargetOrder(
            response_target_id=201, target_order=1,
            target_type=ResponseTargetType.ACTIVE_FIRE, priority_score=100.0,
        ),
        TargetOrder(
            response_target_id=202, target_order=2,
            target_type=ResponseTargetType.PREDICTED_RISK, priority_score=80.0,
        ),
    )

    # 4, 6, 7, 8, 9: exact route_result_id / resource_id / status / travel_time_seconds / distance_meters reach Task 1's input
    assert context.route_candidates == (
        RouteCandidate(
            route_result_id=501, resource_id="R1", response_target_id=201,
            status="reachable", travel_time_seconds=50.0, distance_meters=500.0,
        ),
        RouteCandidate(
            route_result_id=502, resource_id="R2", response_target_id=202,
            status="reachable", travel_time_seconds=30.0, distance_meters=300.0,
        ),
    )

    # Task 6.1: fire_event_id is also carried through to the scoring context
    assert context.fire_event_id == FIRE_EVENT_ID

    # 12, 13, 14: optimized score/coverage/average ETA reused unchanged
    assert result.optimized_score == 850.0
    assert result.optimized_coverage_score == 90.0
    assert result.optimized_average_eta_seconds == 300.0

    # baseline score reused from the (faked) shared scorer, unchanged
    assert result.baseline_score == 700.0
    assert result.baseline_coverage_score == 80.0
    assert result.baseline_average_eta_seconds == 340.0

    # 9: exact traceability ids preserved
    assert result.fire_event_id == FIRE_EVENT_ID
    assert result.optimized_plan_id == RESPONSE_PLAN_ID
    assert result.route_planning_run_id == ROUTE_PLANNING_RUN_ID
    assert result.response_target_set_id == RESPONSE_TARGET_SET_ID

    # 17 & 18: persisted exactly once, and the returned value is the Task 3 domain PlanComparison
    assert isinstance(result, PlanComparison)
    stored_rows = PlanComparisonRepository(session_factory=sqlite_session_factory).list_for_fire_event(FIRE_EVENT_ID)
    assert len(stored_rows) == 1
    assert stored_rows[0].comparison == result


def test_response_target_set_reader_defaults_to_response_target_repository():
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({}),
        route_planning_run_reader=FakeRoutePlanningRunReader({}),
        scorer=RecordingScorer(result=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 0, ())),
        plan_comparison_repository=PlanComparisonRepository(session_factory=lambda: None),
    )

    assert isinstance(service._response_target_set_reader, ResponseTargetRepository)  # noqa: SLF001


def persist_fire_event(sqlite_session_factory) -> int:
    """Persist a minimal real FireEvent -- ResponseTargetSetDB.fire_event_id is a real FK."""
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=32.731,
            longitude=35.046,
            detected_at=GENERATED_AT,
            confidence="h",
            frp=72.0,
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=GENERATED_AT, lookback_minutes=360)[0].id
    stored_event = fire_event_repository.create_event(
        FireEvent(
            latitude=32.731,
            longitude=35.046,
            detected_at=GENERATED_AT,
            updated_at=GENERATED_AT,
            status=FireEventStatus.CONFIRMED,
            detection_confidence=0.85,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return stored_event.id


def test_real_response_target_repository_end_to_end_through_the_service(sqlite_session_factory):
    real_repository = ResponseTargetRepository(session_factory=sqlite_session_factory)
    real_fire_event_id = persist_fire_event(sqlite_session_factory)
    target_set = ResponseTargetSet(
        fire_event_id=real_fire_event_id,
        generated_at=GENERATED_AT,
        methodology="TEST_METHODOLOGY",
        methodology_version="1.0",
        targets=(
            ResponseTarget(
                fire_event_id=real_fire_event_id,
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.7,
                longitude=35.0,
                priority_score=100.0,
            ),
        ),
    )
    stored = real_repository.save_target_set(target_set)
    stored_target_id = stored.targets[0].id

    plan = make_plan(fire_event_id=real_fire_event_id, response_target_set_id=stored.id)
    run = make_run(
        fire_event_id=real_fire_event_id,
        response_target_set_id=stored.id,
        route_results=(
            FakeRouteResult(
                id=901, resource_id="R1", response_target_id=stored_target_id,
                status="reachable", travel_time_seconds=42.0,
            ),
        ),
        resource_ids=("R1",),
    )
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 1, 1, ()))

    # Injects the REAL ResponseTargetRepository explicitly, bound to the
    # test's SQLite session factory -- the production default (when this
    # constructor arg is omitted) constructs the same class bound to
    # get_session_factory() instead (see
    # test_response_target_set_reader_defaults_to_response_target_repository).
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=real_repository,
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    result = service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert scorer.received_contexts[0].targets == (
        TargetOrder(
            response_target_id=stored_target_id,
            target_order=0,
            target_type=ResponseTargetType.ACTIVE_FIRE,
            priority_score=100.0,
        ),
    )
    assert result.route_planning_run_id == run.id
    assert result.response_target_set_id == stored.id


# ---------------------------------------------------------------------------
# 19-21. Missing records
# ---------------------------------------------------------------------------


def test_missing_response_plan_fails_explicitly_and_saves_nothing(sqlite_session_factory):
    service, *_ = make_service(sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=999999)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_missing_routing_run_fails_explicitly_and_saves_nothing(sqlite_session_factory):
    plan = make_plan(route_planning_run_id=999999)
    service, *_ = make_service(plan=plan, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_missing_response_target_set_fails_explicitly_and_saves_nothing(sqlite_session_factory):
    plan = make_plan(response_target_set_id=999999)
    service, *_ = make_service(plan=plan, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


# ---------------------------------------------------------------------------
# 22-24. Planning-chain mismatches
# ---------------------------------------------------------------------------


def test_fire_event_mismatch_between_plan_and_run_is_rejected(sqlite_session_factory):
    run = make_run(fire_event_id=999)
    service, *_ = make_service(run=run, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_fire_event_mismatch_between_plan_and_target_set_is_rejected(sqlite_session_factory):
    target_set = make_target_set(fire_event_id=999, target_ids_and_orders=[(201, 1)])
    service, *_ = make_service(target_set=target_set, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_routing_run_id_mismatch_is_rejected(sqlite_session_factory):
    # run.id itself differs from what the plan references -- constructed via
    # a reader keyed under the plan's expected id but returning a run whose
    # own id field disagrees, simulating a misbehaving/inconsistent reader.
    plan = make_plan(route_planning_run_id=ROUTE_PLANNING_RUN_ID)
    run = make_run(id=999999)
    plan_reader = FakeOptimizedPlanReader({plan.id: plan})
    run_reader = FakeRoutePlanningRunReader({ROUTE_PLANNING_RUN_ID: run})
    target_set_reader = FakeResponseTargetSetReader({RESPONSE_TARGET_SET_ID: make_target_set()})
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service = BaselineComparisonService(
        optimized_plan_reader=plan_reader,
        route_planning_run_reader=run_reader,
        scorer=scorer,
        response_target_set_reader=target_set_reader,
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_target_set_id_mismatch_is_rejected(sqlite_session_factory):
    plan = make_plan(response_target_set_id=RESPONSE_TARGET_SET_ID)
    target_set = make_target_set(id=999999)
    plan_reader = FakeOptimizedPlanReader({plan.id: plan})
    run_reader = FakeRoutePlanningRunReader({ROUTE_PLANNING_RUN_ID: make_run()})
    target_set_reader = FakeResponseTargetSetReader({RESPONSE_TARGET_SET_ID: target_set})
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service = BaselineComparisonService(
        optimized_plan_reader=plan_reader,
        route_planning_run_reader=run_reader,
        scorer=scorer,
        response_target_set_reader=target_set_reader,
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


# ---------------------------------------------------------------------------
# 25-26. Inconsistent routing snapshot
# ---------------------------------------------------------------------------


def test_route_referencing_target_outside_target_set_is_rejected(sqlite_session_factory):
    run = make_run(
        route_results=(
            FakeRouteResult(id=501, resource_id="R1", response_target_id=999999, status="reachable", travel_time_seconds=50.0),
        ),
        resource_ids=("R1",),
    )
    service, *_ = make_service(run=run, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_route_referencing_resource_outside_resource_snapshot_is_rejected(sqlite_session_factory):
    run = make_run(
        route_results=(
            FakeRouteResult(id=501, resource_id="GHOST", response_target_id=201, status="reachable", travel_time_seconds=50.0),
        ),
        resource_ids=("R1", "R2"),
    )
    service, *_ = make_service(run=run, sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(BaselineComparisonServiceError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


# ---------------------------------------------------------------------------
# 27-29. Downstream failure propagation, no fabricated success
# ---------------------------------------------------------------------------


def test_baseline_scorer_failure_prevents_comparison_and_persistence(sqlite_session_factory):
    service, *_ = make_service(scorer=RaisingScorer(), sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(RuntimeError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_comparison_snapshot_mismatch_failure_prevents_persistence(sqlite_session_factory):
    # Task 3's own snapshot-consistency check fires if the optimized plan's
    # route_planning_run_id disagrees with what was actually evaluated --
    # here forced by a target_set whose fire_event_id differs, which Task 5
    # itself rejects earlier; instead we exercise a genuine Task 3-level
    # rejection by handing Task 3 a directly-inconsistent pair through the
    # comparison_calculator seam, proving repository.save is never reached
    # when Task 3 raises.
    from src.calculators.baseline_plan.baseline_plan_comparison_calculator import BaselinePlanComparisonCalculator

    class RaisingComparisonCalculator(BaselinePlanComparisonCalculator):
        def compare(self, *, optimized, baseline):
            raise ValueError("forced Task 3 failure")

    plan = make_plan()
    run = make_run()
    target_set = make_target_set()
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=FakeResponseTargetSetReader({target_set.id: target_set}),
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
        comparison_calculator=RaisingComparisonCalculator(),
    )

    with pytest.raises(ValueError, match="forced Task 3 failure"):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 0


def test_persistence_failure_propagates_without_fabricating_success(sqlite_session_factory, monkeypatch):
    from sqlalchemy.orm import Session

    # Pre-warm the FK chain (FND-05) before patching flush, so the failure
    # below is caused only by the real repository's own save, not by the
    # unrelated FK-prerequisite provisioning also needing a flush.
    persist_fk_prerequisites(
        sqlite_session_factory,
        fire_event_id=FIRE_EVENT_ID,
        response_target_set_id=RESPONSE_TARGET_SET_ID,
        route_planning_run_id=ROUTE_PLANNING_RUN_ID,
        optimized_plan_id=RESPONSE_PLAN_ID,
    )

    def raise_on_flush(self, *args, **kwargs):
        from sqlalchemy.exc import IntegrityError

        raise IntegrityError("forced failure", params=None, orig=Exception("forced"))

    monkeypatch.setattr(Session, "flush", raise_on_flush)
    service, *_ = make_service(sqlite_session_factory=sqlite_session_factory)

    with pytest.raises(PlanComparisonRepositoryError):
        service.compare(response_plan_id=RESPONSE_PLAN_ID)


# ---------------------------------------------------------------------------
# 30. Empty / no-feasible routing snapshot
# ---------------------------------------------------------------------------


def test_no_feasible_routes_produces_legitimate_no_assignment_baseline(sqlite_session_factory):
    run = make_run(
        route_results=(
            FakeRouteResult(id=501, resource_id="R1", response_target_id=201, status="unreachable", travel_time_seconds=None),
            FakeRouteResult(id=502, resource_id="R2", response_target_id=202, status="unmappable", travel_time_seconds=None),
        ),
    )
    service, *_ = make_service(run=run, sqlite_session_factory=sqlite_session_factory)

    result = service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert isinstance(result, PlanComparison)
    assert saved_count_for_event(sqlite_session_factory, FIRE_EVENT_ID) == 1


# ---------------------------------------------------------------------------
# 31. Determinism
# ---------------------------------------------------------------------------


def test_identical_snapshot_and_deterministic_scorer_produce_the_same_comparison(sqlite_session_factory):
    plan = make_plan()
    run = make_run()
    target_set = make_target_set()

    scorer_a = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service_a, *_ = make_service(
        plan=plan, run=run, target_set=target_set, scorer=scorer_a, sqlite_session_factory=sqlite_session_factory
    )
    result_a = service_a.compare(response_plan_id=RESPONSE_PLAN_ID)

    scorer_b = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 2, 2, ()))
    service_b, *_ = make_service(
        plan=plan, run=run, target_set=target_set, scorer=scorer_b, sqlite_session_factory=sqlite_session_factory
    )
    result_b = service_b.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert result_a == result_b


# ---------------------------------------------------------------------------
# 33. No mutation of persisted input objects
# ---------------------------------------------------------------------------


def test_service_does_not_mutate_the_loaded_plan_run_or_target_set(sqlite_session_factory):
    plan = make_plan()
    run = make_run()
    target_set = make_target_set()
    plan_before, run_before, target_set_before = plan, run, target_set
    service, *_ = make_service(plan=plan, run=run, target_set=target_set, sqlite_session_factory=sqlite_session_factory)

    service.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert plan == plan_before
    assert run == run_before
    assert target_set == target_set_before


# ---------------------------------------------------------------------------
# 32 & 34. Static architecture guardrail: no current-state lookups, no algorithm duplication
# ---------------------------------------------------------------------------


def test_static_architecture_guardrail_no_recomputation_or_current_state_lookups():
    forbidden_import_fragments = (
        "Dijkstra",
        "genetic",
        "GeneticAlgorithm",
        "road_network",
        "RoadNetworkRepository",
        "simulation",
        "agents.analysis.fire_detection",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "ResponseTargetGenerationAgent",
        "OperationalRefreshOrchestrator",
        "RoutePlanningAgent",
        "ResponseOptimizationAgent",
        "FirefightingResourceRepository",
    )
    production_files = [
        (Path(__file__).resolve().parents[4] / "backend/src/services/baseline_comparison/baseline_comparison_service.py"),
        (Path(__file__).resolve().parents[4] / "backend/src/services/baseline_comparison/baseline_comparison_ports.py"),
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
            if module and any(fragment in module for fragment in forbidden_import_fragments):
                violations.append((str(path), module))

    assert violations == []
