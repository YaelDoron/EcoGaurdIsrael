"""Acceptance tests for User Story 5.3 Task 6: final acceptance verification.

Tasks 1-5 already have exhaustive unit/service-level coverage:
- 26 tests for the pure greedy baseline (`test_baseline_plan_calculator.py`)
- 18 tests for shared-scoring integration (`test_baseline_plan_evaluator.py`)
- 20 tests for the pure comparison (`test_baseline_plan_comparison_calculator.py`)
- 19 tests for append-only persistence (`test_plan_comparison_repository.py`)
- 19 tests for orchestration (`test_baseline_comparison_service.py`), including
  an end-to-end happy path, a real-`ResponseTargetRepository` round trip,
  every missing-record/snapshot-mismatch/failure-propagation path, a
  determinism check, and a static architecture guardrail.

This file adds ONLY the higher-level acceptance scenarios that prove
behavior *across* Tasks 1-5 together and are not already covered above:
a same-snapshot-vs-newer-decoy-data proof (Scenario B), a single rich
multi-target/multi-resource scenario proving target_order processing,
nearest-unused-reachable selection, tie-break, unreachable/unmappable
exclusion, and partial coverage all survive the full real pipeline
unchanged (Scenarios C-G), a negative-improvement/None-ETA round trip
(Scenarios L/N), a baseline_score == 0 round trip (Scenario M), an
append-only same-plan-twice proof (Scenario P), and a two-different-
snapshots no-cross-contamination proof (Scenario Q).

Every scenario uses the REAL `BaselinePlanCalculator` (via
`BaselinePlanEvaluator`), `BaselinePlanComparisonCalculator`,
`PlanComparisonRepository`, and `BaselineComparisonService`. Only the
still-missing Company 1/2 boundaries (`OptimizedPlanReader`,
`RoutePlanningRunReader`, the shared scorer) are test doubles -- see
`baseline_comparison_ports.py` for why. `ResponseTargetSetReader` is
exercised through the REAL `ResponseTargetRepository` in Scenario B and
the "real Company-3 flow" scenario, exactly matching how it is wired in
production by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

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
from src.repositories.plan_comparison_repository import PlanComparisonRepository
from src.repositories.response_target_repository import (
    ResponseTargetRepository,
    StoredResponseTarget,
    StoredResponseTargetSet,
)
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.baseline_comparison import BaselineComparisonService

GENERATED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test doubles (external Company 1/2 boundaries only -- see module docstring)
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

    def get_by_id(self, response_plan_id: int) -> FakeOptimizedPlan | None:
        return self._plans.get(response_plan_id)


class FakeRoutePlanningRunReader:
    def __init__(self, runs: dict[int, FakeRoutePlanningRun]) -> None:
        self._runs = runs

    def get_by_id(self, route_planning_run_id: int) -> FakeRoutePlanningRun | None:
        return self._runs.get(route_planning_run_id)


class FakeResponseTargetSetReader:
    """In-memory reader built from REAL StoredResponseTargetSet/StoredResponseTarget.

    Deliberately preserves whatever tuple order it is given, unlike the real
    `ResponseTargetRepository` (which always returns targets pre-sorted by
    target_order) -- this is what lets Scenario C prove target_order, not
    tuple/list position, drives processing order.
    """

    def __init__(self, target_sets: dict[int, StoredResponseTargetSet]) -> None:
        self._target_sets = target_sets

    def get_by_id(self, response_target_set_id: int) -> StoredResponseTargetSet | None:
        return self._target_sets.get(response_target_set_id)


class RecordingScorer:
    def __init__(self, *, result: FakePlanScoreBreakdown) -> None:
        self._result = result
        self.call_count = 0
        self.received_contexts = []

    def evaluate(self, context):
        self.call_count += 1
        self.received_contexts.append(context)
        return self._result


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
    fire_event_id: int,
    target_ids_and_orders,
    active_index: int = 0,
) -> StoredResponseTargetSet:
    """Build a real StoredResponseTargetSet; `target_ids_and_orders` supplies (id, target_order) pairs
    in the exact tuple order the fake reader will return them, independent of target_order."""
    stored_targets = []
    domain_targets = []
    for index, (target_id, target_order) in enumerate(target_ids_and_orders):
        target = make_response_target(fire_event_id=fire_event_id, is_active=(index == active_index), index=index)
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


def persist_fire_event(sqlite_session_factory, **overrides) -> int:
    """Persist a minimal real FireEvent -- ResponseTargetSetDB.fire_event_id is a real FK."""
    fire_event_repository = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=32.731 + overrides.get("offset", 0) * 0.1,
            longitude=35.046,
            detected_at=GENERATED_AT,
            confidence="h",
            frp=72.0,
            satellite="N20",
        )
    )
    hotspot_id = max(
        record.id for record in satellite_repository.get_recent_hotspots(as_of=GENERATED_AT, lookback_minutes=360)
    )
    stored_event = fire_event_repository.create_event(
        FireEvent(
            latitude=32.731 + overrides.get("offset", 0) * 0.1,
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


def saved_rows_for_event(sqlite_session_factory, fire_event_id: int):
    repo = PlanComparisonRepository(session_factory=sqlite_session_factory)
    return repo.list_for_fire_event(fire_event_id)


# ---------------------------------------------------------------------------
# FND-05: plan_comparisons.fire_event_id/optimized_plan_id/
# route_planning_run_id/response_target_set_id are now real FKs. These
# scenarios build FakeOptimizedPlan/FakeRoutePlanningRun with hand-picked
# ids with no real backing rows, so wrap PlanComparisonRepository to
# provision a minimal, independent row for each of the 4 ids before
# delegating to the real save() (mirrors test_plan_comparison_repository.py
# and test_baseline_comparison_service.py).
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
    """Wraps PlanComparisonRepository so these scenarios can keep using
    arbitrary hand-picked traceability ids without each one separately
    pre-creating the FK chain plan_comparisons now requires (FND-05)."""

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
# Scenario B -- fair same-snapshot comparison: never substitutes newer/latest data
# ---------------------------------------------------------------------------


def test_scenario_b_uses_exact_referenced_snapshot_never_the_newer_decoy(sqlite_session_factory):
    real_targets = ResponseTargetRepository(session_factory=sqlite_session_factory)
    fire_event_id = persist_fire_event(sqlite_session_factory)

    referenced_set = ResponseTargetSet(
        fire_event_id=fire_event_id,
        generated_at=GENERATED_AT,
        methodology="TEST_METHODOLOGY",
        methodology_version="1.0",
        targets=(
            ResponseTarget(
                fire_event_id=fire_event_id,
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.7,
                longitude=35.0,
                priority_score=100.0,
            ),
        ),
    )
    stored_referenced = real_targets.save_target_set(referenced_set)

    # A newer "decoy" set for the SAME FireEvent, generated later -- this is
    # what get_latest_for_event_as_of would return if the service mistakenly
    # queried "current" data instead of the plan's exact reference.
    decoy_set = ResponseTargetSet(
        fire_event_id=fire_event_id,
        generated_at=GENERATED_AT + timedelta(hours=1),
        methodology="TEST_METHODOLOGY",
        methodology_version="1.0",
        targets=(
            ResponseTarget(
                fire_event_id=fire_event_id,
                target_type=ResponseTargetType.ACTIVE_FIRE,
                latitude=32.9,
                longitude=35.2,
                priority_score=100.0,
            ),
        ),
    )
    stored_decoy = real_targets.save_target_set(decoy_set)

    # Confirm the decoy really is what "latest" resolves to, proving this is
    # a meaningful trap and not a no-op check.
    latest = real_targets.get_latest_for_event_as_of(fire_event_id, as_of=GENERATED_AT + timedelta(hours=2))
    assert latest.id == stored_decoy.id
    assert latest.id != stored_referenced.id

    referenced_target_id = stored_referenced.targets[0].id
    plan = FakeOptimizedPlan(
        id=1,
        fire_event_id=fire_event_id,
        route_planning_run_id=10,
        response_target_set_id=stored_referenced.id,
        score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 1, 1, ()),
    )
    run = FakeRoutePlanningRun(
        id=10,
        fire_event_id=fire_event_id,
        response_target_set_id=stored_referenced.id,
        resource_ids=("R1",),
        route_results=(
            FakeRouteResult(
                id=701, resource_id="R1", response_target_id=referenced_target_id,
                status="reachable", travel_time_seconds=42.0,
            ),
        ),
    )
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 1, 1, ()))
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=real_targets,
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    result = service.compare(response_plan_id=plan.id)

    # The service used exactly the referenced set's target, never the decoy's.
    assert scorer.received_contexts[0].targets[0].response_target_id == referenced_target_id
    assert result.response_target_set_id == stored_referenced.id


# ---------------------------------------------------------------------------
# Scenarios C-G -- target_order, nearest-unused-reachable, tie-break,
# unreachable/unmappable exclusion, and partial coverage, all through the
# REAL Task 1-3 pipeline (Task 1's algorithm itself is not re-derived here;
# these outcomes follow directly from its already-tested rules).
# ---------------------------------------------------------------------------


def test_scenarios_c_to_g_full_pipeline_reuses_persisted_snapshot_correctly(sqlite_session_factory):
    fire_event_id = 1
    # Deliberately NOT sorted by target_order (301:0, 303:1, 302:2, 304:3) --
    # proves Task 1 sorts by the persisted target_order field, not list order.
    target_set = make_stored_target_set(
        id=20,
        fire_event_id=fire_event_id,
        target_ids_and_orders=[(304, 3), (301, 0), (303, 1), (302, 2)],
    )
    run = FakeRoutePlanningRun(
        id=10,
        fire_event_id=fire_event_id,
        response_target_set_id=20,
        resource_ids=("R1", "R2", "R4", "R5", "R6"),
        route_results=(
            # Target 301 (order 0): R1 and R2 tie at ETA 100 -> lowest resource_id (R1) wins.
            FakeRouteResult(id=601, resource_id="R1", response_target_id=301, status="reachable", travel_time_seconds=100.0),
            FakeRouteResult(id=602, resource_id="R2", response_target_id=301, status="reachable", travel_time_seconds=100.0),
            # Target 303 (order 1): R1 would be fastest (ETA 5) but is already
            # used by 301 -> R4 (ETA 20) wins instead. Proves no resource reuse.
            FakeRouteResult(id=603, resource_id="R1", response_target_id=303, status="reachable", travel_time_seconds=5.0),
            FakeRouteResult(id=604, resource_id="R4", response_target_id=303, status="reachable", travel_time_seconds=20.0),
            # Target 302 (order 2): only unreachable/unmappable candidates -> uncovered.
            FakeRouteResult(id=605, resource_id="R5", response_target_id=302, status="unreachable", travel_time_seconds=None),
            FakeRouteResult(id=606, resource_id="R6", response_target_id=302, status="unmappable", travel_time_seconds=None),
            # Target 304 (order 3): no route candidates at all -> uncovered.
        ),
    )
    plan = FakeOptimizedPlan(
        id=1, fire_event_id=fire_event_id, route_planning_run_id=10, response_target_set_id=20,
        score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 4, 4, ()),
    )
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(400.0, 50.0, 60.0, 2, 4, (302, 304)))
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=FakeResponseTargetSetReader({target_set.id: target_set}),
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    result = service.compare(response_plan_id=plan.id)

    allocation = scorer.received_contexts[0].allocation
    assignments_by_target = {a.response_target_id: a for a in allocation.assignments}

    # C: processed by target_order (0,1,2,3), not by the scrambled input list order.
    assert tuple(a.response_target_id for a in allocation.assignments) == (301, 303)
    # D & E: nearest-unused-reachable with deterministic tie-break.
    assert assignments_by_target[301].resource_id == "R1"
    assert assignments_by_target[301].route_result_id == 601
    # D: R1 is NOT reused for target 303 even though it would have been faster.
    assert assignments_by_target[303].resource_id == "R4"
    assert assignments_by_target[303].route_result_id == 604
    # F & G: unreachable/unmappable excluded; both targets with no eligible
    # resource remain legitimately uncovered, no fabricated assignment.
    assert allocation.uncovered_response_target_ids == (302, 304)
    assert 302 not in assignments_by_target
    assert 304 not in assignments_by_target

    # The comparison and persisted row reflect this legitimate partial baseline.
    assert result.baseline_score == 400.0
    assert result.baseline_coverage_score == 50.0
    assert saved_rows_for_event(sqlite_session_factory, fire_event_id)[0].comparison == result


# ---------------------------------------------------------------------------
# Scenario L & N -- optimized worse than baseline: negative difference
# preserved exactly, no clamping; None average ETA preserved.
# ---------------------------------------------------------------------------


def test_scenario_l_optimized_worse_than_baseline_is_preserved_without_clamping(sqlite_session_factory):
    fire_event_id = 1
    target_set = make_stored_target_set(id=20, fire_event_id=fire_event_id, target_ids_and_orders=[(201, 0)])
    run = FakeRoutePlanningRun(
        id=10, fire_event_id=fire_event_id, response_target_set_id=20, resource_ids=("R1",),
        route_results=(FakeRouteResult(id=501, resource_id="R1", response_target_id=201, status="reachable", travel_time_seconds=50.0),),
    )
    # Optimized plan scored worse than the baseline will score, and has no
    # average ETA on record (a legal scorer-defined no-ETA state).
    plan = FakeOptimizedPlan(
        id=1, fire_event_id=fire_event_id, route_planning_run_id=10, response_target_set_id=20,
        score=FakePlanScoreBreakdown(650.0, 70.0, None, 1, 1, ()),
    )
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 1, 1, ()))
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=FakeResponseTargetSetReader({target_set.id: target_set}),
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    result = service.compare(response_plan_id=plan.id)

    assert result.score_difference == pytest.approx(-50.0)
    assert result.improvement_percentage == pytest.approx(((650.0 - 700.0) / 700.0) * 100)
    assert result.improvement_percentage < 0
    assert result.optimized_average_eta_seconds is None
    assert result.baseline_average_eta_seconds == 340.0

    stored = saved_rows_for_event(sqlite_session_factory, fire_event_id)
    assert len(stored) == 1
    assert stored[0].comparison == result  # persisted exactly, unrounded, unclamped


# ---------------------------------------------------------------------------
# Scenario M -- baseline_score == 0
# ---------------------------------------------------------------------------


def test_scenario_m_baseline_score_zero_yields_none_improvement_and_round_trips(sqlite_session_factory):
    fire_event_id = 1
    target_set = make_stored_target_set(id=20, fire_event_id=fire_event_id, target_ids_and_orders=[(201, 0)])
    run = FakeRoutePlanningRun(
        id=10, fire_event_id=fire_event_id, response_target_set_id=20, resource_ids=(),
        route_results=(),  # no feasible routes -> baseline covers nothing
    )
    plan = FakeOptimizedPlan(
        id=1, fire_event_id=fire_event_id, route_planning_run_id=10, response_target_set_id=20,
        score=FakePlanScoreBreakdown(500.0, 60.0, 250.0, 1, 1, ()),
    )
    scorer = RecordingScorer(result=FakePlanScoreBreakdown(0.0, 0.0, None, 0, 1, (201,)))
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=scorer,
        response_target_set_reader=FakeResponseTargetSetReader({target_set.id: target_set}),
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    result = service.compare(response_plan_id=plan.id)

    assert result.baseline_score == 0.0
    assert result.improvement_percentage is None
    assert result.score_difference == 500.0

    stored = saved_rows_for_event(sqlite_session_factory, fire_event_id)
    assert stored[0].comparison.improvement_percentage is None


# ---------------------------------------------------------------------------
# Scenario P -- append-only history: comparing the same plan twice
# ---------------------------------------------------------------------------


def test_scenario_p_comparing_the_same_plan_twice_appends_two_historical_rows(sqlite_session_factory):
    fire_event_id = 1
    target_set = make_stored_target_set(id=20, fire_event_id=fire_event_id, target_ids_and_orders=[(201, 0)])
    run = FakeRoutePlanningRun(
        id=10, fire_event_id=fire_event_id, response_target_set_id=20, resource_ids=("R1",),
        route_results=(FakeRouteResult(id=501, resource_id="R1", response_target_id=201, status="reachable", travel_time_seconds=50.0),),
    )
    plan = FakeOptimizedPlan(
        id=1, fire_event_id=fire_event_id, route_planning_run_id=10, response_target_set_id=20,
        score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 1, 1, ()),
    )
    service = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan.id: plan}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run.id: run}),
        scorer=RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 1, 1, ())),
        response_target_set_reader=FakeResponseTargetSetReader({target_set.id: target_set}),
        plan_comparison_repository=fk_provisioning_repository(sqlite_session_factory),
    )

    first_result = service.compare(response_plan_id=plan.id)
    second_result = service.compare(response_plan_id=plan.id)

    stored = saved_rows_for_event(sqlite_session_factory, fire_event_id)
    assert len(stored) == 2
    assert stored[0].id != stored[1].id
    # The first row is unchanged by the second comparison (no overwrite/upsert).
    assert stored[0].comparison == first_result
    assert stored[1].comparison == second_result


# ---------------------------------------------------------------------------
# Scenario Q -- two different snapshots, no cross-contamination
# ---------------------------------------------------------------------------


def test_scenario_q_two_different_snapshots_do_not_cross_contaminate(sqlite_session_factory):
    target_set_a = make_stored_target_set(id=20, fire_event_id=1, target_ids_and_orders=[(201, 0)])
    target_set_b = make_stored_target_set(id=21, fire_event_id=2, target_ids_and_orders=[(211, 0)])
    run_a = FakeRoutePlanningRun(
        id=10, fire_event_id=1, response_target_set_id=20, resource_ids=("R1",),
        route_results=(FakeRouteResult(id=501, resource_id="R1", response_target_id=201, status="reachable", travel_time_seconds=50.0),),
    )
    run_b = FakeRoutePlanningRun(
        id=11, fire_event_id=2, response_target_set_id=21, resource_ids=("R9",),
        route_results=(FakeRouteResult(id=511, resource_id="R9", response_target_id=211, status="reachable", travel_time_seconds=15.0),),
    )
    plan_a = FakeOptimizedPlan(
        id=100, fire_event_id=1, route_planning_run_id=10, response_target_set_id=20,
        score=FakePlanScoreBreakdown(850.0, 90.0, 300.0, 1, 1, ()),
    )
    plan_b = FakeOptimizedPlan(
        id=101, fire_event_id=2, route_planning_run_id=11, response_target_set_id=21,
        score=FakePlanScoreBreakdown(500.0, 60.0, 200.0, 1, 1, ()),
    )
    scorer_a = RecordingScorer(result=FakePlanScoreBreakdown(700.0, 80.0, 340.0, 1, 1, ()))
    scorer_b = RecordingScorer(result=FakePlanScoreBreakdown(300.0, 40.0, 100.0, 1, 1, ()))
    plan_comparison_repository = fk_provisioning_repository(sqlite_session_factory)

    service_a = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan_a.id: plan_a}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run_a.id: run_a}),
        scorer=scorer_a,
        response_target_set_reader=FakeResponseTargetSetReader({target_set_a.id: target_set_a}),
        plan_comparison_repository=plan_comparison_repository,
    )
    service_b = BaselineComparisonService(
        optimized_plan_reader=FakeOptimizedPlanReader({plan_b.id: plan_b}),
        route_planning_run_reader=FakeRoutePlanningRunReader({run_b.id: run_b}),
        scorer=scorer_b,
        response_target_set_reader=FakeResponseTargetSetReader({target_set_b.id: target_set_b}),
        plan_comparison_repository=plan_comparison_repository,
    )

    result_a = service_a.compare(response_plan_id=plan_a.id)
    result_b = service_b.compare(response_plan_id=plan_b.id)

    assert result_a.route_planning_run_id == 10
    assert result_a.response_target_set_id == 20
    assert result_a.fire_event_id == 1
    assert result_b.route_planning_run_id == 11
    assert result_b.response_target_set_id == 21
    assert result_b.fire_event_id == 2

    event_1_rows = saved_rows_for_event(sqlite_session_factory, 1)
    event_2_rows = saved_rows_for_event(sqlite_session_factory, 2)
    assert len(event_1_rows) == 1
    assert len(event_2_rows) == 1
    assert event_1_rows[0].comparison == result_a
    assert event_2_rows[0].comparison == result_b
    assert scorer_a.received_contexts[0].targets[0].response_target_id == 201
    assert scorer_b.received_contexts[0].targets[0].response_target_id == 211
