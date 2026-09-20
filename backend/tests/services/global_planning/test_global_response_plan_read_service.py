"""Tests for GlobalResponsePlanReadService (Epic 6 UI/API Rework, Tasks
B-BE-4/B-BE-7), using fakes only - no real DB/service call happens here.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api.schemas.response_plans import (
    CoordinateResponse,
    ResponsePlanActionResponse,
    ResponsePlanDetailResponse,
    ResponsePlanMetricsResponse,
    ResponsePlanResourceResponse,
    ResponsePlanRouteResponse,
    ResponsePlanTargetResponse,
)
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.response_plan_details import ResponsePlanDetails
from src.models.response_plan_status import ResponsePlanStatus
from src.models.routing import RouteStatus
from src.repositories.global_planning_run_repository import StoredGlobalPlanningRun, StoredGlobalPlanningRunEvent
from src.services.global_planning.global_response_plan_read_service import GlobalResponsePlanReadService

STARTED_AT = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
COMPLETED_AT = STARTED_AT.replace(minute=5)
GENERATED_AT = STARTED_AT.replace(minute=1)
AS_OF = STARTED_AT.replace(hour=9)


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeGlobalPlanningRunRepository:
    def __init__(
        self,
        latest_materialized: StoredGlobalPlanningRun | None = None,
        members_by_run_id: dict[int, tuple[StoredGlobalPlanningRunEvent, ...]] | None = None,
    ):
        self._latest_materialized = latest_materialized
        self._members_by_run_id = dict(members_by_run_id or {})
        self.get_latest_materialized_generation_calls = 0
        self.get_members_calls: list[int] = []

    def get_latest_materialized_generation(self) -> StoredGlobalPlanningRun | None:
        self.get_latest_materialized_generation_calls += 1
        return self._latest_materialized

    def get_members(self, global_planning_run_id: int) -> tuple[StoredGlobalPlanningRunEvent, ...]:
        self.get_members_calls.append(global_planning_run_id)
        return self._members_by_run_id.get(global_planning_run_id, ())


class FakeResponsePlanDetailsService:
    def __init__(self, details_by_plan_id: dict[int, ResponsePlanDetails] | None = None):
        self._details_by_plan_id = dict(details_by_plan_id or {})
        self.calls: list[int] = []

    def get_plan_details_by_id(self, plan_id: int) -> ResponsePlanDetails | None:
        self.calls.append(plan_id)
        return self._details_by_plan_id.get(plan_id)


class FakeResponsePlanPresenter:
    def __init__(self, response_by_plan_id: dict[int, ResponsePlanDetailResponse] | None = None):
        self._response_by_plan_id = dict(response_by_plan_id or {})
        self.calls: list[ResponsePlanDetails] = []

    def present(self, details: ResponsePlanDetails) -> ResponsePlanDetailResponse:
        self.calls.append(details)
        return self._response_by_plan_id[details.plan_id]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_run(
    run_id: int,
    *,
    status: GlobalPlanningRunStatus = GlobalPlanningRunStatus.COMPLETED,
    fitness_score: float | None = 88.0,
    coverage_score: float | None = 0.9,
    average_eta_seconds: float | None = 150.0,
    random_seed: int | None = 42,
    ga_population_size: int | None = 50,
    ga_generation_count: int | None = 100,
    ga_mutation_rate: float | None = 0.1,
    ga_crossover_rate: float | None = 0.8,
) -> StoredGlobalPlanningRun:
    return StoredGlobalPlanningRun(
        id=run_id,
        run=GlobalPlanningRun(
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            status=status,
            trigger="scheduled",
            methodology="legacy_per_event_orchestration",
            methodology_version="1.0",
            input_fingerprint="f" * 64,
            random_seed=random_seed,
            ga_population_size=ga_population_size,
            ga_generation_count=ga_generation_count,
            ga_mutation_rate=ga_mutation_rate,
            ga_crossover_rate=ga_crossover_rate,
            fitness_score=fitness_score,
            coverage_score=coverage_score,
            average_eta_seconds=average_eta_seconds,
        ),
    )


def make_member(
    member_id: int,
    fire_event_id: int,
    *,
    event_order: int = 0,
    result_status: GlobalPlanningRunEventStatus | None = GlobalPlanningRunEventStatus.PLANNED,
    response_plan_id: int | None = None,
    severity_level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
    severity_score: float | None = 70.0,
    minimum_resources: int | None = 2,
    desired_resources: int | None = 4,
    assigned_resources: int | None = 3,
    required_slots_uncovered: int | None = 0,
    desired_slots_uncovered: int | None = 1,
    coverage_score: float | None = 0.85,
    average_eta_seconds: float | None = 140.0,
) -> StoredGlobalPlanningRunEvent:
    required_slots_covered = (
        minimum_resources - required_slots_uncovered
        if minimum_resources is not None and required_slots_uncovered is not None
        else None
    )
    desired_slots_covered = (
        desired_resources - desired_slots_uncovered
        if desired_resources is not None and desired_slots_uncovered is not None
        else None
    )
    return StoredGlobalPlanningRunEvent(
        id=member_id,
        member=GlobalPlanningRunEvent(
            fire_event_id=fire_event_id,
            event_order=event_order,
            result_status=result_status,
            response_plan_id=response_plan_id,
            local_state_fingerprint="a" * 64 if response_plan_id is not None else None,
            error_code=None,
            severity_level=severity_level,
            severity_score=severity_score,
            minimum_resources=minimum_resources,
            desired_resources=desired_resources,
            assigned_resources=assigned_resources,
            required_slots_covered=required_slots_covered,
            required_slots_uncovered=required_slots_uncovered,
            desired_slots_covered=desired_slots_covered,
            desired_slots_uncovered=desired_slots_uncovered,
            coverage_score=coverage_score,
            average_eta_seconds=average_eta_seconds,
        ),
    )


def make_details(plan_id: int, fire_event_id: int, **overrides) -> ResponsePlanDetails:
    values = dict(
        plan_id=plan_id,
        fire_event_id=fire_event_id,
        response_target_set_id=9,
        route_planning_run_id=11,
        generated_at=GENERATED_AT,
        methodology="genetic_algorithm",
        methodology_version="1.0",
        random_seed=1,
        is_current=True,
        plan_score=95.0,
        coverage_score=0.9,
        average_eta_seconds=130.0,
        actions=(),
        uncovered_target_ids=(),
        baseline_comparison=None,
        optimization_config=None,
    )
    values.update(overrides)
    return ResponsePlanDetails(**values)


def make_presented(plan_id: int, fire_event_id: int, **overrides) -> ResponsePlanDetailResponse:
    values = dict(
        plan_id=plan_id,
        fire_event_id=fire_event_id,
        response_target_set_id=9,
        route_planning_run_id=11,
        generated_at=GENERATED_AT,
        methodology="genetic_algorithm",
        methodology_version="1.0",
        random_seed=1,
        status=ResponsePlanStatus.COMPLETE,
        is_current=True,
        metrics=ResponsePlanMetricsResponse(plan_score=95.0, coverage_score=0.9, average_eta_seconds=130.0),
        actions=[
            ResponsePlanActionResponse(
                resource=ResponsePlanResourceResponse(
                    resource_id="engine-1",
                    station_id="station-1",
                    station_name="Central Station",
                    origin=CoordinateResponse(latitude=32.0, longitude=35.0),
                ),
                target=ResponsePlanTargetResponse(
                    response_target_id=1, target_type="active_fire", priority_score=0.9, latitude=32.1, longitude=35.1
                ),
                route=ResponsePlanRouteResponse(
                    status=RouteStatus.REACHABLE,
                    eta_seconds=120.0,
                    distance_meters=800.0,
                    node_path=[1, 2],
                    path_coordinates=None,
                ),
            )
        ],
        uncovered_targets=[
            ResponsePlanTargetResponse(
                response_target_id=5, target_type="active_fire", priority_score=0.3, latitude=40.0, longitude=41.0
            )
        ],
        baseline_comparison=None,
        optimization_config=None,
        no_resources_during_planning=False,
    )
    values.update(overrides)
    return ResponsePlanDetailResponse(**values)


def make_service(
    *,
    latest_materialized: StoredGlobalPlanningRun | None = None,
    members_by_run_id: dict[int, tuple[StoredGlobalPlanningRunEvent, ...]] | None = None,
    details_by_plan_id: dict[int, ResponsePlanDetails] | None = None,
    response_by_plan_id: dict[int, ResponsePlanDetailResponse] | None = None,
):
    repository = FakeGlobalPlanningRunRepository(latest_materialized, members_by_run_id)
    details_service = FakeResponsePlanDetailsService(details_by_plan_id)
    presenter = FakeResponsePlanPresenter(response_by_plan_id)
    service = GlobalResponsePlanReadService(
        global_planning_run_repository=repository,
        response_plan_details_service=details_service,
        response_plan_presenter=presenter,
    )
    return service, {"run_repository": repository, "details_service": details_service, "presenter": presenter}


# ---------------------------------------------------------------------------
# No materialized generation
# ---------------------------------------------------------------------------


def test_no_materialized_generation_returns_null_plan():
    service, _ = make_service(latest_materialized=None)

    result = service.get_current(as_of=AS_OF)

    assert result.plan is None
    assert result.as_of == AS_OF


def test_get_members_is_not_called_when_no_materialized_generation():
    service, fakes = make_service(latest_materialized=None)

    service.get_current(as_of=AS_OF)

    assert fakes["run_repository"].get_members_calls == []


# ---------------------------------------------------------------------------
# Run metadata + metrics
# ---------------------------------------------------------------------------


def test_run_metadata_is_mapped():
    run = make_run(77, status=GlobalPlanningRunStatus.PARTIAL)
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: ()})

    result = service.get_current(as_of=AS_OF)

    assert result.plan.run_id == 77
    assert result.plan.started_at == STARTED_AT
    assert result.plan.completed_at == COMPLETED_AT
    assert result.plan.status is GlobalPlanningRunStatus.PARTIAL


def test_metrics_are_mapped_from_the_run():
    run = make_run(77, fitness_score=91.5, coverage_score=0.95, average_eta_seconds=180.0)
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: ()})

    result = service.get_current(as_of=AS_OF)

    assert result.plan.metrics.fitness_score == pytest.approx(91.5)
    assert result.plan.metrics.coverage_score == pytest.approx(0.95)
    assert result.plan.metrics.average_eta_seconds == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Optimization config
# ---------------------------------------------------------------------------


def test_optimization_config_is_mapped_when_random_seed_present():
    run = make_run(
        77, random_seed=42, ga_population_size=50, ga_generation_count=100, ga_mutation_rate=0.1, ga_crossover_rate=0.8
    )
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: ()})

    result = service.get_current(as_of=AS_OF)

    config = result.plan.optimization_config
    assert config is not None
    assert config.random_seed == 42
    assert config.population_size == 50
    assert config.generation_count == 100
    assert config.mutation_rate == pytest.approx(0.1)
    assert config.crossover_rate == pytest.approx(0.8)


def test_optimization_config_is_none_when_random_seed_absent():
    run = make_run(77, random_seed=None, ga_population_size=None, ga_generation_count=None, ga_mutation_rate=None, ga_crossover_rate=None)
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: ()})

    result = service.get_current(as_of=AS_OF)

    assert result.plan.optimization_config is None


# ---------------------------------------------------------------------------
# Shortage aggregation
# ---------------------------------------------------------------------------


def test_shortage_sums_across_all_members_including_non_materialized():
    run = make_run(77)
    materialized_member = make_member(
        1, 101, response_plan_id=501, minimum_resources=2, desired_resources=4, assigned_resources=3,
        required_slots_uncovered=0, desired_slots_uncovered=1,
    )
    no_op_member = make_member(
        2, 102, result_status=GlobalPlanningRunEventStatus.NO_OP, response_plan_id=None,
        minimum_resources=1, desired_resources=2, assigned_resources=1,
        required_slots_uncovered=0, desired_slots_uncovered=1,
    )
    service, _ = make_service(
        latest_materialized=run,
        members_by_run_id={77: (materialized_member, no_op_member)},
        details_by_plan_id={501: make_details(501, 101)},
        response_by_plan_id={501: make_presented(501, 101)},
    )

    result = service.get_current(as_of=AS_OF)

    shortage = result.plan.shortage
    assert shortage.total_required == 3
    assert shortage.total_desired == 6
    assert shortage.total_assigned == 4
    assert shortage.unmet_required == 0
    assert shortage.unmet_desired == 2


def test_shortage_skips_members_with_no_demand_snapshot():
    run = make_run(77)
    failed_member = make_member(
        1, 101, result_status=GlobalPlanningRunEventStatus.FAILED, response_plan_id=None,
        severity_level=None, severity_score=None,
        minimum_resources=None, desired_resources=None, assigned_resources=None,
        required_slots_uncovered=None, desired_slots_uncovered=None,
        coverage_score=None, average_eta_seconds=None,
    )
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: (failed_member,)})

    result = service.get_current(as_of=AS_OF)

    shortage = result.plan.shortage
    assert shortage.total_required == 0
    assert shortage.total_desired == 0
    assert shortage.total_assigned == 0
    assert shortage.unmet_required == 0
    assert shortage.unmet_desired == 0


def test_shortage_is_all_zero_when_there_are_no_members():
    run = make_run(77)
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: ()})

    result = service.get_current(as_of=AS_OF)

    shortage = result.plan.shortage
    assert (shortage.total_required, shortage.total_desired, shortage.total_assigned) == (0, 0, 0)
    assert (shortage.unmet_required, shortage.unmet_desired) == (0, 0)


# ---------------------------------------------------------------------------
# Events (materialized children only) + presenter reuse
# ---------------------------------------------------------------------------


def test_only_members_with_a_response_plan_id_become_events():
    run = make_run(77)
    materialized_member = make_member(1, 101, response_plan_id=501)
    no_op_member = make_member(2, 102, result_status=GlobalPlanningRunEventStatus.NO_OP, response_plan_id=None)
    service, _ = make_service(
        latest_materialized=run,
        members_by_run_id={77: (materialized_member, no_op_member)},
        details_by_plan_id={501: make_details(501, 101)},
        response_by_plan_id={501: make_presented(501, 101)},
    )

    result = service.get_current(as_of=AS_OF)

    assert [event.fire_event_id for event in result.plan.events] == [101]


def test_event_plan_carries_its_membership_snapshot_fields():
    run = make_run(77)
    member = make_member(
        1, 101, response_plan_id=501, severity_level=FireSeverityLevel.CRITICAL, severity_score=95.0,
        minimum_resources=3, desired_resources=5, assigned_resources=4, coverage_score=0.8, average_eta_seconds=200.0,
    )
    service, _ = make_service(
        latest_materialized=run,
        members_by_run_id={77: (member,)},
        details_by_plan_id={501: make_details(501, 101)},
        response_by_plan_id={501: make_presented(501, 101)},
    )

    result = service.get_current(as_of=AS_OF)

    (event,) = result.plan.events
    assert event.fire_event_id == 101
    assert event.response_plan_id == 501
    assert event.severity_level is FireSeverityLevel.CRITICAL
    assert event.severity_score == pytest.approx(95.0)
    assert event.minimum_resources == 3
    assert event.desired_resources == 5
    assert event.assigned_resources == 4
    assert event.coverage_score == pytest.approx(0.8)
    assert event.average_eta_seconds == pytest.approx(200.0)


def test_event_plan_reuses_presenter_actions_and_uncovered_targets_verbatim():
    run = make_run(77)
    member = make_member(1, 101, response_plan_id=501)
    presented = make_presented(501, 101)
    service, fakes = make_service(
        latest_materialized=run,
        members_by_run_id={77: (member,)},
        details_by_plan_id={501: make_details(501, 101)},
        response_by_plan_id={501: presented},
    )

    result = service.get_current(as_of=AS_OF)

    (event,) = result.plan.events
    assert event.actions == presented.actions
    assert event.uncovered_targets == presented.uncovered_targets
    assert fakes["presenter"].calls == [make_details(501, 101)]


def test_details_service_is_called_with_the_members_response_plan_id():
    run = make_run(77)
    member = make_member(1, 101, response_plan_id=501)
    service, fakes = make_service(
        latest_materialized=run,
        members_by_run_id={77: (member,)},
        details_by_plan_id={501: make_details(501, 101)},
        response_by_plan_id={501: make_presented(501, 101)},
    )

    service.get_current(as_of=AS_OF)

    assert fakes["details_service"].calls == [501]


def test_missing_child_plan_is_omitted_without_raising():
    run = make_run(77)
    member = make_member(1, 101, response_plan_id=999)
    service, _ = make_service(latest_materialized=run, members_by_run_id={77: (member,)}, details_by_plan_id={})

    result = service.get_current(as_of=AS_OF)

    assert result.plan.events == []


def test_presenter_is_not_called_for_a_missing_child_plan():
    run = make_run(77)
    member = make_member(1, 101, response_plan_id=999)
    service, fakes = make_service(latest_materialized=run, members_by_run_id={77: (member,)}, details_by_plan_id={})

    service.get_current(as_of=AS_OF)

    assert fakes["presenter"].calls == []


# ---------------------------------------------------------------------------
# get_members wiring
# ---------------------------------------------------------------------------


def test_get_members_is_called_with_the_materialized_runs_id():
    run = make_run(77)
    service, fakes = make_service(latest_materialized=run, members_by_run_id={77: ()})

    service.get_current(as_of=AS_OF)

    assert fakes["run_repository"].get_members_calls == [77]


# ---------------------------------------------------------------------------
# as_of handling
# ---------------------------------------------------------------------------


def test_explicit_as_of_is_used_verbatim():
    service, _ = make_service(latest_materialized=None)

    result = service.get_current(as_of=AS_OF)

    assert result.as_of == AS_OF


def test_as_of_defaults_to_current_time_when_omitted():
    service, _ = make_service(latest_materialized=None)

    before = datetime.now(timezone.utc)
    result = service.get_current()
    after = datetime.now(timezone.utc)

    assert before <= result.as_of <= after


def test_naive_as_of_is_rejected():
    service, _ = make_service(latest_materialized=None)

    with pytest.raises(ValueError):
        service.get_current(as_of=datetime(2026, 9, 20, 8, 0))


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_service_source_calls_no_write_operation():
    import inspect

    source = inspect.getsource(GlobalResponsePlanReadService)
    for forbidden in (".save(", ".update(", ".delete(", ".create_run(", "record_member_result", "complete_run"):
        assert forbidden not in source


def test_service_does_not_import_agent_calculator_or_optimization_modules():
    import ast
    from pathlib import Path

    forbidden_fragments = (
        "src.agents",
        "src.external",
        "src.simulation",
        "src.calculators",
        "GlobalPlanningOrchestrator",
        "genetic",
    )
    path = Path("src/services/global_planning/global_response_plan_read_service.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []
