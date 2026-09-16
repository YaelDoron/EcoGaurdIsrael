"""Tests for BaselineComparisonCollaboratorAdapter, using fakes for every dependency (no real DB)."""
from __future__ import annotations

import pytest

from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.repositories.plan_comparison_repository import StoredPlanComparison
from src.services.response_planning.baseline_comparison_collaborator_adapter import (
    BaselineComparisonAdapterError,
    BaselineComparisonCollaboratorAdapter,
)

FIRE_EVENT_ID = 42
RESPONSE_PLAN_ID = 501


class FakeBaselineComparisonService:
    def __init__(self, comparison=None, exc=None):
        self.comparison = comparison
        self.exc = exc
        self.calls = []

    def compare(self, *, response_plan_id):
        self.calls.append(response_plan_id)
        if self.exc is not None:
            raise self.exc
        return self.comparison


class FakePlanComparisonRepository:
    def __init__(self, comparisons=()):
        self.comparisons = comparisons
        self.calls = []

    def list_for_fire_event(self, fire_event_id):
        self.calls.append(fire_event_id)
        return self.comparisons


def make_comparison(plan_id=RESPONSE_PLAN_ID, fire_event_id=FIRE_EVENT_ID) -> PlanComparison:
    return PlanComparison(
        fire_event_id=fire_event_id,
        optimized_plan_id=plan_id,
        route_planning_run_id=601,
        response_target_set_id=77,
        optimized_score=850.0,
        baseline_score=700.0,
        optimized_coverage_score=90.0,
        baseline_coverage_score=80.0,
        optimized_average_eta_seconds=300.0,
        baseline_average_eta_seconds=340.0,
        score_difference=150.0,
        improvement_percentage=21.4,
    )


def make_stored_comparison(comparison_id, plan_id=RESPONSE_PLAN_ID, fire_event_id=FIRE_EVENT_ID):
    return StoredPlanComparison(id=comparison_id, comparison=make_comparison(plan_id=plan_id, fire_event_id=fire_event_id))


def make_adapter(baseline_service=None, plan_comparison_repository=None) -> BaselineComparisonCollaboratorAdapter:
    return BaselineComparisonCollaboratorAdapter(
        baseline_comparison_service=baseline_service or FakeBaselineComparisonService(make_comparison()),
        plan_comparison_repository=plan_comparison_repository
        or FakePlanComparisonRepository((make_stored_comparison(701),)),
    )


# ---------------------------------------------------------------------------
# 20-21. Delegates once, returns persisted comparison id
# ---------------------------------------------------------------------------


def test_calls_baseline_service_with_exact_response_plan_id():
    baseline_service = FakeBaselineComparisonService(make_comparison())
    make_adapter(baseline_service=baseline_service).compare(response_plan_id=RESPONSE_PLAN_ID)

    assert baseline_service.calls == [RESPONSE_PLAN_ID]


def test_returns_stored_plan_comparison_with_persisted_id():
    plan_comparison_repository = FakePlanComparisonRepository((make_stored_comparison(701),))
    result = make_adapter(plan_comparison_repository=plan_comparison_repository).compare(
        response_plan_id=RESPONSE_PLAN_ID
    )

    assert isinstance(result, StoredPlanComparison)
    assert result.id == 701
    assert result.comparison.optimized_plan_id == RESPONSE_PLAN_ID


# ---------------------------------------------------------------------------
# 22-23. Deterministic selection among multiple historical rows
# ---------------------------------------------------------------------------


def test_multiple_historical_comparisons_do_not_cause_arbitrary_selection():
    plan_comparison_repository = FakePlanComparisonRepository(
        (
            make_stored_comparison(701, plan_id=999),  # different plan - must be excluded
            make_stored_comparison(702, plan_id=RESPONSE_PLAN_ID),
            make_stored_comparison(705, plan_id=RESPONSE_PLAN_ID),
            make_stored_comparison(703, plan_id=RESPONSE_PLAN_ID),
        )
    )

    result = make_adapter(plan_comparison_repository=plan_comparison_repository).compare(
        response_plan_id=RESPONSE_PLAN_ID
    )

    assert result.id == 705


def test_deterministically_selects_newest_persisted_row():
    """Repeated calls against the same fixed repository state pick the same row."""
    plan_comparison_repository = FakePlanComparisonRepository(
        (make_stored_comparison(701, plan_id=RESPONSE_PLAN_ID), make_stored_comparison(710, plan_id=RESPONSE_PLAN_ID))
    )
    adapter = make_adapter(plan_comparison_repository=plan_comparison_repository)

    first = adapter.compare(response_plan_id=RESPONSE_PLAN_ID)
    second = adapter.compare(response_plan_id=RESPONSE_PLAN_ID)

    assert first.id == 710
    assert second.id == 710


# ---------------------------------------------------------------------------
# 24. No recalculation
# ---------------------------------------------------------------------------


def test_does_not_recalculate_baseline_itself():
    baseline_service = FakeBaselineComparisonService(make_comparison())
    for forbidden in ("evaluate", "allocate", "compute_score"):
        assert not hasattr(baseline_service, forbidden)
    make_adapter(baseline_service=baseline_service).compare(response_plan_id=RESPONSE_PLAN_ID)
    assert len(baseline_service.calls) == 1


# ---------------------------------------------------------------------------
# 25. Failure propagation
# ---------------------------------------------------------------------------


def test_baseline_service_failure_propagates():
    baseline_service = FakeBaselineComparisonService(exc=RuntimeError("baseline exploded"))

    with pytest.raises(RuntimeError):
        make_adapter(baseline_service=baseline_service).compare(response_plan_id=RESPONSE_PLAN_ID)


def test_no_matching_comparison_after_success_fails_explicitly():
    baseline_service = FakeBaselineComparisonService(make_comparison(plan_id=RESPONSE_PLAN_ID))
    plan_comparison_repository = FakePlanComparisonRepository(())  # nothing recorded - internal inconsistency

    with pytest.raises(BaselineComparisonAdapterError):
        make_adapter(
            baseline_service=baseline_service, plan_comparison_repository=plan_comparison_repository
        ).compare(response_plan_id=RESPONSE_PLAN_ID)
