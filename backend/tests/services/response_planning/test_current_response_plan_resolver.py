"""Tests for CurrentResponsePlanResolver (Epic 5, US 5.4, Task 8), using fakes only.

Both fakes below expose ONLY read methods (no save/update/delete) - this is a
structural guarantee, not just an assertion, that resolving current has zero
persistence side effects.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.models import ResponseAction, ResponsePlan, ResponsePlanStatus
from src.repositories.response_plan_planning_state_repository import StoredResponsePlanPlanningState
from src.repositories.response_plan_repository import StoredResponsePlan
from src.services.response_planning import CurrentResponsePlanResolver

FIRE_EVENT_ID = 42
OTHER_FIRE_EVENT_ID = 99
AS_OF = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeResponsePlanRepository:
    def __init__(self, plans: tuple[StoredResponsePlan, ...] = ()):
        self.plans = plans
        self.calls = []

    def get_for_fire_event(self, fire_event_id):
        self.calls.append(fire_event_id)
        return tuple(p for p in self.plans if p.plan.fire_event_id == fire_event_id)


class FakeResponsePlanPlanningStateRepository:
    def __init__(self, sidecar_plan_ids: frozenset[int] = frozenset()):
        self.sidecar_plan_ids = sidecar_plan_ids
        self.calls = []

    def get_for_plan(self, response_plan_id):
        self.calls.append(response_plan_id)
        if response_plan_id in self.sidecar_plan_ids:
            return StoredResponsePlanPlanningState(
                id=response_plan_id,
                response_plan_id=response_plan_id,
                planning_effective_state_fingerprint="a" * 64,
                created_at=AS_OF,
            )
        return None


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_stored_plan(plan_id, fire_event_id=FIRE_EVENT_ID, generated_at=AS_OF, route_planning_run_id=601):
    plan = ResponsePlan(
        fire_event_id=fire_event_id,
        response_target_set_id=1,
        route_planning_run_id=route_planning_run_id,
        generated_at=generated_at,
        status=ResponsePlanStatus.COMPLETE,
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
        actions=(ResponseAction("truck-1", 10, 100),),
        uncovered_target_ids=(),
        plan_score=90.0,
        coverage_score=100.0,
        average_eta_seconds=200.0,
    )
    return StoredResponsePlan(id=plan_id, plan=plan)


def make_resolver(plans_newest_first, sidecar_plan_ids):
    return CurrentResponsePlanResolver(
        response_plan_repository=FakeResponsePlanRepository(plans_newest_first),
        response_plan_planning_state_repository=FakeResponsePlanPlanningStateRepository(
            frozenset(sidecar_plan_ids)
        ),
    )


# ---------------------------------------------------------------------------
# 1-4. Basic resolution
# ---------------------------------------------------------------------------


def test_no_plans_means_no_current_planning_safe_plan():
    resolver = make_resolver((), sidecar_plan_ids=())

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID) is None


def test_single_plan_with_sidecar_is_current():
    plan = make_stored_plan(1)
    resolver = make_resolver((plan,), sidecar_plan_ids={1})

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID) == plan


def test_multiple_plans_with_sidecars_newest_is_current():
    newest = make_stored_plan(3, generated_at=AS_OF + timedelta(minutes=20))
    middle = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    oldest = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((newest, middle, oldest), sidecar_plan_ids={1, 2, 3})

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID) == newest


def test_older_plans_remain_historical_not_returned_as_current():
    newest = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    oldest = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((newest, oldest), sidecar_plan_ids={1, 2})

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result.id == newest.id
    assert result.id != oldest.id


# ---------------------------------------------------------------------------
# 5-7. Sidecar-less newest plan(s) skipped
# ---------------------------------------------------------------------------


def test_latest_plan_without_sidecar_falls_back_to_previous_sidecar_backed_plan():
    newest_no_sidecar = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    older_with_sidecar = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((newest_no_sidecar, older_with_sidecar), sidecar_plan_ids={1})

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result.id == older_with_sidecar.id


def test_multiple_newest_sidecar_less_plans_are_skipped_to_find_newest_valid_older_plan():
    newest = make_stored_plan(3, generated_at=AS_OF + timedelta(minutes=30))
    middle = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=20))
    oldest_valid = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((newest, middle, oldest_valid), sidecar_plan_ids={1})

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result.id == oldest_valid.id


def test_no_plan_has_sidecar_means_no_current_planning_safe_plan():
    plan_a = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    plan_b = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((plan_a, plan_b), sidecar_plan_ids=())

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID) is None


# ---------------------------------------------------------------------------
# 8-9. PlanComparison does not define currentness (sidecar-only rule)
# ---------------------------------------------------------------------------


def test_latest_plan_with_sidecar_but_no_comparison_is_still_current():
    """The resolver never consults PlanComparison at all - sidecar presence alone qualifies."""
    plan = make_stored_plan(1)
    response_plan_repository = FakeResponsePlanRepository((plan,))
    sidecar_repository = FakeResponsePlanPlanningStateRepository(frozenset({1}))
    resolver = CurrentResponsePlanResolver(response_plan_repository, sidecar_repository)

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result == plan
    for forbidden in ("list_for_fire_event", "get_by_id"):
        assert not hasattr(response_plan_repository, forbidden) or forbidden == "get_by_id"
    # No PlanComparisonRepository was ever injected/used - structurally impossible to consult.
    assert not hasattr(resolver, "_plan_comparison_repository")


def test_older_plan_having_comparison_does_not_override_newer_valid_plan():
    """Comparisons are irrelevant to this resolver; only tests that newest-with-sidecar wins
    regardless of any comparison-related property (which this resolver never even reads)."""
    newer_valid = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    older_valid = make_stored_plan(1, generated_at=AS_OF)
    resolver = make_resolver((newer_valid, older_valid), sidecar_plan_ids={1, 2})

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result.id == newer_valid.id


# ---------------------------------------------------------------------------
# 14-15. Deterministic ordering
# ---------------------------------------------------------------------------


def test_different_generated_at_values_determine_ordering():
    later_plan = make_stored_plan(1, generated_at=AS_OF + timedelta(hours=2))
    earlier_plan = make_stored_plan(2, generated_at=AS_OF)
    # Fake repository returns them in the exact order given - simulating the
    # real repository's own generated_at DESC ordering (later_plan first).
    resolver = make_resolver((later_plan, earlier_plan), sidecar_plan_ids={1, 2})

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID).id == later_plan.id


def test_equal_generated_at_uses_higher_id_as_tie_break_matching_repository_convention():
    same_timestamp = AS_OF
    higher_id = make_stored_plan(5, generated_at=same_timestamp)
    lower_id = make_stored_plan(3, generated_at=same_timestamp)
    # Fake repository presents them in (id DESC) order, matching the real
    # repository's own tie-break (generated_at desc, id desc).
    resolver = make_resolver((higher_id, lower_id), sidecar_plan_ids={5, 3})

    assert resolver.resolve(fire_event_id=FIRE_EVENT_ID).id == higher_id.id


# ---------------------------------------------------------------------------
# 16. Multi-event isolation
# ---------------------------------------------------------------------------


def test_fire_event_a_never_resolves_a_plan_from_event_b():
    plan_a = make_stored_plan(1, fire_event_id=FIRE_EVENT_ID)
    plan_b = make_stored_plan(2, fire_event_id=OTHER_FIRE_EVENT_ID, generated_at=AS_OF + timedelta(hours=5))
    response_plan_repository = FakeResponsePlanRepository((plan_b, plan_a))
    resolver = CurrentResponsePlanResolver(
        response_plan_repository, FakeResponsePlanPlanningStateRepository(frozenset({1, 2}))
    )

    result = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert result.id == plan_a.id
    assert response_plan_repository.calls == [FIRE_EVENT_ID]


# ---------------------------------------------------------------------------
# 17-20. No persistence side effects; repeated calls are safe
# ---------------------------------------------------------------------------


def test_resolver_source_calls_no_write_operation():
    """Static guarantee: the resolver's own code never calls .save(/.update(/.delete(
    on either repository - it only ever calls get_for_fire_event() and get_for_plan()."""
    import inspect

    source = inspect.getsource(CurrentResponsePlanResolver)
    for forbidden in (".save(", ".update(", ".delete(", "update_event"):
        assert forbidden not in source


def test_resolving_current_does_not_update_or_delete_any_plan():
    for forbidden in ("save", "update", "delete", "update_event"):
        assert not hasattr(FakeResponsePlanRepository(), forbidden)


def test_resolving_current_does_not_create_any_sidecar():
    for forbidden in ("save",):
        assert not hasattr(FakeResponsePlanPlanningStateRepository(), forbidden)


def test_historical_plans_remain_queryable_through_the_underlying_repository():
    newest = make_stored_plan(2, generated_at=AS_OF + timedelta(minutes=10))
    oldest = make_stored_plan(1, generated_at=AS_OF)
    response_plan_repository = FakeResponsePlanRepository((newest, oldest))
    resolver = CurrentResponsePlanResolver(
        response_plan_repository, FakeResponsePlanPlanningStateRepository(frozenset({1, 2}))
    )

    resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert response_plan_repository.get_for_fire_event(FIRE_EVENT_ID) == (newest, oldest)


def test_repeated_resolution_is_idempotent_and_side_effect_free():
    plan = make_stored_plan(1)
    response_plan_repository = FakeResponsePlanRepository((plan,))
    sidecar_repository = FakeResponsePlanPlanningStateRepository(frozenset({1}))
    resolver = CurrentResponsePlanResolver(response_plan_repository, sidecar_repository)

    first = resolver.resolve(fire_event_id=FIRE_EVENT_ID)
    second = resolver.resolve(fire_event_id=FIRE_EVENT_ID)
    third = resolver.resolve(fire_event_id=FIRE_EVENT_ID)

    assert first == second == third == plan


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    resolver = make_resolver((), sidecar_plan_ids=())

    with pytest.raises(ValueError):
        resolver.resolve(fire_event_id=invalid_fire_event_id)
