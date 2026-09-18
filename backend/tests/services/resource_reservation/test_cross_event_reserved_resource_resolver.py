"""Tests for CrossEventReservedResourceResolver (Global Multi-Incident Optimizer,
Stage 0 + Stage 1), using fakes only.

All fakes below expose ONLY read methods (no save/update/delete) - this is a
structural guarantee, not just an assertion, that resolving reserved
resources has zero persistence side effects, matching
CurrentResponsePlanResolver's own precedent (tests/services/response_planning/
test_current_response_plan_resolver.py).
"""
from __future__ import annotations

import pytest

from src.services.resource_reservation.cross_event_reserved_resource_resolver import (
    CrossEventReservedResourceResolver,
)

FIRE_EVENT_ID = 42


# ---------------------------------------------------------------------------
# Fakes (read-only by construction)
# ---------------------------------------------------------------------------


class FakeFireEventRepository:
    def __init__(self, active_fire_event_ids: tuple[int, ...] = ()):
        self.active_fire_event_ids = active_fire_event_ids
        self.calls = 0

    def get_active_fire_event_ids(self) -> tuple[int, ...]:
        self.calls += 1
        return self.active_fire_event_ids


class FakeResponsePlanRepository:
    def __init__(self, resource_ids: frozenset[str] = frozenset()):
        self.resource_ids = resource_ids
        self.calls: list[tuple[int, ...]] = []

    def get_current_plan_resource_ids_for_fire_events(self, fire_event_ids) -> frozenset[str]:
        self.calls.append(tuple(fire_event_ids))
        return self.resource_ids


class FakeResourceCommitmentRepository:
    def __init__(self, resource_ids: frozenset[str] = frozenset()):
        self.resource_ids = resource_ids
        self.calls: list[int] = []

    def get_resource_ids_for_other_active_events(self, excluded_fire_event_id: int) -> frozenset[str]:
        self.calls.append(excluded_fire_event_id)
        return self.resource_ids


def make_resolver(active_fire_event_ids, inferred_resource_ids=frozenset(), committed_resource_ids=frozenset()):
    fire_event_repository = FakeFireEventRepository(active_fire_event_ids)
    response_plan_repository = FakeResponsePlanRepository(inferred_resource_ids)
    resource_commitment_repository = FakeResourceCommitmentRepository(committed_resource_ids)
    resolver = CrossEventReservedResourceResolver(
        fire_event_repository=fire_event_repository,
        response_plan_repository=response_plan_repository,
        resource_commitment_repository=resource_commitment_repository,
    )
    return resolver, fire_event_repository, response_plan_repository, resource_commitment_repository


# ---------------------------------------------------------------------------
# Task 5 (Stage 0): self-exclusion (the FireEvent being planned is never treated as "other")
# ---------------------------------------------------------------------------


def test_excludes_the_fire_event_being_planned_from_the_query():
    resolver, _, response_plan_repository, resource_commitment_repository = make_resolver(
        (1, 2, 3), inferred_resource_ids=frozenset({"R1"})
    )

    resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert response_plan_repository.calls == [(2, 3)]
    assert resource_commitment_repository.calls == [1]


def test_only_active_fire_event_is_the_one_being_planned_returns_empty_without_querying_either_source():
    """Task 7/8: when the excluded FireEvent is the only active one, there are
    no "other" active events to check - both the commitment and inferred
    queries are skipped entirely."""
    resolver, _, response_plan_repository, resource_commitment_repository = make_resolver((1,))

    result = resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert result == frozenset()
    assert response_plan_repository.calls == []
    assert resource_commitment_repository.calls == []


def test_no_active_fire_events_at_all_returns_empty_without_querying_either_source():
    resolver, _, response_plan_repository, resource_commitment_repository = make_resolver(())

    result = resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert result == frozenset()
    assert response_plan_repository.calls == []
    assert resource_commitment_repository.calls == []


# ---------------------------------------------------------------------------
# Stage 1: union of explicit commitments and the Stage-0 inferred fallback
# ---------------------------------------------------------------------------


def test_returns_the_resource_ids_found_for_other_active_events():
    resolver, _, _, _ = make_resolver((1, 2), inferred_resource_ids=frozenset({"R1", "R2"}))

    result = resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert result == frozenset({"R1", "R2"})


def test_unions_explicit_commitments_with_stage0_inferred_resource_ids():
    resolver, _, _, _ = make_resolver(
        (1, 2), inferred_resource_ids=frozenset({"R1"}), committed_resource_ids=frozenset({"R2"})
    )

    result = resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert result == frozenset({"R1", "R2"})


def test_overlapping_ids_from_both_sources_are_deduplicated():
    resolver, _, _, _ = make_resolver(
        (1, 2), inferred_resource_ids=frozenset({"R1"}), committed_resource_ids=frozenset({"R1"})
    )

    result = resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert result == frozenset({"R1"})


def test_relies_on_fire_event_repository_for_active_status_filtering():
    """Task 7: the resolver does not re-derive SUSPECTED/CONFIRMED filtering
    itself - it trusts get_active_fire_event_ids() completely (that method
    already excludes RESOLVED/DISMISSED events, tested at the repository
    level - see test_get_active_events_excludes_resolved_and_dismissed)."""
    resolver, fire_event_repository, _, _ = make_resolver((1, 2))

    resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=1)

    assert fire_event_repository.calls == 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_fire_event_id", [0, -1, True, "42"])
def test_invalid_fire_event_id_rejected(invalid_fire_event_id):
    resolver, _, _, _ = make_resolver(())

    with pytest.raises(ValueError):
        resolver.get_resource_ids_reserved_by_other_active_plans(excluded_fire_event_id=invalid_fire_event_id)


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_resolver_source_calls_no_write_operation():
    import inspect

    source = inspect.getsource(CrossEventReservedResourceResolver)
    for forbidden in (".save(", ".update(", ".delete(", "update_event", "set_statuses"):
        assert forbidden not in source


def test_fakes_expose_no_write_methods():
    for forbidden in ("save", "update_event", "set_statuses", "update_status", "replace_commitments_for_plan"):
        assert not hasattr(FakeFireEventRepository(), forbidden)
        assert not hasattr(FakeResponsePlanRepository(), forbidden)
        assert not hasattr(FakeResourceCommitmentRepository(), forbidden)
