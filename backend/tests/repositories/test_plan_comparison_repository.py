"""Unit tests for PlanComparisonRepository using SQLite in-memory.

Importing PlanComparisonRepository (and its ORM module) here is sufficient
to register PlanComparisonDB on Base.metadata before the shared
`sqlite_engine` fixture runs `Base.metadata.create_all()` -- conftest.py
does not need to be touched (same pattern as
tests/repositories/test_fire_spread_prediction_repository.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.database.models.fire_event_db import FireEventDB
from src.database.models.plan_comparison_db import PlanComparisonDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.repositories.plan_comparison_repository import (
    PlanComparisonRepository,
    PlanComparisonRepositoryError,
    StoredPlanComparison,
)

GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def persist_fk_prerequisites(session_factory, comparison: PlanComparison) -> None:
    """Persist a real row for each of the 4 traceability ids a PlanComparison
    carries (FND-05 gave plan_comparisons all four real FKs). These 4 ids are
    independent snapshot identities, not a validated chain, so each parent
    row is created standalone - only real enough to satisfy its own table's
    FKs (ResponsePlanDB needs fire_event_id/response_target_set_id/
    route_planning_run_id in turn, so those are reused rather than faked
    again). Idempotent and a no-op (no flush/commit) when every id already
    exists, so tests that monkeypatch Session.flush to simulate a write
    failure can pre-warm the chain beforehand without tripping the patch."""
    session = session_factory()
    added = False
    try:
        # Existence checks must not autoflush: a test may deliberately break
        # Session.flush to simulate a write failure, and a plain read should
        # never trigger that patched method when nothing is actually dirty.
        with session.no_autoflush:
            needs_fire_event = session.get(FireEventDB, comparison.fire_event_id) is None
            needs_target_set = session.get(ResponseTargetSetDB, comparison.response_target_set_id) is None
            needs_route_run = session.get(RoutePlanningRunDB, comparison.route_planning_run_id) is None
            needs_response_plan = session.get(ResponsePlanDB, comparison.optimized_plan_id) is None

        if needs_fire_event:
            session.add(
                FireEventDB(
                    id=comparison.fire_event_id,
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
                    id=comparison.response_target_set_id,
                    fire_event_id=comparison.fire_event_id,
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
                    id=comparison.route_planning_run_id,
                    fire_event_id=comparison.fire_event_id,
                    response_target_set_id=comparison.response_target_set_id,
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
                    id=comparison.optimized_plan_id,
                    fire_event_id=comparison.fire_event_id,
                    response_target_set_id=comparison.response_target_set_id,
                    route_planning_run_id=comparison.route_planning_run_id,
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
    """Wraps PlanComparisonRepository so existing tests can keep using
    arbitrary hand-picked traceability ids without each one separately
    pre-creating the FK chain plan_comparisons now requires (FND-05)."""

    def __init__(self, inner: PlanComparisonRepository, session_factory) -> None:
        self._inner = inner
        self._session_factory = session_factory

    def save(self, comparison):
        if isinstance(comparison, PlanComparison):
            persist_fk_prerequisites(self._session_factory, comparison)
        return self._inner.save(comparison)

    def get_by_id(self, comparison_id):
        return self._inner.get_by_id(comparison_id)

    def list_for_fire_event(self, fire_event_id):
        return self._inner.list_for_fire_event(fire_event_id)


@pytest.fixture
def repository(sqlite_session_factory) -> PlanComparisonRepository:
    return _FKProvisioningRepository(
        PlanComparisonRepository(session_factory=sqlite_session_factory), sqlite_session_factory
    )


def make_comparison(**overrides) -> PlanComparison:
    defaults = dict(
        fire_event_id=1,
        optimized_plan_id=1,
        route_planning_run_id=1,
        response_target_set_id=1,
        optimized_score=850.0,
        baseline_score=700.0,
        optimized_coverage_score=90.0,
        baseline_coverage_score=80.0,
        optimized_average_eta_seconds=300.0,
        baseline_average_eta_seconds=340.0,
        score_difference=150.0,
        improvement_percentage=((850.0 - 700.0) / 700.0) * 100,
    )
    defaults.update(overrides)
    return PlanComparison(**defaults)


# ---------------------------------------------------------------------------
# 1. Save
# ---------------------------------------------------------------------------


def test_save_persists_a_valid_comparison(repository):
    stored = repository.save(make_comparison())

    assert isinstance(stored, StoredPlanComparison)
    assert isinstance(stored.id, int)
    assert stored.id > 0


# ---------------------------------------------------------------------------
# 2. Field round-trip
# ---------------------------------------------------------------------------


def test_every_field_round_trips_through_save_and_get(repository):
    comparison = make_comparison(
        fire_event_id=11,
        optimized_plan_id=22,
        route_planning_run_id=33,
        response_target_set_id=44,
        optimized_score=853.5,
        baseline_score=701.25,
        optimized_coverage_score=91.1,
        baseline_coverage_score=79.4,
        optimized_average_eta_seconds=288.0,
        baseline_average_eta_seconds=352.0,
        score_difference=152.25,
        improvement_percentage=21.71183,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved is not None
    assert retrieved.comparison == comparison


# ---------------------------------------------------------------------------
# 3 & 4. None handling
# ---------------------------------------------------------------------------


def test_none_improvement_percentage_survives_persistence(repository):
    comparison = make_comparison(baseline_score=0.0, score_difference=850.0, improvement_percentage=None)

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.improvement_percentage is None


def test_none_average_eta_survives_persistence_on_either_side(repository):
    comparison = make_comparison(
        optimized_average_eta_seconds=None,
        baseline_average_eta_seconds=None,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.optimized_average_eta_seconds is None
    assert retrieved.comparison.baseline_average_eta_seconds is None


def test_one_sided_none_average_eta_survives_persistence(repository):
    comparison = make_comparison(optimized_average_eta_seconds=None, baseline_average_eta_seconds=340.0)

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.optimized_average_eta_seconds is None
    assert retrieved.comparison.baseline_average_eta_seconds == 340.0


# ---------------------------------------------------------------------------
# 5. Negative difference preserved exactly
# ---------------------------------------------------------------------------


def test_negative_score_difference_is_stored_exactly(repository):
    comparison = make_comparison(
        optimized_score=650.0,
        baseline_score=700.0,
        score_difference=-50.0,
        improvement_percentage=((650.0 - 700.0) / 700.0) * 100,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.score_difference == -50.0
    assert retrieved.comparison.improvement_percentage < 0


# ---------------------------------------------------------------------------
# 6. Zero scores
# ---------------------------------------------------------------------------


def test_zero_scores_persist_correctly(repository):
    comparison = make_comparison(
        optimized_score=0.0,
        baseline_score=0.0,
        optimized_coverage_score=0.0,
        baseline_coverage_score=0.0,
        optimized_average_eta_seconds=None,
        baseline_average_eta_seconds=None,
        score_difference=0.0,
        improvement_percentage=None,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.optimized_score == 0.0
    assert retrieved.comparison.baseline_score == 0.0
    assert retrieved.comparison.score_difference == 0.0
    assert retrieved.comparison.improvement_percentage is None


# ---------------------------------------------------------------------------
# 7 & 8. Append-only
# ---------------------------------------------------------------------------


def test_two_saves_create_two_distinct_records_and_first_is_unchanged(repository):
    first_comparison = make_comparison(optimized_plan_id=1)
    second_comparison = make_comparison(optimized_plan_id=2)

    first = repository.save(first_comparison)
    second = repository.save(second_comparison)

    assert first.id != second.id
    assert repository.get_by_id(first.id).comparison == first_comparison


def test_saving_the_same_comparison_twice_creates_two_rows(repository):
    comparison = make_comparison()

    first = repository.save(comparison)
    second = repository.save(comparison)

    assert first.id != second.id
    assert repository.get_by_id(first.id) is not None
    assert repository.get_by_id(second.id) is not None


# ---------------------------------------------------------------------------
# 9. Traceability ids preserved
# ---------------------------------------------------------------------------


def test_traceability_ids_round_trip_unchanged(repository):
    comparison = make_comparison(
        fire_event_id=101,
        optimized_plan_id=202,
        route_planning_run_id=303,
        response_target_set_id=404,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.fire_event_id == 101
    assert retrieved.comparison.optimized_plan_id == 202
    assert retrieved.comparison.route_planning_run_id == 303
    assert retrieved.comparison.response_target_set_id == 404


# ---------------------------------------------------------------------------
# 10 & 11. Retrieve by id / missing id
# ---------------------------------------------------------------------------


def test_retrieve_by_persistence_id(repository):
    stored = repository.save(make_comparison())

    retrieved = repository.get_by_id(stored.id)

    assert retrieved is not None
    assert retrieved.id == stored.id


def test_missing_id_returns_none_matching_existing_repository_convention(repository):
    assert repository.get_by_id(999999) is None


# ---------------------------------------------------------------------------
# 12. Transaction failure does not leave a partial record
# ---------------------------------------------------------------------------


def test_failed_save_does_not_leave_a_partial_comparison(repository, sqlite_session_factory, monkeypatch):
    comparison = make_comparison()
    # Pre-warm the FK chain (FND-05) before patching flush, so the failure
    # below is caused only by the real repository's own save, not by the
    # unrelated FK-prerequisite provisioning also needing a flush.
    persist_fk_prerequisites(sqlite_session_factory, comparison)

    def raise_integrity_error(self, *args, **kwargs):
        raise IntegrityError("forced failure", params=None, orig=Exception("forced"))

    monkeypatch.setattr(Session, "flush", raise_integrity_error)

    with pytest.raises(PlanComparisonRepositoryError):
        repository.save(comparison)

    monkeypatch.undo()

    session = sqlite_session_factory()
    rows = session.execute(select(PlanComparisonDB)).scalars().all()
    session.close()
    assert rows == []


# ---------------------------------------------------------------------------
# 13. Multi-event isolation
# ---------------------------------------------------------------------------


def test_fire_event_query_isolates_comparisons_by_fire_event(repository):
    repository.save(make_comparison(fire_event_id=1))
    repository.save(make_comparison(fire_event_id=1))
    repository.save(make_comparison(fire_event_id=2))

    event_1_results = repository.list_for_fire_event(1)
    event_2_results = repository.list_for_fire_event(2)

    assert len(event_1_results) == 2
    assert len(event_2_results) == 1
    assert all(result.comparison.fire_event_id == 1 for result in event_1_results)
    assert all(result.comparison.fire_event_id == 2 for result in event_2_results)


# ---------------------------------------------------------------------------
# 14. No recalculation -- exact non-round values preserved
# ---------------------------------------------------------------------------


def test_non_round_values_prove_persistence_is_pass_through_not_recalculation(repository):
    comparison = make_comparison(
        optimized_score=853.14159,
        baseline_score=701.2345,
        optimized_coverage_score=90.66667,
        baseline_coverage_score=80.11111,
        optimized_average_eta_seconds=301.333,
        baseline_average_eta_seconds=341.777,
        score_difference=853.14159 - 701.2345,
        improvement_percentage=((853.14159 - 701.2345) / 701.2345) * 100,
    )

    stored = repository.save(comparison)
    retrieved = repository.get_by_id(stored.id)

    assert retrieved.comparison.score_difference == 853.14159 - 701.2345
    assert retrieved.comparison.improvement_percentage == ((853.14159 - 701.2345) / 701.2345) * 100
    assert retrieved.comparison.optimized_score == 853.14159
    assert retrieved.comparison.baseline_score == 701.2345


# ---------------------------------------------------------------------------
# 15. Deterministic read order
# ---------------------------------------------------------------------------


def test_list_for_fire_event_returns_deterministic_save_order(repository):
    first = repository.save(make_comparison(fire_event_id=5, optimized_plan_id=1))
    second = repository.save(make_comparison(fire_event_id=5, optimized_plan_id=2))
    third = repository.save(make_comparison(fire_event_id=5, optimized_plan_id=3))

    results = repository.list_for_fire_event(5)

    assert [result.id for result in results] == [first.id, second.id, third.id]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_save_rejects_non_plan_comparison(repository):
    with pytest.raises(PlanComparisonRepositoryError):
        repository.save("not-a-comparison")


def test_get_by_id_rejects_non_positive_id(repository):
    with pytest.raises(PlanComparisonRepositoryError):
        repository.get_by_id(0)


def test_list_for_fire_event_rejects_non_positive_id(repository):
    with pytest.raises(PlanComparisonRepositoryError):
        repository.list_for_fire_event(0)
