"""Plan lifecycle over a realistic sequence, with the REAL FireEventRepository eligibility queries.

FireEvent rows live in SQLite; only the global-planning side is faked. This
proves the lifecycle is derived strictly from CONFIRMED (response-eligible)
events - SUSPECTED fires never count - and that reading it never triggers
planning/routing work.

    A, B SUSPECTED            -> none        0/0   (both monitoring-only)
    A CONFIRMED, no plan      -> generating  0/1   (B still SUSPECTED, not counted)
    plan for A saved          -> current     1/1
    B CONFIRMED               -> updating    1/2   pending = [B]
    plan covering A and B     -> current     2/2
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import update

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.database.models.fire_event_db import FireEventDB
from src.repositories.fire_event_repository import FireEventRepository
from src.services.global_planning.global_response_plan_read_service import GlobalResponsePlanReadService
from tests.services.global_planning.test_global_response_plan_read_service import (
    AS_OF,
    FakeResponsePlanDetailsService,
    FakeResponsePlanPresenter,
    make_details,
    make_member,
    make_presented,
    make_run,
)

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)


class ScriptedRunRepository:
    """Global-planning reads only (no create/record/complete methods exist here, so no planning can run)."""

    def __init__(self):
        self.materialized = None
        self.members = ()

    def get_latest_materialized_generation(self):
        return self.materialized

    def get_latest(self):
        return self.materialized

    def get_members(self, run_id):
        return self.members if self.materialized is not None and run_id == self.materialized.id else ()


def _insert_event(session_factory, status: str) -> int:
    with session_factory() as session:
        row = FireEventDB(
            latitude=31.7,
            longitude=35.1,
            detected_at=NOW,
            updated_at=NOW,
            status=status,
            detection_confidence=0.6,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
            created_at=NOW,
        )
        session.add(row)
        session.commit()
        return row.id


def _confirm(session_factory, fire_event_id: int) -> None:
    with session_factory() as session:
        session.execute(update(FireEventDB).where(FireEventDB.id == fire_event_id).values(status="confirmed"))
        session.commit()


def test_suspected_fires_never_count_and_the_lifecycle_follows_confirmations(sqlite_session_factory):
    runs = ScriptedRunRepository()
    plan_ids = (1, 2)
    service = GlobalResponsePlanReadService(
        global_planning_run_repository=runs,
        response_plan_details_service=FakeResponsePlanDetailsService({i: make_details(i, 100 + i) for i in plan_ids}),
        response_plan_presenter=FakeResponsePlanPresenter({i: make_presented(i, 100 + i) for i in plan_ids}),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
    )
    snapshot = lambda: service.get_current(as_of=AS_OF).coverage  # noqa: E731

    fire_a = _insert_event(sqlite_session_factory, "suspected")
    fire_b = _insert_event(sqlite_session_factory, "suspected")
    c = snapshot()
    assert (c.state, c.eligible_count, c.covered_count) == ("none", 0, 0)
    assert c.eligible_fire_event_ids == [] and c.pending_fire_event_ids == []
    assert c.monitoring_fire_event_ids == [fire_a, fire_b]

    _confirm(sqlite_session_factory, fire_a)
    c = snapshot()
    assert (c.state, c.eligible_count, c.covered_count) == ("generating", 1, 0)
    assert c.eligible_fire_event_ids == [fire_a] and c.pending_fire_event_ids == [fire_a]
    assert c.monitoring_fire_event_ids == [fire_b]  # still SUSPECTED, not part of 0/1

    runs.materialized = make_run(7)
    runs.members = (make_member(1, fire_a, response_plan_id=1),)
    c = snapshot()
    assert (c.state, c.eligible_count, c.covered_count) == ("current", 1, 1)
    assert c.covered_fire_event_ids == [fire_a]

    _confirm(sqlite_session_factory, fire_b)
    c = snapshot()
    assert (c.state, c.eligible_count, c.covered_count) == ("updating", 2, 1)
    assert c.covered_fire_event_ids == [fire_a] and c.pending_fire_event_ids == [fire_b]

    runs.materialized = make_run(8)
    runs.members = (
        make_member(1, fire_a, response_plan_id=1),
        make_member(2, fire_b, response_plan_id=2, event_order=1),
    )
    c = snapshot()
    assert (c.state, c.eligible_count, c.covered_count) == ("current", 2, 2)
    assert c.pending_fire_event_ids == []


def test_reading_the_lifecycle_never_starts_planning_or_routing():
    """The read service has no planning collaborators at all - only read repositories/services."""
    import inspect

    source = inspect.getsource(GlobalResponsePlanReadService)
    for forbidden in ("Orchestrator", "RoutePlanningAgent", "Dijkstra", "GlobalResponseOptimization", ".run(", "create_run"):
        assert forbidden not in source
