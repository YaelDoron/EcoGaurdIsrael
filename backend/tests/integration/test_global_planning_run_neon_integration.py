"""Live integration test for GlobalPlanningRun persistence (Stage 2 of the
Global Multi-Incident Optimizer refactor) against Neon.

Focused on the ONE thing SQLite-based tests cannot verify at all: that
response_plans.global_planning_run_id's FOREIGN KEY constraint genuinely
exists and is enforced by a real PostgreSQL database (SQLite has no
ALTER TABLE ADD CONSTRAINT support - see
scripts/migrate_global_planning_run.py and tests/scripts/test_migrate_
global_planning_run.py for that documented limitation). Running the full
GlobalPlanningOrchestrator end-to-end (real Dijkstra/GA) is exercised via
SQLite-based tests instead (tests/services/global_planning/) - this file
does not repeat that, only the real-database-only persistence guarantees.
No DB mutation survives past this test's cleanup.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.config.settings import settings
from src.database.connection import get_engine, get_session_factory, init_db
from src.database.models.fire_event_db import FireEventDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models.fire_event_status import FireEventStatus
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.response_plan import ResponsePlan
from src.models.response_plan_status import ResponsePlanStatus
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.response_plan_repository import ResponsePlanRepository

pytestmark = pytest.mark.integration

METHODOLOGY_VERSION = "stage2-global-planning-run-integration"
AS_OF = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


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
                "DELETE FROM global_planning_run_events WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        run_ids = connection.execute(
            text(
                "SELECT global_planning_run_id FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version"
                ") AND global_planning_run_id IS NOT NULL"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        ).scalars().all()
        # response_plans must be deleted BEFORE the global_planning_runs
        # row(s) it references, or the FK constraint (Task 5's whole point)
        # blocks the delete.
        connection.execute(
            text(
                "DELETE FROM response_plans WHERE fire_event_id IN ("
                "SELECT id FROM fire_events WHERE methodology_version = :methodology_version)"
            ),
            {"methodology_version": METHODOLOGY_VERSION},
        )
        for run_id in run_ids:
            connection.execute(text("DELETE FROM global_planning_runs WHERE id = :run_id"), {"run_id": run_id})
        connection.execute(
            text(
                "DELETE FROM route_planning_runs WHERE fire_event_id IN ("
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


def _persist_fire_event_and_plan(fire_event_id_holder: dict) -> tuple[int, int]:
    session = get_session_factory()()
    event = FireEventDB(
        latitude=31.5,
        longitude=34.5,
        detected_at=AS_OF,
        updated_at=AS_OF,
        status=FireEventStatus.CONFIRMED.value,
        detection_confidence=0.9,
        methodology="STAGE2_IT_DETECTION",
        methodology_version=METHODOLOGY_VERSION,
    )
    session.add(event)
    session.flush()
    fire_event_id = event.id
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=AS_OF, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=AS_OF,
        methodology="m",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.commit()
    target_set_id, route_run_id = target_set.id, route_run.id
    session.close()

    response_plan_repository = ResponsePlanRepository()
    stored_plan = response_plan_repository.save(
        ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=target_set_id,
            route_planning_run_id=route_run_id,
            generated_at=AS_OF,
            status=ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS,
            methodology="m",
            methodology_version="1.0",
            random_seed=1,
            actions=(),
            uncovered_target_ids=(),
            plan_score=0.0,
            coverage_score=0.0,
            average_eta_seconds=None,
        )
    )
    fire_event_id_holder["id"] = fire_event_id
    return fire_event_id, stored_plan.id


def test_create_run_and_stamp_plan_round_trip_neon():
    fire_event_id_holder: dict = {}
    fire_event_id, plan_id = _persist_fire_event_and_plan(fire_event_id_holder)

    run_repository = GlobalPlanningRunRepository()
    stored_run = run_repository.create_run(
        started_at=AS_OF,
        trigger="manual",
        methodology="legacy_per_event_orchestration",
        methodology_version="1.0",
        input_fingerprint="f" * 64,
        fire_event_ids=(fire_event_id,),
    )
    run_repository.record_member_result(
        stored_run.id,
        fire_event_id,
        result_status=GlobalPlanningRunEventStatus.PLANNED,
        response_plan_id=plan_id,
        local_state_fingerprint=None,
        error_code=None,
    )
    completed = run_repository.complete_run(
        stored_run.id, status=GlobalPlanningRunStatus.COMPLETED, completed_at=AS_OF
    )

    response_plan_repository = ResponsePlanRepository()
    response_plan_repository.set_global_planning_run_id(plan_id, stored_run.id)

    assert completed.run.status is GlobalPlanningRunStatus.COMPLETED
    assert response_plan_repository.get_global_planning_run_id(plan_id) == stored_run.id
    members = run_repository.get_members(stored_run.id)
    assert [m.member.fire_event_id for m in members] == [fire_event_id]
    assert members[0].member.result_status is GlobalPlanningRunEventStatus.PLANNED


def test_response_plans_global_planning_run_id_fk_is_enforced_neon():
    """The ONE assertion SQLite cannot make: real PostgreSQL rejects a
    global_planning_run_id that does not reference a real run."""
    fire_event_id_holder: dict = {}
    _, plan_id = _persist_fire_event_and_plan(fire_event_id_holder)
    response_plan_repository = ResponsePlanRepository()

    with pytest.raises(IntegrityError):
        response_plan_repository.set_global_planning_run_id(plan_id, 999999999)

    assert response_plan_repository.get_global_planning_run_id(plan_id) is None
