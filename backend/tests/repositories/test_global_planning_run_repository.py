"""Unit tests for GlobalPlanningRunRepository using SQLite in-memory (Stage 2
of the Global Multi-Incident Optimizer refactor)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database.models.fire_event_db import FireEventDB
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.repositories.exceptions import GlobalPlanningRunRepositoryError
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository

STARTED_AT = datetime(2026, 9, 20, 8, 0, tzinfo=timezone.utc)
COMPLETED_AT = STARTED_AT.replace(hour=8, minute=5)


@pytest.fixture
def repository(sqlite_session_factory) -> GlobalPlanningRunRepository:
    return GlobalPlanningRunRepository(sqlite_session_factory)


def _persist_fire_events(sqlite_session_factory, count: int) -> tuple[int, ...]:
    session = sqlite_session_factory()
    ids = []
    for _ in range(count):
        event = FireEventDB(
            latitude=32.7,
            longitude=35.0,
            detected_at=STARTED_AT,
            updated_at=STARTED_AT,
            status="confirmed",
            detection_confidence=0.9,
            methodology="m",
            methodology_version="1.0",
        )
        session.add(event)
        session.flush()
        ids.append(event.id)
    session.commit()
    session.close()
    return tuple(ids)


# ---------------------------------------------------------------------------
# create_run
# ---------------------------------------------------------------------------


def test_create_run_persists_run_and_membership_rows(repository, sqlite_session_factory):
    fire_event_ids = _persist_fire_events(sqlite_session_factory, 2)

    stored = repository.create_run(
        started_at=STARTED_AT,
        trigger="manual",
        methodology="legacy_per_event_orchestration",
        methodology_version="1.0",
        input_fingerprint="f" * 64,
        fire_event_ids=fire_event_ids,
    )

    assert stored.run.status is GlobalPlanningRunStatus.RUNNING
    assert stored.run.completed_at is None
    members = repository.get_members(stored.id)
    assert [m.member.fire_event_id for m in members] == list(fire_event_ids)
    assert [m.member.event_order for m in members] == [0, 1]
    assert all(m.member.result_status is None for m in members)


def test_create_run_with_no_fire_event_ids_persists_a_run_with_no_members(repository):
    stored = repository.create_run(
        started_at=STARTED_AT,
        trigger="manual",
        methodology="m",
        methodology_version="1.0",
        input_fingerprint=None,
        fire_event_ids=(),
    )

    assert repository.get_members(stored.id) == ()
    assert stored.run.input_fingerprint is None


def test_create_run_rejects_duplicate_fire_event_ids(repository, sqlite_session_factory):
    (fire_event_id,) = _persist_fire_events(sqlite_session_factory, 1)
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.create_run(
            started_at=STARTED_AT,
            trigger="manual",
            methodology="m",
            methodology_version="1.0",
            input_fingerprint=None,
            fire_event_ids=(fire_event_id, fire_event_id),
        )


# ---------------------------------------------------------------------------
# record_member_result
# ---------------------------------------------------------------------------


def test_record_member_result_updates_the_row(repository, sqlite_session_factory):
    fire_event_ids = _persist_fire_events(sqlite_session_factory, 1)
    stored = repository.create_run(
        started_at=STARTED_AT,
        trigger="manual",
        methodology="m",
        methodology_version="1.0",
        input_fingerprint=None,
        fire_event_ids=fire_event_ids,
    )

    repository.record_member_result(
        stored.id,
        fire_event_ids[0],
        result_status=GlobalPlanningRunEventStatus.PLANNED,
        response_plan_id=None,
        local_state_fingerprint="a" * 64,
        error_code=None,
    )

    (member,) = repository.get_members(stored.id)
    assert member.member.result_status is GlobalPlanningRunEventStatus.PLANNED
    assert member.member.local_state_fingerprint == "a" * 64


def test_record_member_result_raises_for_unknown_membership(repository):
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.record_member_result(
            999999,
            888888,
            result_status=GlobalPlanningRunEventStatus.FAILED,
            response_plan_id=None,
            local_state_fingerprint=None,
            error_code="child_planning_failed",
        )


# ---------------------------------------------------------------------------
# complete_run
# ---------------------------------------------------------------------------


def test_complete_run_sets_status_and_completed_at(repository):
    stored = repository.create_run(
        started_at=STARTED_AT,
        trigger="manual",
        methodology="m",
        methodology_version="1.0",
        input_fingerprint=None,
        fire_event_ids=(),
    )

    completed = repository.complete_run(stored.id, status=GlobalPlanningRunStatus.NO_ACTIVE_EVENTS, completed_at=COMPLETED_AT)

    assert completed.run.status is GlobalPlanningRunStatus.NO_ACTIVE_EVENTS
    assert completed.run.completed_at == COMPLETED_AT


def test_complete_run_rejects_running_as_a_terminal_status(repository):
    stored = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.complete_run(stored.id, status=GlobalPlanningRunStatus.RUNNING, completed_at=COMPLETED_AT)


def test_complete_run_raises_for_unknown_run(repository):
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.complete_run(999999, status=GlobalPlanningRunStatus.COMPLETED, completed_at=COMPLETED_AT)


# ---------------------------------------------------------------------------
# get_by_id / get_latest
# ---------------------------------------------------------------------------


def test_get_by_id_returns_none_when_absent(repository):
    assert repository.get_by_id(999999) is None


def test_get_latest_returns_the_most_recently_started_run(repository):
    first = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    second = repository.create_run(
        started_at=STARTED_AT.replace(minute=30), trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )

    latest = repository.get_latest()

    assert latest.id == second.id
    assert latest.id != first.id


def test_get_latest_returns_none_when_no_runs_exist(repository):
    assert repository.get_latest() is None


# ---------------------------------------------------------------------------
# Stage 6 - set_input_fingerprint / get_latest_activated
# ---------------------------------------------------------------------------


def test_set_input_fingerprint_updates_the_row(repository):
    stored = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    fingerprint = "a" * 64

    repository.set_input_fingerprint(stored.id, fingerprint)

    assert repository.get_by_id(stored.id).run.input_fingerprint == fingerprint


def test_set_input_fingerprint_raises_for_unknown_run(repository):
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.set_input_fingerprint(999, "a" * 64)


def test_get_latest_activated_ignores_running_and_failed_runs(repository):
    running = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    failed = repository.create_run(
        started_at=STARTED_AT.replace(minute=1), trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    repository.complete_run(failed.id, status=GlobalPlanningRunStatus.FAILED, completed_at=COMPLETED_AT)
    completed = repository.create_run(
        started_at=STARTED_AT.replace(minute=2), trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    repository.complete_run(completed.id, status=GlobalPlanningRunStatus.COMPLETED, completed_at=COMPLETED_AT)

    latest_activated = repository.get_latest_activated()

    assert latest_activated.id == completed.id


def test_get_latest_activated_includes_partial_status(repository):
    stored = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    repository.complete_run(stored.id, status=GlobalPlanningRunStatus.PARTIAL, completed_at=COMPLETED_AT)

    assert repository.get_latest_activated().id == stored.id


def test_get_latest_activated_excludes_given_run_id(repository):
    stored = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )
    repository.complete_run(stored.id, status=GlobalPlanningRunStatus.COMPLETED, completed_at=COMPLETED_AT)

    assert repository.get_latest_activated(exclude_run_id=stored.id) is None


def test_get_latest_activated_returns_none_when_no_runs_exist(repository):
    assert repository.get_latest_activated() is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_arguments_rejected(repository):
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.create_run(
            started_at=datetime(2026, 1, 1), trigger="manual", methodology="m", methodology_version="1.0",
            input_fingerprint=None, fire_event_ids=(),
        )
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.create_run(
            started_at=STARTED_AT, trigger="", methodology="m", methodology_version="1.0",
            input_fingerprint=None, fire_event_ids=(),
        )
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.get_by_id(0)
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.get_members(-1)


# ---------------------------------------------------------------------------
# get_recent (Task A6, Activity Feed)
# ---------------------------------------------------------------------------


def test_get_recent_empty_database_returns_empty_tuple(repository):
    assert repository.get_recent(10) == ()


def test_get_recent_orders_by_started_at_desc(repository, sqlite_session_factory):
    fire_event_ids = _persist_fire_events(sqlite_session_factory, 1)
    older = repository.create_run(
        started_at=STARTED_AT,
        trigger="manual",
        methodology="m",
        methodology_version="1.0",
        input_fingerprint=None,
        fire_event_ids=fire_event_ids,
    )
    newer = repository.create_run(
        started_at=STARTED_AT + timedelta(hours=1),
        trigger="manual",
        methodology="m",
        methodology_version="1.0",
        input_fingerprint=None,
        fire_event_ids=fire_event_ids,
    )

    recent = repository.get_recent(10)

    assert [stored.id for stored in recent] == [newer.id, older.id]


def test_get_recent_respects_limit(repository, sqlite_session_factory):
    fire_event_ids = _persist_fire_events(sqlite_session_factory, 1)
    for index in range(5):
        repository.create_run(
            started_at=STARTED_AT + timedelta(hours=index),
            trigger="manual",
            methodology="m",
            methodology_version="1.0",
            input_fingerprint=None,
            fire_event_ids=fire_event_ids,
        )

    recent = repository.get_recent(2)

    assert len(recent) == 2


def test_get_recent_rejects_invalid_limit(repository):
    with pytest.raises(GlobalPlanningRunRepositoryError):
        repository.get_recent(0)


# ---------------------------------------------------------------------------
# get_member_counts (Task A6, Activity Feed N+1 avoidance)
# ---------------------------------------------------------------------------


def test_get_member_counts_empty_input_returns_empty_dict(repository):
    assert repository.get_member_counts([]) == {}


def test_get_member_counts_batches_across_multiple_runs(repository, sqlite_session_factory):
    two_events = _persist_fire_events(sqlite_session_factory, 2)
    one_event = _persist_fire_events(sqlite_session_factory, 1)
    run_with_two = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=two_events,
    )
    run_with_one = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=one_event,
    )

    counts = repository.get_member_counts([run_with_two.id, run_with_one.id])

    assert counts == {run_with_two.id: 2, run_with_one.id: 1}


def test_get_member_counts_omits_runs_with_no_members(repository):
    run = repository.create_run(
        started_at=STARTED_AT, trigger="manual", methodology="m", methodology_version="1.0",
        input_fingerprint=None, fire_event_ids=(),
    )

    counts = repository.get_member_counts([run.id])

    assert counts == {}
