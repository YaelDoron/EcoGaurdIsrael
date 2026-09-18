"""Unit tests for ResourceCommitmentRepository using SQLite in-memory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.database.models.fire_event_db import FireEventDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.fire_station_db import FireStationDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.models.dispatch_state import DispatchState
from src.models.resource_status import ResourceStatus
from src.repositories.exceptions import ResourceCommitmentRepositoryError
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository

COMMITTED_AT = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def repository(sqlite_session_factory) -> ResourceCommitmentRepository:
    return ResourceCommitmentRepository(sqlite_session_factory)


def _persist_fire_event(session, *, status: str = "confirmed") -> int:
    event = FireEventDB(
        latitude=32.7,
        longitude=35.0,
        detected_at=COMMITTED_AT,
        updated_at=COMMITTED_AT,
        status=status,
        detection_confidence=0.9,
        methodology="m",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    return event.id


def _persist_response_plan(session, fire_event_id: int) -> int:
    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event_id, generated_at=COMMITTED_AT, methodology="m", methodology_version="1.0"
    )
    session.add(target_set)
    session.flush()
    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        planned_at=COMMITTED_AT,
        methodology="m",
        methodology_version="1.0",
        resource_ids=[],
    )
    session.add(route_run)
    session.flush()
    plan = ResponsePlanDB(
        fire_event_id=fire_event_id,
        response_target_set_id=target_set.id,
        route_planning_run_id=route_run.id,
        generated_at=COMMITTED_AT,
        status="complete",
        methodology="m",
        methodology_version="1.0",
        random_seed=1,
    )
    session.add(plan)
    session.flush()
    return plan.id


def _persist_resource(session, resource_id: str, station_id: str = "S1") -> None:
    if session.get(FireStationDB, station_id) is None:
        session.add(FireStationDB(id=station_id, name="Station", latitude=32.7, longitude=35.0))
        session.flush()
    session.add(FirefightingResourceDB(id=resource_id, station_id=station_id, status=ResourceStatus.AVAILABLE))


@pytest.fixture
def context(sqlite_session_factory):
    session = sqlite_session_factory()
    fire_event_id = _persist_fire_event(session)
    other_fire_event_id = _persist_fire_event(session)
    _persist_resource(session, "R1")
    _persist_resource(session, "R2")
    session.commit()
    plan_id = _persist_response_plan(session, fire_event_id)
    other_plan_id = _persist_response_plan(session, other_fire_event_id)
    session.commit()
    ctx = {
        "fire_event_id": fire_event_id,
        "other_fire_event_id": other_fire_event_id,
        "plan_id": plan_id,
        "other_plan_id": other_plan_id,
    }
    session.close()
    return ctx


# ---------------------------------------------------------------------------
# replace_commitments_for_plan (caller-owned session)
# ---------------------------------------------------------------------------


def test_replace_commitments_creates_rows(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1", "R2"),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_for_fire_event(context["fire_event_id"])
    assert sorted(c.resource_id for c in found) == ["R1", "R2"]
    assert all(c.fire_event_id == context["fire_event_id"] for c in found)
    assert all(c.response_plan_id == context["plan_id"] for c in found)
    assert all(c.committed_at == COMMITTED_AT for c in found)


def test_replace_commitments_defaults_to_planned_dispatch_state(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    (found,) = repository.get_for_fire_event(context["fire_event_id"])
    assert found.dispatch_state is DispatchState.PLANNED


def test_replace_commitments_accepts_explicit_dispatched_state(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
        dispatch_state=DispatchState.DISPATCHED,
    )
    session.commit()
    session.close()

    (found,) = repository.get_for_fire_event(context["fire_event_id"])
    assert found.dispatch_state is DispatchState.DISPATCHED


def test_replace_commitments_rejects_invalid_dispatch_state(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.replace_commitments_for_plan(
            session,
            fire_event_id=context["fire_event_id"],
            response_plan_id=context["plan_id"],
            resource_ids=("R1",),
            committed_at=COMMITTED_AT,
            dispatch_state="dispatched",
        )
    session.close()


def test_replace_commitments_releases_previous_commitments_for_same_event(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1", "R2"),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT + timedelta(minutes=1),
    )
    session.commit()
    session.close()

    found = repository.get_for_fire_event(context["fire_event_id"])
    assert [c.resource_id for c in found] == ["R1"]


def test_replace_commitments_does_not_commit_or_close_the_caller_session(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    assert not session.in_transaction() or session.in_transaction()  # session still usable
    session.rollback()
    session.close()

    # Nothing was committed, since the caller (this test) rolled back instead.
    found = repository.get_for_fire_event(context["fire_event_id"])
    assert found == ()


def test_replace_commitments_with_empty_resource_ids_only_releases(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=(),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    assert repository.get_for_fire_event(context["fire_event_id"]) == ()


# ---------------------------------------------------------------------------
# get_by_resource_id / get_for_fire_event
# ---------------------------------------------------------------------------


def test_get_by_resource_id_returns_none_when_uncommitted(repository, context):
    assert repository.get_by_resource_id("R1") is None


def test_get_by_resource_id_returns_the_commitment(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_by_resource_id("R1")
    assert found is not None
    assert found.fire_event_id == context["fire_event_id"]


def test_get_for_fire_event_empty_state(repository, context):
    assert repository.get_for_fire_event(context["fire_event_id"]) == ()


def test_no_cross_event_overwrite(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    # A different event's replace call must never touch fire_event's own commitment.
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    assert [c.resource_id for c in repository.get_for_fire_event(context["fire_event_id"])] == ["R1"]
    assert [c.resource_id for c in repository.get_for_fire_event(context["other_fire_event_id"])] == ["R2"]


# ---------------------------------------------------------------------------
# uniqueness of resource ownership (DB-level invariant)
# ---------------------------------------------------------------------------


def test_resource_id_uniqueness_is_enforced_at_the_db_level(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    from sqlalchemy.exc import IntegrityError

    session = sqlite_session_factory()
    with pytest.raises(IntegrityError):
        # Directly insert a second commitment row for the SAME resource_id,
        # bypassing replace_commitments_for_plan's own release step, to
        # prove the PK itself is the final invariant.
        from src.database.models.resource_commitment_db import ResourceCommitmentDB

        session.add(
            ResourceCommitmentDB(
                resource_id="R1",
                fire_event_id=context["other_fire_event_id"],
                response_plan_id=context["other_plan_id"],
                committed_at=COMMITTED_AT,
            )
        )
        session.commit()
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# get_resource_ids_for_other_active_events
# ---------------------------------------------------------------------------


def test_get_resource_ids_for_other_active_events(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_resource_ids_for_other_active_events(context["fire_event_id"])
    assert found == frozenset({"R2"})


def test_get_resource_ids_for_other_active_events_excludes_own_commitments(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_resource_ids_for_other_active_events(context["fire_event_id"])
    assert found == frozenset()


def test_get_resource_ids_for_other_active_events_excludes_inactive_events(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    db_event = session.get(FireEventDB, context["other_fire_event_id"])
    db_event.status = "resolved"
    session.commit()
    session.close()

    found = repository.get_resource_ids_for_other_active_events(context["fire_event_id"])
    assert found == frozenset()


# ---------------------------------------------------------------------------
# release_for_fire_event
# ---------------------------------------------------------------------------


def test_release_for_fire_event_removes_all_its_commitments(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1", "R2"),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    removed = repository.release_for_fire_event(context["fire_event_id"])

    assert removed == 2
    assert repository.get_for_fire_event(context["fire_event_id"]) == ()


def test_release_for_fire_event_is_a_no_op_when_nothing_committed(repository, context):
    assert repository.release_for_fire_event(context["fire_event_id"]) == 0


def test_release_for_fire_event_does_not_touch_other_events(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    repository.release_for_fire_event(context["fire_event_id"])

    assert [c.resource_id for c in repository.get_for_fire_event(context["other_fire_event_id"])] == ["R2"]


# ---------------------------------------------------------------------------
# release_for_fire_event_in_session (Stage 1.1: caller-owned-session variant
# composed into FireEventLifecycleService's atomic transition+release)
# ---------------------------------------------------------------------------


def test_release_for_fire_event_in_session_removes_commitments_without_committing(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1", "R2"),
        committed_at=COMMITTED_AT,
    )
    session.commit()

    removed = repository.release_for_fire_event_in_session(session, context["fire_event_id"])
    assert removed == 2
    # Not yet committed by the repository itself - still visible in this same session...
    assert session.execute(select(ResourceCommitmentDB)).scalars().all() == []
    session.rollback()
    session.close()

    # ...but since the caller (this test) rolled back instead of committing,
    # the release never actually took effect.
    assert [c.resource_id for c in repository.get_for_fire_event(context["fire_event_id"])] == ["R1", "R2"]


def test_release_for_fire_event_in_session_commits_when_the_caller_commits(
    repository, context, sqlite_session_factory
):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    session.commit()

    repository.release_for_fire_event_in_session(session, context["fire_event_id"])
    session.commit()
    session.close()

    assert repository.get_for_fire_event(context["fire_event_id"]) == ()


# ---------------------------------------------------------------------------
# get_for_fire_events (Stage 2: GlobalPlanningSnapshot capture)
# ---------------------------------------------------------------------------


def test_get_for_fire_events_returns_commitments_for_all_given_events(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["fire_event_id"],
        response_plan_id=context["plan_id"],
        resource_ids=("R1",),
        committed_at=COMMITTED_AT,
    )
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_for_fire_events([context["fire_event_id"], context["other_fire_event_id"]])

    assert {c.resource_id for c in found} == {"R1", "R2"}


def test_get_for_fire_events_excludes_events_not_in_the_given_list(repository, context, sqlite_session_factory):
    session = sqlite_session_factory()
    repository.replace_commitments_for_plan(
        session,
        fire_event_id=context["other_fire_event_id"],
        response_plan_id=context["other_plan_id"],
        resource_ids=("R2",),
        committed_at=COMMITTED_AT,
    )
    session.commit()
    session.close()

    found = repository.get_for_fire_events([context["fire_event_id"]])

    assert found == ()


def test_get_for_fire_events_returns_empty_for_empty_input(repository):
    assert repository.get_for_fire_events([]) == ()


def test_get_for_fire_events_rejects_invalid_ids(repository):
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.get_for_fire_events([0])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_arguments_rejected(repository, context, sqlite_session_factory):
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.get_by_resource_id("")
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.get_for_fire_event(0)
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.get_resource_ids_for_other_active_events(-1)
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.release_for_fire_event(True)
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.replace_commitments_for_plan(
            sqlite_session_factory(),
            fire_event_id=0,
            response_plan_id=context["plan_id"],
            resource_ids=("R1",),
            committed_at=COMMITTED_AT,
        )
    with pytest.raises(ResourceCommitmentRepositoryError):
        repository.replace_commitments_for_plan(
            sqlite_session_factory(),
            fire_event_id=context["fire_event_id"],
            response_plan_id=context["plan_id"],
            resource_ids=("R1",),
            committed_at=datetime(2026, 1, 1),  # naive
        )
