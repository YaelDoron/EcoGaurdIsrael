"""Unit tests for ResponsePlanPlanningStateRepository using SQLite in-memory.

Importing ResponsePlanPlanningStateRepository (and its ORM module) here is
sufficient to register ResponsePlanPlanningStateDB on Base.metadata before
the shared `sqlite_engine` fixture runs `Base.metadata.create_all()` --
conftest.py does not need to be touched (same pattern as
tests/repositories/test_plan_comparison_repository.py).
"""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models.fire_event_db import FireEventDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.repositories.response_plan_planning_state_repository import (
    ResponsePlanPlanningStateRepository,
    ResponsePlanPlanningStateRepositoryError,
    StoredResponsePlanPlanningState,
)

VALID_FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64
GENERATED_AT = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def repository(sqlite_session_factory) -> ResponsePlanPlanningStateRepository:
    return ResponsePlanPlanningStateRepository(session_factory=sqlite_session_factory)


def make_response_plan(session_factory, **overrides) -> int:
    """Insert a minimal but FK-valid ResponsePlanDB row directly and return its id.

    ResponsePlanPlanningStateRepository never constructs a ResponsePlan
    itself (that belongs to Company 2's ResponseOptimizationAgent); tests
    only need a real row for the sidecar's FK to point at. Mirrors the
    `persisted_context` fixture in test_response_plan_repository.py.
    """
    session = session_factory()
    event = FireEventDB(
        latitude=32.731,
        longitude=35.046,
        detected_at=GENERATED_AT - timedelta(minutes=30),
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(event)
    session.flush()
    target_set = ResponseTargetSetDB(
        fire_event_id=event.id,
        generated_at=GENERATED_AT - timedelta(minutes=1),
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
    session.flush()

    defaults = dict(
        fire_event_id=event.id,
        response_target_set_id=target_set.id,
        route_planning_run_id=1,
        generated_at=GENERATED_AT,
        status="complete",
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
    )
    defaults.update(overrides)
    db_plan = ResponsePlanDB(**defaults)
    session.add(db_plan)
    session.commit()
    plan_id = db_plan.id
    session.close()
    return plan_id


# ---------------------------------------------------------------------------
# 1-2. Save and round-trip
# ---------------------------------------------------------------------------


def test_valid_response_plan_id_and_fingerprint_can_be_saved(repository, sqlite_session_factory):
    plan_id = make_response_plan(sqlite_session_factory)

    stored = repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)

    assert isinstance(stored, StoredResponsePlanPlanningState)
    assert isinstance(stored.id, int)
    assert stored.id > 0
    assert stored.response_plan_id == plan_id
    assert stored.planning_effective_state_fingerprint == VALID_FINGERPRINT


def test_saved_fingerprint_round_trips_through_get_for_plan(repository, sqlite_session_factory):
    plan_id = make_response_plan(sqlite_session_factory)
    repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)

    retrieved = repository.get_for_plan(plan_id)

    assert retrieved is not None
    assert retrieved.response_plan_id == plan_id
    assert retrieved.planning_effective_state_fingerprint == VALID_FINGERPRINT


# ---------------------------------------------------------------------------
# 3. Missing sidecar
# ---------------------------------------------------------------------------


def test_get_for_plan_returns_none_when_no_sidecar_exists(repository, sqlite_session_factory):
    plan_id = make_response_plan(sqlite_session_factory)

    assert repository.get_for_plan(plan_id) is None


# ---------------------------------------------------------------------------
# 4. Fingerprint preserved exactly
# ---------------------------------------------------------------------------


def test_fingerprint_is_preserved_exactly(repository, sqlite_session_factory):
    plan_id = make_response_plan(sqlite_session_factory)
    fingerprint = "0123456789abcdef" * 4
    assert len(fingerprint) == 64

    repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=fingerprint)
    retrieved = repository.get_for_plan(plan_id)

    assert retrieved.planning_effective_state_fingerprint == fingerprint


# ---------------------------------------------------------------------------
# 5. Invalid fingerprint format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_fingerprint",
    [
        "",
        "abc",
        "A" * 64,  # uppercase not allowed
        "g" * 64,  # non-hex character
        "a" * 63,  # too short
        "a" * 65,  # too long
        None,
        12345,
    ],
)
def test_invalid_fingerprint_format_rejected(repository, sqlite_session_factory, invalid_fingerprint):
    plan_id = make_response_plan(sqlite_session_factory)

    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=invalid_fingerprint)


# ---------------------------------------------------------------------------
# 6. 1:1 uniqueness
# ---------------------------------------------------------------------------


def test_second_sidecar_for_same_response_plan_id_is_rejected(repository, sqlite_session_factory):
    plan_id = make_response_plan(sqlite_session_factory)
    repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)

    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=OTHER_FINGERPRINT)


# ---------------------------------------------------------------------------
# 7-8. Multiple plans, historical rows untouched
# ---------------------------------------------------------------------------


def test_two_different_response_plans_may_have_distinct_fingerprints(repository, sqlite_session_factory):
    first_plan_id = make_response_plan(sqlite_session_factory)
    second_plan_id = make_response_plan(sqlite_session_factory)

    repository.save(response_plan_id=first_plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)
    repository.save(response_plan_id=second_plan_id, planning_effective_state_fingerprint=OTHER_FINGERPRINT)

    assert repository.get_for_plan(first_plan_id).planning_effective_state_fingerprint == VALID_FINGERPRINT
    assert repository.get_for_plan(second_plan_id).planning_effective_state_fingerprint == OTHER_FINGERPRINT


def test_historical_sidecar_rows_remain_unchanged_after_newer_plans_are_saved(repository, sqlite_session_factory):
    first_plan_id = make_response_plan(sqlite_session_factory)
    repository.save(response_plan_id=first_plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)

    for index in range(3):
        newer_plan_id = make_response_plan(sqlite_session_factory)
        repository.save(
            response_plan_id=newer_plan_id,
            planning_effective_state_fingerprint=f"{index:064d}",
        )

    assert repository.get_for_plan(first_plan_id).planning_effective_state_fingerprint == VALID_FINGERPRINT


# ---------------------------------------------------------------------------
# 9-12. No duplicated planning-chain data
# ---------------------------------------------------------------------------


def test_stored_result_does_not_duplicate_planning_chain_identifiers():
    field_names = {field.name for field in fields(StoredResponsePlanPlanningState)}

    assert field_names == {"id", "response_plan_id", "planning_effective_state_fingerprint", "created_at"}
    assert "fire_event_id" not in field_names
    assert "response_target_set_id" not in field_names
    assert "route_planning_run_id" not in field_names
    assert "resource_ids" not in field_names


def test_db_table_does_not_duplicate_planning_chain_identifiers():
    column_names = set(ResponsePlanPlanningStateDB.__table__.columns.keys())

    assert column_names == {"id", "response_plan_id", "planning_effective_state_fingerprint", "created_at"}
    assert "fire_event_id" not in column_names
    assert "response_target_set_id" not in column_names
    assert "route_planning_run_id" not in column_names
    assert "resource_ids" not in column_names


# ---------------------------------------------------------------------------
# 13. No update/overwrite method
# ---------------------------------------------------------------------------


def test_repository_has_no_update_overwrite_or_upsert_method(repository):
    for forbidden in ("update", "overwrite", "upsert"):
        assert not hasattr(repository, forbidden)


# ---------------------------------------------------------------------------
# 14. Schema creation
# ---------------------------------------------------------------------------


def test_fresh_test_db_schema_creates_the_new_table(sqlite_engine):
    from sqlalchemy import inspect

    table_names = set(inspect(sqlite_engine).get_table_names())

    assert "response_plan_planning_states" in table_names


# ---------------------------------------------------------------------------
# Foreign key enforcement (PRAGMA foreign_keys=ON in sqlite_engine fixture)
# ---------------------------------------------------------------------------


def test_nonexistent_response_plan_id_is_rejected_by_foreign_key(repository):
    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.save(response_plan_id=999999, planning_effective_state_fingerprint=VALID_FINGERPRINT)


# ---------------------------------------------------------------------------
# Transaction failure does not leave a partial record
# ---------------------------------------------------------------------------


def test_failed_save_does_not_leave_a_partial_row(repository, sqlite_session_factory, monkeypatch):
    plan_id = make_response_plan(sqlite_session_factory)

    def raise_integrity_error(self, *args, **kwargs):
        raise IntegrityError("forced failure", params=None, orig=Exception("forced"))

    monkeypatch.setattr(Session, "flush", raise_integrity_error)

    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.save(response_plan_id=plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)

    monkeypatch.undo()

    assert repository.get_for_plan(plan_id) is None


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_plan_id", [0, -1, True, "1", None])
def test_save_rejects_invalid_response_plan_id(repository, invalid_plan_id):
    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.save(response_plan_id=invalid_plan_id, planning_effective_state_fingerprint=VALID_FINGERPRINT)


@pytest.mark.parametrize("invalid_plan_id", [0, -1, True, "1", None])
def test_get_for_plan_rejects_invalid_response_plan_id(repository, invalid_plan_id):
    with pytest.raises(ResponsePlanPlanningStateRepositoryError):
        repository.get_for_plan(invalid_plan_id)
