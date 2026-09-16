"""Persistence layer linking a ResponsePlan to the PlanningEffectiveState
fingerprint that produced it (Epic 5, User Story 5.4, Task 4).

This repository never constructs a ResponsePlan, never invokes
ResponseOptimizationAgent, and never queries ResponsePlanRepository - it only
records/reads the append-only 1:1 link from an already-persisted
response_plan_id to the semantic fingerprint (Task 3's
`PlanningEffectiveState.fingerprint`) that produced it. Uniqueness of
response_plan_id and its existence are both enforced by the database
(unique constraint + foreign key on `ResponsePlanPlanningStateDB`), not by an
upstream lookup here.

`ResponsePlanPlanningStateRepositoryError` is defined here rather than added
to the shared `src/repositories/exceptions.py` registry, to keep this Task 4
change fully additive within US 5.4's own new files and avoid touching a file
other teams' repositories also edit - mirroring `PlanComparisonRepositoryError`.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.models.fire_spread_effective_state_fingerprint import validate_effective_state_fingerprint

logger = logging.getLogger(__name__)


class ResponsePlanPlanningStateRepositoryError(Exception):
    """Base exception for response-plan planning-state sidecar repository errors."""


@dataclass(frozen=True)
class StoredResponsePlanPlanningState:
    """Persisted ResponsePlan -> PlanningEffectiveState fingerprint link plus database identity."""

    id: int
    response_plan_id: int
    planning_effective_state_fingerprint: str
    created_at: datetime


class ResponsePlanPlanningStateRepository:
    """Persists the append-only 1:1 ResponsePlan -> PlanningEffectiveState fingerprint link."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        """Run a block of work in a session, committing on success and rolling back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save(
        self,
        *,
        response_plan_id: int,
        planning_effective_state_fingerprint: str,
    ) -> StoredResponsePlanPlanningState:
        """Link one ResponsePlan to the fingerprint that produced it.

        Always inserts a new row - never updates or overwrites an existing
        link. A second call for the same response_plan_id fails explicitly
        (unique-constraint violation), and a response_plan_id that does not
        exist fails explicitly (foreign-key violation) - both enforced by
        the database, not by an upstream lookup here.
        """
        self._validate_response_plan_id(response_plan_id)
        self._validate_fingerprint(planning_effective_state_fingerprint)

        with self._session_scope() as session:
            db_state = ResponsePlanPlanningStateDB(
                response_plan_id=response_plan_id,
                planning_effective_state_fingerprint=planning_effective_state_fingerprint,
            )
            session.add(db_state)
            try:
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise ResponsePlanPlanningStateRepositoryError(
                    "Response-plan planning-state persistence failed."
                ) from exc

            logger.info(
                "Linked ResponsePlan %s to planning-state fingerprint %s",
                response_plan_id,
                planning_effective_state_fingerprint,
            )
            return self._to_domain(db_state)

    def get_for_plan(self, response_plan_id: int) -> StoredResponsePlanPlanningState | None:
        """Return the fingerprint linked to one ResponsePlan, or None if none has been recorded."""
        self._validate_response_plan_id(response_plan_id)
        with self._session_scope() as session:
            db_state = (
                session.execute(
                    select(ResponsePlanPlanningStateDB).where(
                        ResponsePlanPlanningStateDB.response_plan_id == response_plan_id
                    )
                )
                .scalars()
                .one_or_none()
            )
            return self._to_domain(db_state) if db_state is not None else None

    @classmethod
    def _to_domain(cls, db_state: ResponsePlanPlanningStateDB) -> StoredResponsePlanPlanningState:
        return StoredResponsePlanPlanningState(
            id=db_state.id,
            response_plan_id=db_state.response_plan_id,
            planning_effective_state_fingerprint=db_state.planning_effective_state_fingerprint,
            created_at=cls._ensure_aware_datetime(db_state.created_at),
        )

    @staticmethod
    def _validate_response_plan_id(response_plan_id: object) -> None:
        if isinstance(response_plan_id, bool) or not isinstance(response_plan_id, int) or response_plan_id <= 0:
            raise ResponsePlanPlanningStateRepositoryError(
                f"response_plan_id must be a positive integer, got {response_plan_id!r}."
            )

    @staticmethod
    def _validate_fingerprint(value: object) -> None:
        if value is None:
            raise ResponsePlanPlanningStateRepositoryError("planning_effective_state_fingerprint is required.")
        try:
            validate_effective_state_fingerprint(value)
        except ValueError as exc:
            raise ResponsePlanPlanningStateRepositoryError(str(exc)) from exc

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
