"""Persistence layer for Task 3's optimized-vs-baseline `PlanComparison` history.

Implements User Story 5.3 Task 4: storage and traceability only. This
repository never computes or recomputes `score_difference`,
`improvement_percentage`, coverage, average ETA, or any other value on
`PlanComparison` -- Task 3's `BaselinePlanComparisonCalculator` already
did that. It only maps an already-built `PlanComparison` to/from
`PlanComparisonDB` rows and persists them append-only (see
`PlanComparisonDB` for why `save()` never updates an existing row).

`PlanComparisonRepositoryError` is defined here rather than added to the
shared `src/repositories/exceptions.py` registry, to keep this Task 4
change fully additive within Company 3's own new files and avoid touching
a file other teams' repositories also edit. Centralizing it there later
(one import line + one class, matching the existing `*RepositoryError`
pattern) is a safe follow-up, not required for Task 4 to work.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.calculators.baseline_plan.baseline_plan_comparison_calculator import PlanComparison
from src.database.connection import get_session_factory
from src.database.models.plan_comparison_db import PlanComparisonDB

logger = logging.getLogger(__name__)


class PlanComparisonRepositoryError(Exception):
    """Base exception for PlanComparison repository errors."""


@dataclass(frozen=True)
class StoredPlanComparison:
    """Persisted PlanComparison plus its database identity."""

    id: int
    comparison: PlanComparison


class PlanComparisonRepository:
    """Persists Task 3's `PlanComparison` domain objects via SQLAlchemy, append-only."""

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

    def save(self, comparison: PlanComparison) -> StoredPlanComparison:
        """Append `comparison` as a new, immutable history row.

        Always inserts a new row -- never updates or deduplicates an
        earlier comparison, even if it is identical. Every field is copied
        directly from `comparison`; nothing is recalculated here.
        """
        self._validate_comparison(comparison)

        with self._session_scope() as session:
            db_comparison = self._to_db(comparison)
            session.add(db_comparison)
            try:
                session.flush()
            except (IntegrityError, SQLAlchemyError) as exc:
                raise PlanComparisonRepositoryError("PlanComparison persistence failed.") from exc

            logger.info(
                "Stored PlanComparison %s for FireEvent %s "
                "(optimized_plan_id=%s, route_planning_run_id=%s, response_target_set_id=%s)",
                db_comparison.id,
                comparison.fire_event_id,
                comparison.optimized_plan_id,
                comparison.route_planning_run_id,
                comparison.response_target_set_id,
            )
            return self._to_domain(db_comparison)

    def get_by_id(self, comparison_id: int) -> StoredPlanComparison | None:
        """Return a persisted comparison by database id, or None if absent."""
        self._validate_comparison_id(comparison_id)
        with self._session_scope() as session:
            db_comparison = session.get(PlanComparisonDB, comparison_id)
            return self._to_domain(db_comparison) if db_comparison is not None else None

    def list_for_fire_event(self, fire_event_id: int) -> tuple[StoredPlanComparison, ...]:
        """Return all persisted comparisons for one FireEvent, oldest first."""
        self._validate_fire_event_id(fire_event_id)
        with self._session_scope() as session:
            rows = (
                session.execute(
                    select(PlanComparisonDB)
                    .where(PlanComparisonDB.fire_event_id == fire_event_id)
                    .order_by(PlanComparisonDB.id.asc())
                )
                .scalars()
                .all()
            )
            return tuple(self._to_domain(row) for row in rows)

    @staticmethod
    def _to_db(comparison: PlanComparison) -> PlanComparisonDB:
        return PlanComparisonDB(
            fire_event_id=comparison.fire_event_id,
            optimized_plan_id=comparison.optimized_plan_id,
            route_planning_run_id=comparison.route_planning_run_id,
            response_target_set_id=comparison.response_target_set_id,
            optimized_score=comparison.optimized_score,
            baseline_score=comparison.baseline_score,
            optimized_coverage_score=comparison.optimized_coverage_score,
            baseline_coverage_score=comparison.baseline_coverage_score,
            optimized_average_eta_seconds=comparison.optimized_average_eta_seconds,
            baseline_average_eta_seconds=comparison.baseline_average_eta_seconds,
            score_difference=comparison.score_difference,
            improvement_percentage=comparison.improvement_percentage,
        )

    @staticmethod
    def _to_domain(db_comparison: PlanComparisonDB) -> StoredPlanComparison:
        return StoredPlanComparison(
            id=db_comparison.id,
            comparison=PlanComparison(
                fire_event_id=db_comparison.fire_event_id,
                optimized_plan_id=db_comparison.optimized_plan_id,
                route_planning_run_id=db_comparison.route_planning_run_id,
                response_target_set_id=db_comparison.response_target_set_id,
                optimized_score=db_comparison.optimized_score,
                baseline_score=db_comparison.baseline_score,
                optimized_coverage_score=db_comparison.optimized_coverage_score,
                baseline_coverage_score=db_comparison.baseline_coverage_score,
                optimized_average_eta_seconds=db_comparison.optimized_average_eta_seconds,
                baseline_average_eta_seconds=db_comparison.baseline_average_eta_seconds,
                score_difference=db_comparison.score_difference,
                improvement_percentage=db_comparison.improvement_percentage,
            ),
        )

    @staticmethod
    def _validate_comparison(comparison: PlanComparison) -> None:
        if not isinstance(comparison, PlanComparison):
            raise PlanComparisonRepositoryError(f"comparison must be a PlanComparison, got {comparison!r}")

    @staticmethod
    def _validate_comparison_id(comparison_id: int) -> None:
        if isinstance(comparison_id, bool) or not isinstance(comparison_id, int) or comparison_id <= 0:
            raise PlanComparisonRepositoryError(
                f"comparison_id must be a positive integer, got {comparison_id!r}."
            )

    @staticmethod
    def _validate_fire_event_id(fire_event_id: int) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise PlanComparisonRepositoryError(
                f"fire_event_id must be a positive integer, got {fire_event_id!r}."
            )
