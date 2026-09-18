"""Persistence layer for generated response-plan recommendations."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.database.connection import get_session_factory
from src.database.models.response_action_db import ResponseActionDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB
from src.models.response_action import ResponseAction
from src.models.response_plan import ResponsePlan
from src.models.response_plan_status import ResponsePlanStatus
from src.repositories.exceptions import ResponsePlanRepositoryError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredResponsePlan:
    """Persisted response plan plus database identity."""

    id: int
    plan: ResponsePlan


class ResponsePlanRepository:
    """Persists append-only response-plan recommendations via SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker[Session] | None = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save(self, plan: ResponsePlan) -> StoredResponsePlan:
        """Atomically persist a response plan with actions and uncovered targets."""
        with self._session_scope() as session:
            return self.save_in_session(session, plan)

    def save_in_session(self, session: Session, plan: ResponsePlan) -> StoredResponsePlan:
        """Same persistence as save, composed into a CALLER-OWNED session/
        transaction (Stage 6: GlobalResponsePlanActivationService persists
        several FireEvents' ResponsePlans, route runs, and commitments as
        one atomic activation). Does NOT commit or close the session."""
        if not isinstance(plan, ResponsePlan):
            raise ResponsePlanRepositoryError(f"plan must be a ResponsePlan, got {plan!r}.")

        db_plan = self._to_db_plan(plan)
        session.add(db_plan)
        try:
            session.flush()
            for action_order, action in enumerate(plan.actions):
                session.add(self._to_db_action(db_plan.id, action_order, action))
            for target_order, target_id in enumerate(plan.uncovered_target_ids):
                session.add(self._to_db_uncovered_target(db_plan.id, target_order, target_id))
            session.flush()
        except (IntegrityError, SQLAlchemyError) as exc:
            raise ResponsePlanRepositoryError("Response-plan persistence failed.") from exc

        logger.info(
            "Stored response plan %s for FireEvent %s with %s actions",
            db_plan.id,
            plan.fire_event_id,
            len(plan.actions),
        )
        return self._to_stored_plan(db_plan)

    def get_by_id(self, response_plan_id: int) -> StoredResponsePlan | None:
        self._validate_positive_int("response_plan_id", response_plan_id)
        with self._session_scope() as session:
            db_plan = self._get_db_plan(session, response_plan_id)
            return self._to_stored_plan(db_plan) if db_plan is not None else None

    def get_for_fire_event(self, fire_event_id: int) -> tuple[StoredResponsePlan, ...]:
        self._validate_positive_int("fire_event_id", fire_event_id)
        with self._session_scope() as session:
            db_plans = (
                session.execute(
                    select(ResponsePlanDB)
                    .options(
                        selectinload(ResponsePlanDB.actions),
                        selectinload(ResponsePlanDB.uncovered_targets),
                    )
                    .where(ResponsePlanDB.fire_event_id == fire_event_id)
                    .order_by(ResponsePlanDB.generated_at.desc(), ResponsePlanDB.id.desc())
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_plan(db_plan) for db_plan in db_plans)

    def get_latest_for_fire_event(self, fire_event_id: int) -> StoredResponsePlan | None:
        self._validate_positive_int("fire_event_id", fire_event_id)
        with self._session_scope() as session:
            db_plan = (
                session.execute(
                    select(ResponsePlanDB)
                    .options(
                        selectinload(ResponsePlanDB.actions),
                        selectinload(ResponsePlanDB.uncovered_targets),
                    )
                    .where(ResponsePlanDB.fire_event_id == fire_event_id)
                    .order_by(ResponsePlanDB.generated_at.desc(), ResponsePlanDB.id.desc())
                    .limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_plan(db_plan) if db_plan is not None else None

    def set_global_planning_run_id(self, response_plan_id: int, global_planning_run_id: int) -> None:
        """Stamp a NEWLY created ResponsePlan with the GlobalPlanningRun that
        created it (Stage 2, Task 11). A narrow, out-of-band UPDATE - not
        part of `save()`'s append-only write - because the existing per-
        event pipeline (routing -> optimization -> activation) that creates
        a plan has no knowledge of any surrounding global cycle (Task 10:
        Stage 2 must not be woven into that pipeline). GlobalPlanningOrchestrator
        calls this only when it has confirmed this plan is new (its id
        differs from whatever was the fire event's latest plan before this
        cycle's child refresh ran) - never for a NO_OP/baseline-only-
        recovery result reusing an already-existing plan.
        """
        self._validate_positive_int("response_plan_id", response_plan_id)
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        with self._session_scope() as session:
            db_plan = session.get(ResponsePlanDB, response_plan_id)
            if db_plan is None:
                raise ResponsePlanRepositoryError(f"ResponsePlan {response_plan_id!r} was not found.")
            db_plan.global_planning_run_id = global_planning_run_id

    def get_global_planning_run_id(self, response_plan_id: int) -> int | None:
        """Return which GlobalPlanningRun created this ResponsePlan, or None (Stage 2, Task 26)."""
        self._validate_positive_int("response_plan_id", response_plan_id)
        with self._session_scope() as session:
            db_plan = session.get(ResponsePlanDB, response_plan_id)
            if db_plan is None:
                raise ResponsePlanRepositoryError(f"ResponsePlan {response_plan_id!r} was not found.")
            return db_plan.global_planning_run_id

    def get_current_plan_resource_ids_for_fire_events(self, fire_event_ids: Iterable[int]) -> frozenset[str]:
        """Return the union of resource_ids used by each given FireEvent's CURRENT plan.

        "Current" matches CurrentResponsePlanResolver's own definition exactly
        (Epic 5, US 5.4, Task 8): the newest ResponsePlan (generated_at desc,
        id desc) for that FireEvent that has a linked
        ResponsePlanPlanningState sidecar. A plan without a sidecar is never
        considered current, even if it is the newest row for that FireEvent -
        the inner join below excludes it before ranking, exactly mirroring
        CurrentResponsePlanResolver.resolve()'s "skip sidecar-less plans
        entirely" behavior. A superseded older (sidecar-backed) plan is
        excluded by the ranking itself.

        Resolved in one batched query (a window function partitioned per
        fire_event_id) rather than one query per event - see Stage 0 of the
        Global Multi-Incident Optimizer refactor
        (src/services/resource_reservation/). FireEvent ids with no current
        plan, or whose current plan has no actions, simply contribute no
        resource ids.
        """
        ids = self._normalize_fire_event_ids(fire_event_ids)
        if not ids:
            return frozenset()

        with self._session_scope() as session:
            row_number = (
                func.row_number()
                .over(
                    partition_by=ResponsePlanDB.fire_event_id,
                    order_by=(ResponsePlanDB.generated_at.desc(), ResponsePlanDB.id.desc()),
                )
                .label("row_number")
            )
            ranked = (
                select(ResponsePlanDB.id.label("response_plan_id"), row_number)
                .join(
                    ResponsePlanPlanningStateDB,
                    ResponsePlanPlanningStateDB.response_plan_id == ResponsePlanDB.id,
                )
                .where(ResponsePlanDB.fire_event_id.in_(ids))
                .subquery()
            )
            resource_ids = (
                session.execute(
                    select(ResponseActionDB.resource_id)
                    .join(ranked, ranked.c.response_plan_id == ResponseActionDB.response_plan_id)
                    .where(ranked.c.row_number == 1)
                )
                .scalars()
                .all()
            )
            return frozenset(resource_ids)

    @classmethod
    def _to_db_plan(cls, plan: ResponsePlan) -> ResponsePlanDB:
        config = plan.optimization_config
        return ResponsePlanDB(
            fire_event_id=plan.fire_event_id,
            response_target_set_id=plan.response_target_set_id,
            route_planning_run_id=plan.route_planning_run_id,
            generated_at=plan.generated_at,
            status=plan.status.value,
            methodology=plan.methodology,
            methodology_version=plan.methodology_version,
            random_seed=plan.random_seed,
            plan_score=plan.plan_score,
            coverage_score=plan.coverage_score,
            average_eta_seconds=plan.average_eta_seconds,
            population_size=config.population_size if config is not None else None,
            generation_count=config.generation_count if config is not None else None,
            mutation_rate=config.mutation_rate if config is not None else None,
            crossover_rate=config.crossover_rate if config is not None else None,
            eta_reference_seconds=config.eta_reference_seconds if config is not None else None,
            initial_assignment_probability=config.initial_assignment_probability if config is not None else None,
            tournament_size=config.tournament_size if config is not None else None,
            elitism_count=config.elitism_count if config is not None else None,
        )

    @staticmethod
    def _to_db_action(response_plan_id: int, action_order: int, action: ResponseAction) -> ResponseActionDB:
        return ResponseActionDB(
            response_plan_id=response_plan_id,
            action_order=action_order,
            resource_id=str(action.resource_id),
            response_target_id=action.response_target_id,
            route_result_id=action.route_result_id,
        )

    @staticmethod
    def _to_db_uncovered_target(
        response_plan_id: int,
        target_order: int,
        response_target_id: int,
    ) -> ResponsePlanUncoveredTargetDB:
        return ResponsePlanUncoveredTargetDB(
            response_plan_id=response_plan_id,
            target_order=target_order,
            response_target_id=response_target_id,
        )

    @staticmethod
    def _get_db_plan(session: Session, response_plan_id: int) -> ResponsePlanDB | None:
        return (
            session.execute(
                select(ResponsePlanDB)
                .options(
                    selectinload(ResponsePlanDB.actions),
                    selectinload(ResponsePlanDB.uncovered_targets),
                )
                .where(ResponsePlanDB.id == response_plan_id)
            )
            .scalars()
            .one_or_none()
        )

    @classmethod
    def _to_stored_plan(cls, db_plan: ResponsePlanDB) -> StoredResponsePlan:
        actions = tuple(
            ResponseAction(
                resource_id=db_action.resource_id,
                response_target_id=db_action.response_target_id,
                route_result_id=db_action.route_result_id,
            )
            for db_action in sorted(db_plan.actions, key=lambda action: (action.action_order, action.id))
        )
        uncovered_target_ids = tuple(
            db_target.response_target_id
            for db_target in sorted(db_plan.uncovered_targets, key=lambda target: (target.target_order, target.id))
        )
        return StoredResponsePlan(
            id=db_plan.id,
            plan=ResponsePlan(
                fire_event_id=db_plan.fire_event_id,
                response_target_set_id=db_plan.response_target_set_id,
                route_planning_run_id=db_plan.route_planning_run_id,
                generated_at=cls._ensure_aware_datetime(db_plan.generated_at),
                status=ResponsePlanStatus(db_plan.status),
                methodology=db_plan.methodology,
                methodology_version=db_plan.methodology_version,
                random_seed=db_plan.random_seed,
                actions=actions,
                uncovered_target_ids=uncovered_target_ids,
                plan_score=db_plan.plan_score,
                coverage_score=db_plan.coverage_score,
                average_eta_seconds=db_plan.average_eta_seconds,
                optimization_config=cls._to_optimization_config(db_plan),
            ),
        )

    @staticmethod
    def _to_optimization_config(db_plan: ResponsePlanDB) -> ResponseOptimizationConfig | None:
        """Reconstruct the exact persisted GA config, or None for a legacy plan.

        All 8 columns are NULL together or present together (enforced by
        ck_response_plans_optimization_config_all_or_none), so checking one
        is sufficient. Never fabricates values from current source-code
        defaults for a legacy (all-NULL) row.
        """
        if db_plan.population_size is None:
            return None
        return ResponseOptimizationConfig(
            population_size=db_plan.population_size,
            generation_count=db_plan.generation_count,
            mutation_rate=db_plan.mutation_rate,
            crossover_rate=db_plan.crossover_rate,
            random_seed=db_plan.random_seed,
            eta_reference_seconds=db_plan.eta_reference_seconds,
            initial_assignment_probability=db_plan.initial_assignment_probability,
            tournament_size=db_plan.tournament_size,
            elitism_count=db_plan.elitism_count,
        )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResponsePlanRepositoryError(f"{field_name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _normalize_fire_event_ids(fire_event_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            values = tuple(fire_event_ids)
        except TypeError as exc:
            raise ResponsePlanRepositoryError("fire_event_ids must be iterable.") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResponsePlanRepositoryError(f"fire_event_ids must contain positive integer ids, got {value!r}.")
        return tuple(sorted(set(values)))

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
