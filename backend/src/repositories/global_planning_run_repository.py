"""Persistence layer for GlobalPlanningRun + GlobalPlanningRunEvent (Stage 2
of the Global Multi-Incident Optimizer refactor).

`create_run` persists the run row AND every membership row for its
snapshot atomically (one transaction): a run must never exist with a
partially-recorded membership set, since that membership IS the audit
trail of "which FireEvents were considered" (Task 3). Membership *result*
updates (`record_member_result`) and finalizing the run (`complete_run`)
are separate, later writes - by design, since they happen incrementally as
GlobalPlanningOrchestrator works through the snapshot member-by-member
(Task 14: one member's failure must not block recording the rest).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from src.database.connection import get_session_factory
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB
from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_incident_demand import GlobalIncidentDemand
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.global_resource_shortage import GlobalResourceShortage
from src.repositories.exceptions import GlobalPlanningRunRepositoryError


@dataclass(frozen=True)
class StoredGlobalPlanningRun:
    """Persisted GlobalPlanningRun plus database identity."""

    id: int
    run: GlobalPlanningRun


@dataclass(frozen=True)
class StoredGlobalPlanningRunEvent:
    """Persisted GlobalPlanningRunEvent plus database identity."""

    id: int
    member: GlobalPlanningRunEvent


class GlobalPlanningRunRepository:
    """Persists GlobalPlanningRun runs and their FireEvent membership rows."""

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

    def create_run(
        self,
        *,
        started_at: datetime,
        trigger: str,
        methodology: str,
        methodology_version: str,
        input_fingerprint: str | None,
        fire_event_ids: Iterable[int],
    ) -> StoredGlobalPlanningRun:
        """Atomically create one RUNNING run plus one membership row per fire_event_id.

        `fire_event_ids` order is preserved as-is into `event_order` -
        callers (GlobalPlanningOrchestrator) are responsible for having
        already applied Stage 2's deterministic ordering (Task 9).
        """
        self._validate_aware_datetime("started_at", started_at)
        self._validate_non_empty_string("trigger", trigger)
        self._validate_non_empty_string("methodology", methodology)
        self._validate_non_empty_string("methodology_version", methodology_version)
        if input_fingerprint is not None:
            self._validate_non_empty_string("input_fingerprint", input_fingerprint)
        ids = self._normalize_fire_event_ids(fire_event_ids)

        with self._session_scope() as session:
            db_run = GlobalPlanningRunDB(
                started_at=started_at,
                completed_at=None,
                status=GlobalPlanningRunStatus.RUNNING.value,
                trigger=trigger,
                methodology=methodology,
                methodology_version=methodology_version,
                input_fingerprint=input_fingerprint,
            )
            session.add(db_run)
            session.flush()
            for event_order, fire_event_id in enumerate(ids):
                session.add(
                    GlobalPlanningRunEventDB(
                        global_planning_run_id=db_run.id,
                        fire_event_id=fire_event_id,
                        event_order=event_order,
                        result_status=None,
                        response_plan_id=None,
                        local_state_fingerprint=None,
                        error_code=None,
                    )
                )
            session.flush()
            return self._to_stored_run(db_run)

    def record_member_result(
        self,
        global_planning_run_id: int,
        fire_event_id: int,
        *,
        result_status: GlobalPlanningRunEventStatus,
        response_plan_id: int | None,
        local_state_fingerprint: str | None,
        error_code: str | None,
        incident_demand: GlobalIncidentDemand | None = None,
        event_optimization_result: GlobalEventOptimizationResult | None = None,
    ) -> None:
        """Update one already-created membership row with its child outcome.

        `incident_demand`/`event_optimization_result` (Stage 6, demand/
        shortage historical persistence) are this FireEvent's exact
        severity/demand snapshot for THIS cycle - pass both when the cycle
        reached the GA for this event, leave both None (e.g. a NO_OP
        member, or an event this run never got to) to leave the snapshot
        columns NULL rather than fabricating or duplicating a value.
        """
        with self._session_scope() as session:
            self.record_member_result_in_session(
                session,
                global_planning_run_id,
                fire_event_id,
                result_status=result_status,
                response_plan_id=response_plan_id,
                local_state_fingerprint=local_state_fingerprint,
                error_code=error_code,
                incident_demand=incident_demand,
                event_optimization_result=event_optimization_result,
            )

    def record_member_result_in_session(
        self,
        session: Session,
        global_planning_run_id: int,
        fire_event_id: int,
        *,
        result_status: GlobalPlanningRunEventStatus,
        response_plan_id: int | None,
        local_state_fingerprint: str | None,
        error_code: str | None,
        incident_demand: GlobalIncidentDemand | None = None,
        event_optimization_result: GlobalEventOptimizationResult | None = None,
    ) -> None:
        """Same update as record_member_result, composed into a CALLER-OWNED
        session/transaction (final-closure atomicity fix: GlobalResponsePlanActivationService
        writes this member result in the SAME transaction as its
        ResponsePlan/ResourceCommitment activation, so an authoritative
        generation can never commit without its required history also
        committing). Does NOT commit or close the session."""
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        self._validate_positive_int("fire_event_id", fire_event_id)
        if not isinstance(result_status, GlobalPlanningRunEventStatus):
            raise GlobalPlanningRunRepositoryError(
                f"result_status must be a GlobalPlanningRunEventStatus, got {result_status!r}"
            )
        if response_plan_id is not None:
            self._validate_positive_int("response_plan_id", response_plan_id)
        if local_state_fingerprint is not None:
            self._validate_non_empty_string("local_state_fingerprint", local_state_fingerprint)
        if error_code is not None:
            self._validate_non_empty_string("error_code", error_code)
        if incident_demand is not None and not isinstance(incident_demand, GlobalIncidentDemand):
            raise GlobalPlanningRunRepositoryError(
                f"incident_demand must be a GlobalIncidentDemand or None, got {incident_demand!r}"
            )
        if event_optimization_result is not None and not isinstance(
            event_optimization_result, GlobalEventOptimizationResult
        ):
            raise GlobalPlanningRunRepositoryError(
                "event_optimization_result must be a GlobalEventOptimizationResult or None, got "
                f"{event_optimization_result!r}"
            )
        if incident_demand is not None and incident_demand.fire_event_id != fire_event_id:
            raise GlobalPlanningRunRepositoryError(
                f"incident_demand.fire_event_id {incident_demand.fire_event_id!r} does not match "
                f"fire_event_id {fire_event_id!r}."
            )
        if event_optimization_result is not None and event_optimization_result.fire_event_id != fire_event_id:
            raise GlobalPlanningRunRepositoryError(
                f"event_optimization_result.fire_event_id {event_optimization_result.fire_event_id!r} does not "
                f"match fire_event_id {fire_event_id!r}."
            )

        db_member = session.execute(
            select(GlobalPlanningRunEventDB).where(
                GlobalPlanningRunEventDB.global_planning_run_id == global_planning_run_id,
                GlobalPlanningRunEventDB.fire_event_id == fire_event_id,
            )
        ).scalar_one_or_none()
        if db_member is None:
            raise GlobalPlanningRunRepositoryError(
                f"No membership row for GlobalPlanningRun {global_planning_run_id!r}, "
                f"FireEvent {fire_event_id!r}."
            )
        db_member.result_status = result_status.value
        db_member.response_plan_id = response_plan_id
        db_member.local_state_fingerprint = local_state_fingerprint
        db_member.error_code = error_code

        if incident_demand is not None:
            db_member.severity_assessment_id = incident_demand.severity_assessment_id
            db_member.severity_level = (
                incident_demand.severity_level.value if incident_demand.severity_level is not None else None
            )
            db_member.severity_score = incident_demand.severity_score
            db_member.demand_source = incident_demand.demand_source.value
            db_member.demand_policy_methodology = incident_demand.policy_methodology
            db_member.demand_policy_version = incident_demand.policy_version
            db_member.minimum_resources = incident_demand.minimum_resources
            db_member.desired_resources = incident_demand.desired_resources

        if event_optimization_result is not None:
            demand_result = event_optimization_result.demand_result
            db_member.assigned_resources = demand_result.suppression_resources_assigned
            db_member.required_slots_covered = demand_result.required_slots_covered
            db_member.required_slots_uncovered = demand_result.required_slots_uncovered
            db_member.desired_slots_covered = demand_result.desired_slots_covered
            db_member.desired_slots_uncovered = demand_result.desired_slots_uncovered
            db_member.predicted_risk_slots_covered = demand_result.predicted_risk_slots_covered
            db_member.locked_resources_preserved = demand_result.locked_resources_preserved
            db_member.coverage_score = event_optimization_result.coverage_score
            db_member.average_eta_seconds = event_optimization_result.average_eta_seconds

    def record_global_optimization_metadata(
        self,
        global_planning_run_id: int,
        *,
        raw_input_fingerprint: str,
        optimization_policy_fingerprint: str,
        random_seed: int,
        ga_population_size: int,
        ga_generation_count: int,
        ga_mutation_rate: float,
        ga_crossover_rate: float,
        demand_scoring_policy_methodology: str,
        demand_scoring_policy_version: str,
        severity_demand_policy_methodology: str,
        severity_demand_policy_version: str,
        stability_policy_methodology: str,
        stability_policy_version: str,
        fitness_score: float,
        coverage_score: float,
        average_eta_seconds: float | None,
        shortage: GlobalResourceShortage,
    ) -> None:
        """Stamp a run with its Global GA optimization metadata + shortage
        snapshot exactly once, after the GA has actually produced a result -
        a narrow, out-of-band UPDATE mirroring `set_input_fingerprint`'s own
        precedent, since none of this is known at `create_run` time."""
        with self._session_scope() as session:
            self.record_global_optimization_metadata_in_session(
                session,
                global_planning_run_id,
                raw_input_fingerprint=raw_input_fingerprint,
                optimization_policy_fingerprint=optimization_policy_fingerprint,
                random_seed=random_seed,
                ga_population_size=ga_population_size,
                ga_generation_count=ga_generation_count,
                ga_mutation_rate=ga_mutation_rate,
                ga_crossover_rate=ga_crossover_rate,
                demand_scoring_policy_methodology=demand_scoring_policy_methodology,
                demand_scoring_policy_version=demand_scoring_policy_version,
                severity_demand_policy_methodology=severity_demand_policy_methodology,
                severity_demand_policy_version=severity_demand_policy_version,
                stability_policy_methodology=stability_policy_methodology,
                stability_policy_version=stability_policy_version,
                fitness_score=fitness_score,
                coverage_score=coverage_score,
                average_eta_seconds=average_eta_seconds,
                shortage=shortage,
            )

    def record_global_optimization_metadata_in_session(
        self,
        session: Session,
        global_planning_run_id: int,
        *,
        raw_input_fingerprint: str,
        optimization_policy_fingerprint: str,
        random_seed: int,
        ga_population_size: int,
        ga_generation_count: int,
        ga_mutation_rate: float,
        ga_crossover_rate: float,
        demand_scoring_policy_methodology: str,
        demand_scoring_policy_version: str,
        severity_demand_policy_methodology: str,
        severity_demand_policy_version: str,
        stability_policy_methodology: str,
        stability_policy_version: str,
        fitness_score: float,
        coverage_score: float,
        average_eta_seconds: float | None,
        shortage: GlobalResourceShortage,
    ) -> None:
        """Same update as record_global_optimization_metadata, composed
        into a CALLER-OWNED session/transaction (final-closure atomicity
        fix - see record_member_result_in_session). Does NOT commit or
        close the session."""
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        self._validate_non_empty_string("raw_input_fingerprint", raw_input_fingerprint)
        self._validate_non_empty_string("optimization_policy_fingerprint", optimization_policy_fingerprint)
        if not isinstance(shortage, GlobalResourceShortage):
            raise GlobalPlanningRunRepositoryError(f"shortage must be a GlobalResourceShortage, got {shortage!r}")

        db_run = session.get(GlobalPlanningRunDB, global_planning_run_id)
        if db_run is None:
            raise GlobalPlanningRunRepositoryError(f"GlobalPlanningRun {global_planning_run_id!r} was not found.")
        db_run.raw_input_fingerprint = raw_input_fingerprint
        db_run.optimization_policy_fingerprint = optimization_policy_fingerprint
        db_run.random_seed = random_seed
        db_run.ga_population_size = ga_population_size
        db_run.ga_generation_count = ga_generation_count
        db_run.ga_mutation_rate = ga_mutation_rate
        db_run.ga_crossover_rate = ga_crossover_rate
        db_run.demand_scoring_policy_methodology = demand_scoring_policy_methodology
        db_run.demand_scoring_policy_version = demand_scoring_policy_version
        db_run.severity_demand_policy_methodology = severity_demand_policy_methodology
        db_run.severity_demand_policy_version = severity_demand_policy_version
        db_run.stability_policy_methodology = stability_policy_methodology
        db_run.stability_policy_version = stability_policy_version
        db_run.fitness_score = fitness_score
        db_run.coverage_score = coverage_score
        db_run.average_eta_seconds = average_eta_seconds
        db_run.shortage_total_required = shortage.total_required
        db_run.shortage_total_desired = shortage.total_desired
        db_run.shortage_total_assigned = shortage.total_assigned
        db_run.shortage_unmet_required = shortage.unmet_required
        db_run.shortage_unmet_desired = shortage.unmet_desired
        db_run.shortage_candidate_assignable_resource_count = shortage.candidate_assignable_resource_count
        db_run.shortage_committed_resource_count = shortage.committed_resource_count
        db_run.shortage_unavailable_resource_count = shortage.unavailable_resource_count
        db_run.shortage_locked_resources_preserved = shortage.locked_resources_preserved

    def complete_run(
        self,
        global_planning_run_id: int,
        *,
        status: GlobalPlanningRunStatus,
        completed_at: datetime,
    ) -> StoredGlobalPlanningRun:
        """Finalize a run with its aggregated final status."""
        with self._session_scope() as session:
            return self.complete_run_in_session(
                session, global_planning_run_id, status=status, completed_at=completed_at
            )

    def complete_run_in_session(
        self,
        session: Session,
        global_planning_run_id: int,
        *,
        status: GlobalPlanningRunStatus,
        completed_at: datetime,
    ) -> StoredGlobalPlanningRun:
        """Same finalization as complete_run, composed into a CALLER-OWNED
        session/transaction (final-closure atomicity fix - see
        record_member_result_in_session). Does NOT commit or close the
        session; the caller must session.flush() if it needs the returned
        StoredGlobalPlanningRun's values reflected before its own commit."""
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        if not isinstance(status, GlobalPlanningRunStatus) or status is GlobalPlanningRunStatus.RUNNING:
            raise GlobalPlanningRunRepositoryError(
                f"status must be a terminal GlobalPlanningRunStatus (not RUNNING), got {status!r}"
            )
        self._validate_aware_datetime("completed_at", completed_at)

        db_run = session.get(GlobalPlanningRunDB, global_planning_run_id)
        if db_run is None:
            raise GlobalPlanningRunRepositoryError(f"GlobalPlanningRun {global_planning_run_id!r} was not found.")
        db_run.status = status.value
        db_run.completed_at = completed_at
        session.flush()
        return self._to_stored_run(db_run)

    def set_input_fingerprint(self, global_planning_run_id: int, input_fingerprint: str) -> None:
        """Stamp a run with its (Stage 6: combined input+policy) fingerprint
        once it is known - a narrow, out-of-band UPDATE, mirroring
        ResponsePlanRepository.set_global_planning_run_id's own precedent,
        because `create_run` must persist the run/membership snapshot
        BEFORE GlobalPlanningInputBuilder can compute this value."""
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        self._validate_non_empty_string("input_fingerprint", input_fingerprint)
        with self._session_scope() as session:
            db_run = session.get(GlobalPlanningRunDB, global_planning_run_id)
            if db_run is None:
                raise GlobalPlanningRunRepositoryError(f"GlobalPlanningRun {global_planning_run_id!r} was not found.")
            db_run.input_fingerprint = input_fingerprint

    def get_by_id(self, global_planning_run_id: int) -> StoredGlobalPlanningRun | None:
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        with self._session_scope() as session:
            db_run = session.get(GlobalPlanningRunDB, global_planning_run_id)
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_latest(self) -> StoredGlobalPlanningRun | None:
        """Return the most recently started run, or None if none exist."""
        with self._session_scope() as session:
            db_run = (
                session.execute(
                    select(GlobalPlanningRunDB).order_by(
                        GlobalPlanningRunDB.started_at.desc(), GlobalPlanningRunDB.id.desc()
                    ).limit(1)
                )
                .scalars()
                .one_or_none()
            )
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_latest_activated(self, *, exclude_run_id: int | None = None) -> StoredGlobalPlanningRun | None:
        """Return the most recent run whose final status is COMPLETED or
        PARTIAL (Stage 6: "the latest successfully activated generation"
        for GlobalPlanningRefreshCoordinator's NO_OP check) - excludes a
        given run id (the CURRENT in-flight cycle, still RUNNING, would
        otherwise never match anyway, but exclusion is explicit rather than
        relying on that)."""
        if exclude_run_id is not None:
            self._validate_positive_int("exclude_run_id", exclude_run_id)
        with self._session_scope() as session:
            statement = (
                select(GlobalPlanningRunDB)
                .where(
                    GlobalPlanningRunDB.status.in_(
                        (GlobalPlanningRunStatus.COMPLETED.value, GlobalPlanningRunStatus.PARTIAL.value)
                    )
                )
                .order_by(GlobalPlanningRunDB.started_at.desc(), GlobalPlanningRunDB.id.desc())
                .limit(1)
            )
            if exclude_run_id is not None:
                statement = statement.where(GlobalPlanningRunDB.id != exclude_run_id)
            db_run = session.execute(statement).scalars().one_or_none()
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_latest_materialized_generation(self) -> StoredGlobalPlanningRun | None:
        """Return the most recent COMPLETED/PARTIAL run that actually
        materialized a ResponsePlan (Task B-BE-3).

        A NO_OP cycle's run is still finalized as COMPLETED even though no
        membership row ever got a response_plan_id - `get_latest_activated`
        alone would happily return that empty run. This adds an EXISTS
        filter requiring at least one membership row with a non-null
        response_plan_id, so UI callers wanting "the latest generation with
        an actual plan to show" skip NO_OP runs without needing to inspect
        membership rows themselves."""
        with self._session_scope() as session:
            has_materialized_member = (
                select(GlobalPlanningRunEventDB.id)
                .where(
                    GlobalPlanningRunEventDB.global_planning_run_id == GlobalPlanningRunDB.id,
                    GlobalPlanningRunEventDB.response_plan_id.is_not(None),
                )
                .exists()
            )
            statement = (
                select(GlobalPlanningRunDB)
                .where(
                    GlobalPlanningRunDB.status.in_(
                        (GlobalPlanningRunStatus.COMPLETED.value, GlobalPlanningRunStatus.PARTIAL.value)
                    ),
                    has_materialized_member,
                )
                .order_by(GlobalPlanningRunDB.completed_at.desc(), GlobalPlanningRunDB.id.desc())
                .limit(1)
            )
            db_run = session.execute(statement).scalars().one_or_none()
            return self._to_stored_run(db_run) if db_run is not None else None

    def get_members(self, global_planning_run_id: int) -> tuple[StoredGlobalPlanningRunEvent, ...]:
        """Return every membership row for a run, ordered by event_order (Task 9's snapshot order)."""
        self._validate_positive_int("global_planning_run_id", global_planning_run_id)
        with self._session_scope() as session:
            db_members = (
                session.execute(
                    select(GlobalPlanningRunEventDB)
                    .where(GlobalPlanningRunEventDB.global_planning_run_id == global_planning_run_id)
                    .order_by(GlobalPlanningRunEventDB.event_order)
                )
                .scalars()
                .all()
            )
            return tuple(self._to_stored_member(db_member) for db_member in db_members)

    @staticmethod
    def _to_stored_run(db_run: GlobalPlanningRunDB) -> StoredGlobalPlanningRun:
        return StoredGlobalPlanningRun(
            id=db_run.id,
            run=GlobalPlanningRun(
                started_at=GlobalPlanningRunRepository._ensure_aware_datetime(db_run.started_at),
                completed_at=(
                    GlobalPlanningRunRepository._ensure_aware_datetime(db_run.completed_at)
                    if db_run.completed_at is not None
                    else None
                ),
                status=GlobalPlanningRunStatus(db_run.status),
                trigger=db_run.trigger,
                methodology=db_run.methodology,
                methodology_version=db_run.methodology_version,
                input_fingerprint=db_run.input_fingerprint,
                raw_input_fingerprint=db_run.raw_input_fingerprint,
                optimization_policy_fingerprint=db_run.optimization_policy_fingerprint,
                random_seed=db_run.random_seed,
                ga_population_size=db_run.ga_population_size,
                ga_generation_count=db_run.ga_generation_count,
                ga_mutation_rate=db_run.ga_mutation_rate,
                ga_crossover_rate=db_run.ga_crossover_rate,
                demand_scoring_policy_methodology=db_run.demand_scoring_policy_methodology,
                demand_scoring_policy_version=db_run.demand_scoring_policy_version,
                severity_demand_policy_methodology=db_run.severity_demand_policy_methodology,
                severity_demand_policy_version=db_run.severity_demand_policy_version,
                stability_policy_methodology=db_run.stability_policy_methodology,
                stability_policy_version=db_run.stability_policy_version,
                fitness_score=db_run.fitness_score,
                coverage_score=db_run.coverage_score,
                average_eta_seconds=db_run.average_eta_seconds,
                shortage_total_required=db_run.shortage_total_required,
                shortage_total_desired=db_run.shortage_total_desired,
                shortage_total_assigned=db_run.shortage_total_assigned,
                shortage_unmet_required=db_run.shortage_unmet_required,
                shortage_unmet_desired=db_run.shortage_unmet_desired,
                shortage_candidate_assignable_resource_count=db_run.shortage_candidate_assignable_resource_count,
                shortage_committed_resource_count=db_run.shortage_committed_resource_count,
                shortage_unavailable_resource_count=db_run.shortage_unavailable_resource_count,
                shortage_locked_resources_preserved=db_run.shortage_locked_resources_preserved,
            ),
        )

    @staticmethod
    def _to_stored_member(db_member: GlobalPlanningRunEventDB) -> StoredGlobalPlanningRunEvent:
        return StoredGlobalPlanningRunEvent(
            id=db_member.id,
            member=GlobalPlanningRunEvent(
                fire_event_id=db_member.fire_event_id,
                event_order=db_member.event_order,
                result_status=(
                    GlobalPlanningRunEventStatus(db_member.result_status)
                    if db_member.result_status is not None
                    else None
                ),
                response_plan_id=db_member.response_plan_id,
                local_state_fingerprint=db_member.local_state_fingerprint,
                error_code=db_member.error_code,
                severity_assessment_id=db_member.severity_assessment_id,
                severity_level=(
                    FireSeverityLevel(db_member.severity_level) if db_member.severity_level is not None else None
                ),
                severity_score=db_member.severity_score,
                demand_source=(DemandSource(db_member.demand_source) if db_member.demand_source is not None else None),
                demand_policy_methodology=db_member.demand_policy_methodology,
                demand_policy_version=db_member.demand_policy_version,
                minimum_resources=db_member.minimum_resources,
                desired_resources=db_member.desired_resources,
                assigned_resources=db_member.assigned_resources,
                required_slots_covered=db_member.required_slots_covered,
                required_slots_uncovered=db_member.required_slots_uncovered,
                desired_slots_covered=db_member.desired_slots_covered,
                desired_slots_uncovered=db_member.desired_slots_uncovered,
                predicted_risk_slots_covered=db_member.predicted_risk_slots_covered,
                locked_resources_preserved=db_member.locked_resources_preserved,
                coverage_score=db_member.coverage_score,
                average_eta_seconds=db_member.average_eta_seconds,
            ),
        )

    @staticmethod
    def _normalize_fire_event_ids(fire_event_ids: Iterable[int]) -> tuple[int, ...]:
        try:
            values = tuple(fire_event_ids)
        except TypeError as exc:
            raise GlobalPlanningRunRepositoryError("fire_event_ids must be iterable.") from exc
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GlobalPlanningRunRepositoryError(
                    f"fire_event_ids must contain positive integer ids, got {value!r}."
                )
        if len(set(values)) != len(values):
            raise GlobalPlanningRunRepositoryError("fire_event_ids must not contain duplicates.")
        return values

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GlobalPlanningRunRepositoryError(f"{field_name} must be a positive integer, got {value!r}.")

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise GlobalPlanningRunRepositoryError(f"{field_name} must be a non-empty string, got {value!r}.")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise GlobalPlanningRunRepositoryError(f"{field_name} must be a timezone-aware datetime, got {value!r}.")

    @staticmethod
    def _ensure_aware_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
