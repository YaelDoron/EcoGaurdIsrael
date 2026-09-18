"""GlobalResponsePlanActivationService: atomic multi-event activation of one
GlobalOptimizationResult (Stage 6 of the Global Multi-Incident Optimizer
refactor, Tasks 26-29).

Mirrors ResponsePlanActivationService's proven pattern (Stage 1) - lock
rows, revalidate under lock, write everything, commit once - extended
across every FireEvent the global result touches, in ONE transaction:

    lock all participating FireEvent rows (fire_event_id order)
        -> lock every resource row the result touches (resource_id order)
        -> revalidate world state (stale-input check)
        -> per event: persist RoutePlanningRun+RouteResults (projected from
           the already-computed route facts - no Dijkstra rerun), then
           ResponsePlan+ResponseActions, then the ResponsePlanPlanningState
           sidecar that makes it current, then replace ResourceCommitments
           (DISPATCHED - see dispatch_state.py: activation IS the moment
           EcoGuard considers a resource to have left its station)
        -> commit everything together, or roll back everything together

If any event fails, or the world materially changed since GlobalPlanningInput
was built, NOTHING is written - GlobalPlanningStaleInput or
ResourceCommitmentConflict is raised and the previous authoritative
generation remains fully intact and current.

GlobalPlanningRun/GlobalPlanningRunEvent bookkeeping (final-closure
atomicity fix): when the caller passes `run_history` to activate(), the
required GlobalPlanningRunEvent member results, the run's optimization
metadata, and its final COMPLETED/PARTIAL status are written and committed
IN THIS SAME TRANSACTION as the ResponsePlan/ResourceCommitment writes -
never as separate, later, independently-committed calls. This closes a
crash window that previously existed: activate() commits (making a
generation authoritative) is a DIFFERENT transaction than the coordinator's
own later record_member_result/record_global_optimization_metadata/
complete_run calls, so a crash between the two could leave an authoritative
generation with a GlobalPlanningRun stuck at RUNNING and/or missing its
demand/shortage snapshot forever - a misleading "incomplete but successful"
audit history. `run_history` is optional (defaults to None, skipping this
write) purely so existing direct callers of activate() - unit/integration
tests exercising activation in isolation - are unaffected; the ONE
production caller (GlobalPlanningRefreshCoordinator) always passes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.calculators.global_response_optimization.global_response_optimization_config import (
    METHODOLOGY as GLOBAL_METHODOLOGY,
    METHODOLOGY_VERSION as GLOBAL_METHODOLOGY_VERSION,
)
from src.calculators.response_optimization.response_optimization_config import ResponseOptimizationConfig
from src.database.connection import get_session_factory
from src.database.models.fire_event_db import FireEventDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.models.dispatch_state import DispatchState
from src.models.fire_event_status import FireEventStatus
from src.models.global_optimization_result import GlobalOptimizationResult
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.response_action import ResponseAction
from src.models.response_plan import ResponsePlan
from src.models.response_plan_status import ResponsePlanStatus
from src.models.resource_status import ResourceStatus
from src.models.routing import RouteResult, RoutePlanningRun, RouteStatus
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.response_plan_planning_state_repository import ResponsePlanPlanningStateRepository
from src.repositories.response_plan_repository import ResponsePlanRepository
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.repositories.route_planning_repository import RoutePlanningRepository
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict

_ACTIVE_STATUSES = frozenset({FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED})


class GlobalPlanningStaleInput(Exception):
    """Raised when material world state changed since GlobalPlanningInput was
    built - a different active FireEvent set, or a resource whose
    commitment/operational state disagrees with what the input recorded.
    Nothing is written when this is raised (Task 29/30/32)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class GlobalActivationResult:
    """What one successful atomic activation produced."""

    global_planning_run_id: int
    response_plan_ids_by_event: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalRunHistoryContext:
    """Everything activate() needs, beyond global_planning_input/
    global_optimization_result, to ALSO persist this cycle's required
    GlobalPlanningRun/GlobalPlanningRunEvent history in the SAME
    transaction as its ResponsePlan/ResourceCommitment writes (final-closure
    atomicity fix - see this module's docstring). The rest of the required
    metadata (random_seed, GA config, fitness/coverage/eta, shortage,
    per-event demand snapshot) is already fully available on
    global_optimization_result/global_planning_input by the time activate()
    is called - only the combined fingerprint and the three policies' own
    identity are not."""

    combined_fingerprint: str
    optimization_policy_fingerprint: str
    demand_scoring_policy_methodology: str
    demand_scoring_policy_version: str
    severity_demand_policy_methodology: str
    severity_demand_policy_version: str
    stability_policy_methodology: str
    stability_policy_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "combined_fingerprint",
            "optimization_policy_fingerprint",
            "demand_scoring_policy_methodology",
            "demand_scoring_policy_version",
            "severity_demand_policy_methodology",
            "severity_demand_policy_version",
            "stability_policy_methodology",
            "stability_policy_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")


class GlobalResponsePlanActivationService:
    """Atomically activates one GlobalOptimizationResult across every FireEvent it touches."""

    def __init__(
        self,
        response_plan_repository: ResponsePlanRepository | None = None,
        response_plan_planning_state_repository: ResponsePlanPlanningStateRepository | None = None,
        route_planning_repository: RoutePlanningRepository | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        session_factory: sessionmaker[Session] | None = None,
    ) -> None:
        self._response_plan_repository = response_plan_repository or ResponsePlanRepository()
        self._response_plan_planning_state_repository = (
            response_plan_planning_state_repository or ResponsePlanPlanningStateRepository()
        )
        self._route_planning_repository = route_planning_repository or RoutePlanningRepository()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._session_factory = session_factory or get_session_factory()

    def activate(
        self,
        *,
        global_planning_input: GlobalPlanningInput,
        global_optimization_result: GlobalOptimizationResult,
        as_of: datetime,
        run_history: GlobalRunHistoryContext | None = None,
    ) -> GlobalActivationResult:
        self._validate_request(global_planning_input, global_optimization_result, as_of)
        if run_history is not None and not isinstance(run_history, GlobalRunHistoryContext):
            raise ValueError(f"run_history must be a GlobalRunHistoryContext or None, got {run_history!r}")

        fire_event_ids_sorted = tuple(sorted(global_planning_input.active_fire_event_ids))
        all_resource_ids = tuple(sorted({action.resource_id for action in global_optimization_result.actions}))

        session = self._session_factory()
        try:
            db_events_by_id = self._lock_fire_events(session, fire_event_ids_sorted)
            db_resources_by_id = self._lock_resources(session, all_resource_ids)

            self._validate_not_stale(
                session,
                global_planning_input,
                global_optimization_result,
                fire_event_ids_sorted,
                db_events_by_id,
                db_resources_by_id,
            )

            response_plan_ids_by_event: dict[int, int] = {}
            for fire_event_id in fire_event_ids_sorted:
                response_plan_id = self._activate_one_event(
                    session, global_planning_input, global_optimization_result, fire_event_id, as_of
                )
                if response_plan_id is not None:
                    response_plan_ids_by_event[fire_event_id] = response_plan_id

            if run_history is not None:
                self._record_run_history_in_session(
                    session, global_planning_input, global_optimization_result, fire_event_ids_sorted,
                    response_plan_ids_by_event, as_of, run_history,
                )

            session.commit()
            return GlobalActivationResult(
                global_planning_run_id=global_planning_input.global_planning_run_id,
                response_plan_ids_by_event=response_plan_ids_by_event,
            )
        except IntegrityError as exc:
            session.rollback()
            raise ResourceCommitmentConflict(
                fire_event_id=fire_event_ids_sorted[0] if fire_event_ids_sorted else 0,
                conflicts={resource_id: "concurrent_commitment_conflict" for resource_id in all_resource_ids},
            ) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- run history (final-closure atomicity fix) --------------------------

    def _record_run_history_in_session(
        self,
        session: Session,
        global_planning_input: GlobalPlanningInput,
        global_optimization_result: GlobalOptimizationResult,
        fire_event_ids_sorted: tuple[int, ...],
        response_plan_ids_by_event: dict[int, int],
        as_of: datetime,
        run_history: GlobalRunHistoryContext,
    ) -> None:
        """Write every GlobalPlanningRunEvent member result, the run's
        optimization metadata, and its final COMPLETED/PARTIAL status,
        using the SAME (not-yet-committed) session as the ResponsePlan/
        ResourceCommitment writes above - so all of it commits, or none of
        it does, together with them."""
        demand_by_event_id = {demand.fire_event_id: demand for demand in global_planning_input.incident_demands}
        event_result_by_event_id = {
            result.fire_event_id: result for result in global_optimization_result.event_results
        }

        all_planned = True
        for fire_event_id in fire_event_ids_sorted:
            response_plan_id = response_plan_ids_by_event.get(fire_event_id)
            if response_plan_id is None:
                all_planned = False
            self._global_planning_run_repository.record_member_result_in_session(
                session,
                global_planning_input.global_planning_run_id,
                fire_event_id,
                result_status=(
                    GlobalPlanningRunEventStatus.PLANNED
                    if response_plan_id is not None
                    else GlobalPlanningRunEventStatus.INSUFFICIENT_DATA
                ),
                response_plan_id=response_plan_id,
                local_state_fingerprint=run_history.combined_fingerprint,
                error_code=None,
                incident_demand=demand_by_event_id.get(fire_event_id),
                event_optimization_result=event_result_by_event_id.get(fire_event_id),
            )

        self._global_planning_run_repository.record_global_optimization_metadata_in_session(
            session,
            global_planning_input.global_planning_run_id,
            raw_input_fingerprint=global_planning_input.input_fingerprint,
            optimization_policy_fingerprint=run_history.optimization_policy_fingerprint,
            random_seed=global_optimization_result.random_seed,
            ga_population_size=global_optimization_result.config.population_size,
            ga_generation_count=global_optimization_result.config.generation_count,
            ga_mutation_rate=global_optimization_result.config.mutation_rate,
            ga_crossover_rate=global_optimization_result.config.crossover_rate,
            demand_scoring_policy_methodology=run_history.demand_scoring_policy_methodology,
            demand_scoring_policy_version=run_history.demand_scoring_policy_version,
            severity_demand_policy_methodology=run_history.severity_demand_policy_methodology,
            severity_demand_policy_version=run_history.severity_demand_policy_version,
            stability_policy_methodology=run_history.stability_policy_methodology,
            stability_policy_version=run_history.stability_policy_version,
            fitness_score=global_optimization_result.fitness_score,
            coverage_score=global_optimization_result.coverage_score,
            average_eta_seconds=global_optimization_result.average_eta_seconds,
            shortage=global_optimization_result.shortage,
        )

        final_status = GlobalPlanningRunStatus.COMPLETED if all_planned else GlobalPlanningRunStatus.PARTIAL
        self._global_planning_run_repository.complete_run_in_session(
            session, global_planning_input.global_planning_run_id, status=final_status, completed_at=as_of
        )

    # -- locking -----------------------------------------------------------

    @staticmethod
    def _lock_fire_events(session: Session, fire_event_ids: tuple[int, ...]) -> dict[int, FireEventDB]:
        if not fire_event_ids:
            return {}
        db_events = (
            session.execute(
                select(FireEventDB).where(FireEventDB.id.in_(fire_event_ids)).order_by(FireEventDB.id).with_for_update()
            )
            .scalars()
            .all()
        )
        return {db_event.id: db_event for db_event in db_events}

    @staticmethod
    def _lock_resources(session: Session, resource_ids: tuple[str, ...]) -> dict[str, FirefightingResourceDB]:
        if not resource_ids:
            return {}
        db_resources = (
            session.execute(
                select(FirefightingResourceDB)
                .where(FirefightingResourceDB.id.in_(resource_ids))
                .order_by(FirefightingResourceDB.id)
                .with_for_update()
            )
            .scalars()
            .all()
        )
        return {db_resource.id: db_resource for db_resource in db_resources}

    # -- stale-input revalidation (Task 29) ---------------------------------

    def _validate_not_stale(
        self,
        session: Session,
        global_planning_input: GlobalPlanningInput,
        global_optimization_result: GlobalOptimizationResult,
        fire_event_ids_sorted: tuple[int, ...],
        db_events_by_id: dict[int, FireEventDB],
        db_resources_by_id: dict[str, FirefightingResourceDB],
    ) -> None:
        # 1. Every event the input was built for must still exist and be active.
        for fire_event_id in fire_event_ids_sorted:
            db_event = db_events_by_id.get(fire_event_id)
            if db_event is None or FireEventStatus(db_event.status) not in _ACTIVE_STATUSES:
                raise GlobalPlanningStaleInput(
                    f"FireEvent {fire_event_id!r} is no longer active/found; input is stale."
                )

        # 2. The currently-active FireEvent set must not have grown beyond
        # what the input covers - a new fire mid-optimization is exactly the
        # "must retry with a bigger input" scenario (Task 30).
        currently_active_ids = frozenset(
            row[0]
            for row in session.execute(
                select(FireEventDB.id).where(FireEventDB.status.in_(status.value for status in _ACTIVE_STATUSES))
            )
        )
        unexpected_new_events = currently_active_ids - set(fire_event_ids_sorted)
        if unexpected_new_events:
            raise GlobalPlanningStaleInput(
                f"Active FireEvent set grew to include {sorted(unexpected_new_events)!r} since input was built."
            )

        # 3. Every resource the result touches must still be assignable and
        # not currently committed to a DIFFERENT FireEvent than this
        # activation is about to write it to (mirrors ResponsePlanActivationService's
        # own _find_conflicts, generalized across every event in this cycle).
        # Compared against the INTENDED new fire_event_id (from the result's
        # actions), not just the originally-recorded prior assignment - a
        # resource that was UNCOMMITTED at input-build time but got claimed
        # by a competing cycle before this activation runs must be caught
        # too (Task 52).
        resource_ids = tuple(sorted(db_resources_by_id))
        assignment_by_resource_id = {
            assignment.resource_id: assignment for assignment in global_planning_input.current_assignments
        }
        intended_fire_event_by_resource = {
            action.resource_id: action.fire_event_id for action in global_optimization_result.actions
        }
        existing_commitments = (
            session.execute(select(ResourceCommitmentDB).where(ResourceCommitmentDB.resource_id.in_(resource_ids)))
            .scalars()
            .all()
            if resource_ids
            else ()
        )
        commitment_by_resource_id = {commitment.resource_id: commitment for commitment in existing_commitments}
        for resource_id in resource_ids:
            db_resource = db_resources_by_id[resource_id]
            if db_resource.status is ResourceStatus.UNAVAILABLE:
                raise GlobalPlanningStaleInput(f"resource {resource_id!r} has gone UNAVAILABLE; input is stale.")

            existing_commitment = commitment_by_resource_id.get(resource_id)
            if existing_commitment is None:
                continue
            intended_fire_event_id = intended_fire_event_by_resource.get(resource_id)
            if intended_fire_event_id is not None and existing_commitment.fire_event_id != intended_fire_event_id:
                raise GlobalPlanningStaleInput(
                    f"resource {resource_id!r} is now committed to FireEvent {existing_commitment.fire_event_id!r}, "
                    f"but this activation would assign it to {intended_fire_event_id!r}; input is stale."
                )

            recorded_assignment = assignment_by_resource_id.get(resource_id)
            if (
                recorded_assignment is not None
                and existing_commitment.dispatch_state != recorded_assignment.dispatch_state.value
            ):
                raise GlobalPlanningStaleInput(
                    f"resource {resource_id!r}'s dispatch_state changed since the input was built; input is stale."
                )

    # -- per-event projection (Task 20/21) ----------------------------------

    def _activate_one_event(
        self,
        session: Session,
        global_planning_input: GlobalPlanningInput,
        global_optimization_result: GlobalOptimizationResult,
        fire_event_id: int,
        as_of: datetime,
    ) -> int | None:
        response_target_set_id = global_planning_input.event_target_set_ids.get(fire_event_id)
        if response_target_set_id is None:
            # No usable ResponseTargetSet exists for this event - nothing
            # truthful can be persisted (Task 1: never fabricate a target).
            return None

        event_actions = tuple(
            action for action in global_optimization_result.actions if action.fire_event_id == fire_event_id
        )
        event_result = next(
            (result for result in global_optimization_result.event_results if result.fire_event_id == fire_event_id),
            None,
        )

        route_run = self._build_route_planning_run(
            global_planning_input, fire_event_id, response_target_set_id, event_actions, as_of
        )
        stored_route_run = self._route_planning_repository.save_run_in_session(session, route_run)
        route_result_id_by_key = {
            (stored_route.route_result.resource_id, stored_route.route_result.response_target_id): stored_route.id
            for stored_route in stored_route_run.routes
        }

        plan_actions = tuple(
            ResponseAction(
                resource_id=action.resource_id,
                response_target_id=action.response_target_id,
                route_result_id=route_result_id_by_key[(action.resource_id, action.response_target_id)],
            )
            for action in event_actions
        )

        covered_target_ids = {action.response_target_id for action in plan_actions}
        uncovered_target_ids = tuple(
            sorted(
                {
                    int(slot_id.split("#", 1)[0])
                    for slot_id in (event_result.uncovered_slot_ids if event_result is not None else ())
                }
                - covered_target_ids
            )
        )

        if not plan_actions:
            status = ResponsePlanStatus.NO_FEASIBLE_ASSIGNMENTS
        elif uncovered_target_ids:
            status = ResponsePlanStatus.PARTIAL
        else:
            status = ResponsePlanStatus.COMPLETE

        optimization_config = ResponseOptimizationConfig(
            population_size=global_optimization_result.config.population_size,
            generation_count=global_optimization_result.config.generation_count,
            mutation_rate=global_optimization_result.config.mutation_rate,
            crossover_rate=global_optimization_result.config.crossover_rate,
            random_seed=global_optimization_result.random_seed,
            eta_reference_seconds=global_optimization_result.config.eta_reference_seconds,
            initial_assignment_probability=global_optimization_result.config.initial_assignment_probability,
            tournament_size=global_optimization_result.config.tournament_size,
            elitism_count=global_optimization_result.config.elitism_count,
        )

        plan = ResponsePlan(
            fire_event_id=fire_event_id,
            response_target_set_id=response_target_set_id,
            route_planning_run_id=stored_route_run.id,
            generated_at=as_of,
            status=status,
            methodology=global_optimization_result.optimization_methodology,
            methodology_version=global_optimization_result.optimization_methodology_version,
            random_seed=global_optimization_result.random_seed,
            actions=plan_actions,
            uncovered_target_ids=uncovered_target_ids,
            plan_score=global_optimization_result.fitness_score,
            coverage_score=event_result.coverage_score if event_result is not None else 0.0,
            average_eta_seconds=event_result.average_eta_seconds if event_result is not None else None,
            optimization_config=optimization_config,
        )
        stored_plan = self._response_plan_repository.save_in_session(session, plan)
        db_plan = session.get(ResponsePlanDB, stored_plan.id)
        db_plan.global_planning_run_id = global_planning_input.global_planning_run_id

        self._response_plan_planning_state_repository.save_in_session(
            session,
            response_plan_id=stored_plan.id,
            planning_effective_state_fingerprint=global_planning_input.input_fingerprint,
        )

        resource_ids_for_event = tuple(action.resource_id for action in event_actions)
        self._resource_commitment_repository.replace_commitments_for_plan(
            session,
            fire_event_id=fire_event_id,
            response_plan_id=stored_plan.id,
            resource_ids=resource_ids_for_event,
            committed_at=as_of,
            dispatch_state=DispatchState.DISPATCHED,
        )

        return stored_plan.id

    @staticmethod
    def _build_route_planning_run(
        global_planning_input: GlobalPlanningInput,
        fire_event_id: int,
        response_target_set_id: int,
        event_actions,
        as_of: datetime,
    ) -> RoutePlanningRun:
        event_target_ids = {
            target.response_target_id
            for target in global_planning_input.targets
            if target.fire_event_id == fire_event_id
        }
        event_resource_ids = tuple(
            sorted(
                resource.resource_id
                for resource in global_planning_input.resources
                if resource.is_assignable
            )
        )
        routes: list[RouteResult] = []
        seen: set[tuple[str, int]] = set()
        for action in event_actions:
            key = (action.resource_id, action.response_target_id)
            if key in seen:
                continue
            seen.add(key)
            node_path = tuple(action.node_path)
            routes.append(
                RouteResult(
                    resource_id=action.resource_id,
                    response_target_id=action.response_target_id,
                    status=RouteStatus.REACHABLE,
                    source_node_id=node_path[0] if node_path else None,
                    target_node_id=node_path[-1] if node_path else None,
                    node_path=node_path,
                    distance_meters=action.route_distance_meters,
                    travel_time_seconds=action.eta_seconds,
                )
            )
        resource_id_universe = tuple(sorted(set(event_resource_ids) | {resource_id for resource_id, _ in seen}))
        return RoutePlanningRun(
            fire_event_id=fire_event_id,
            response_target_set_id=response_target_set_id,
            planned_at=as_of,
            methodology=GLOBAL_METHODOLOGY,
            methodology_version=GLOBAL_METHODOLOGY_VERSION,
            resource_ids=resource_id_universe,
            routes=tuple(routes),
        )

    @staticmethod
    def _validate_request(
        global_planning_input: object, global_optimization_result: object, as_of: object
    ) -> None:
        if not isinstance(global_planning_input, GlobalPlanningInput):
            raise ValueError(f"global_planning_input must be a GlobalPlanningInput, got {global_planning_input!r}")
        if not isinstance(global_optimization_result, GlobalOptimizationResult):
            raise ValueError(
                f"global_optimization_result must be a GlobalOptimizationResult, got {global_optimization_result!r}"
            )
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
