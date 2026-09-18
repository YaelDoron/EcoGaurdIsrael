"""GlobalResponseOptimizationService: the Stage-4/5 entry point (Task 33).

optimize(global_planning_input, config, ...) -> GlobalOptimizationResult

Pure: takes the ALREADY-BUILT GlobalPlanningInput (Stage 3, now carrying
Stage 5's incident_demands) and configs, returns a result. No repository/DB
access, no Dijkstra, no target/severity regeneration, no per-event
optimizer invocation, no persistence - matching Task 1's architectural
boundary and Task 41's "no persistence yet."

Flow: build demand-aware allocation slots -> build optimization problem ->
run the Global GA (demand-aware scorer) -> the GA's best chromosome is
already decoded into actions -> compute result metrics -> project per-event
views (pure grouping, not a second optimization - Task 22) -> compute the
global shortage summary (Tasks 18/20/21).
"""
from __future__ import annotations

from src.calculators.global_response_optimization.global_assignment_change_calculator import (
    compute_assignment_changes,
)
from src.calculators.global_response_optimization.global_assignment_stability_policy import (
    GlobalAssignmentStabilityPolicy,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_genetic_optimizer import (
    GlobalCandidateEvaluator,
    GlobalGeneticResponseOptimizer,
)
from src.calculators.global_response_optimization.global_initial_population_generator import (
    GlobalInitialPopulationGenerator,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    METHODOLOGY,
    METHODOLOGY_VERSION,
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import GlobalResponsePlanScorer
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy
from src.models.global_allocation_slot import GlobalAllocationSlot
from src.models.global_event_optimization_result import GlobalEventOptimizationResult
from src.models.global_event_resource_demand_result import GlobalEventResourceDemandResult
from src.models.global_optimization_result import GlobalOptimizationResult
from src.models.global_planning_input import GlobalPlanningInput
from src.models.global_resource_shortage import GlobalResourceShortage
from src.models.global_response_action import GlobalResponseAction
from src.models.global_response_plan_chromosome import GlobalResponsePlanChromosome
from src.models.response_target_type import ResponseTargetType


class GlobalResponseOptimizationService:
    """Runs the Global GA over a GlobalPlanningInput and returns a structured, demand-aware, per-event-projected result."""

    def __init__(self, optimizer: GlobalGeneticResponseOptimizer | None = None) -> None:
        self._optimizer = optimizer

    def optimize(
        self,
        global_planning_input: GlobalPlanningInput,
        config: GlobalResponseOptimizationConfig | None = None,
        demand_scoring_policy: GlobalDemandScoringPolicy | None = None,
        severity_demand_policy: SeverityDemandPolicy | None = None,
        stability_policy: GlobalAssignmentStabilityPolicy | None = None,
    ) -> GlobalOptimizationResult:
        if not isinstance(global_planning_input, GlobalPlanningInput):
            raise ValueError(f"global_planning_input must be a GlobalPlanningInput, got {global_planning_input!r}")
        config = config or GlobalResponseOptimizationConfig()
        if not isinstance(config, GlobalResponseOptimizationConfig):
            raise ValueError(f"config must be a GlobalResponseOptimizationConfig, got {config!r}")
        demand_scoring_policy = demand_scoring_policy or GlobalDemandScoringPolicy()
        severity_demand_policy = severity_demand_policy or SeverityDemandPolicy()
        stability_policy = stability_policy or GlobalAssignmentStabilityPolicy()

        scorer = GlobalResponsePlanScorer(config, demand_scoring_policy, severity_demand_policy, stability_policy)
        optimizer = self._optimizer or GlobalGeneticResponseOptimizer(evaluator=GlobalCandidateEvaluator(scorer=scorer))

        problem = build_global_optimization_problem(global_planning_input)
        initial_population = GlobalInitialPopulationGenerator.generate(problem, config, scorer=scorer)
        ga_result = optimizer.optimize(problem, config, initial_population)

        resource_to_slot_id = dict(zip(ga_result.best_chromosome.resource_ids, ga_result.best_chromosome.genes))

        event_results = self._project_per_event(
            active_fire_event_ids=global_planning_input.active_fire_event_ids,
            actions=ga_result.actions,
            resource_to_slot_id=resource_to_slot_id,
            problem=problem,
        )
        shortage = self._compute_shortage(
            global_planning_input=global_planning_input, problem=problem, event_results=event_results
        )
        assignment_changes = compute_assignment_changes(global_planning_input.current_assignments, ga_result.actions)

        return GlobalOptimizationResult(
            global_planning_run_id=global_planning_input.global_planning_run_id,
            input_fingerprint=global_planning_input.input_fingerprint,
            optimization_methodology=METHODOLOGY,
            optimization_methodology_version=METHODOLOGY_VERSION,
            random_seed=config.random_seed,
            fitness_score=ga_result.score.fitness_score,
            coverage_score=ga_result.score.coverage_score,
            average_eta_seconds=ga_result.score.average_eta_seconds,
            actions=ga_result.actions,
            uncovered_slot_ids=ga_result.score.uncovered_slot_ids,
            event_results=event_results,
            shortage=shortage,
            config=config,
            assignment_changes=assignment_changes,
        )

    @staticmethod
    def _project_per_event(
        *,
        active_fire_event_ids: tuple[int, ...],
        actions: tuple[GlobalResponseAction, ...],
        resource_to_slot_id: dict[str, str | None],
        problem: GlobalResponseOptimizationProblem,
    ) -> tuple[GlobalEventOptimizationResult, ...]:
        actions_by_event: dict[int, list[GlobalResponseAction]] = {
            fire_event_id: [] for fire_event_id in active_fire_event_ids
        }
        for action in actions:
            actions_by_event.setdefault(action.fire_event_id, []).append(action)

        slots_by_event: dict[int, list[GlobalAllocationSlot]] = {
            fire_event_id: [] for fire_event_id in active_fire_event_ids
        }
        for slot in problem.slots:
            slots_by_event.setdefault(slot.fire_event_id, []).append(slot)

        results = []
        for fire_event_id in active_fire_event_ids:
            event_actions = tuple(
                sorted(actions_by_event.get(fire_event_id, ()), key=lambda action: (action.response_target_id, action.resource_id))
            )
            event_slots = slots_by_event.get(fire_event_id, ())
            # Which slot_id each action actually covers - read from the winning
            # chromosome itself (resource_id -> slot_id), NOT re-derived from
            # response_target_id, since Stage 5 allows several slots (and
            # several resources) to share the same response_target_id.
            covered_slot_ids = tuple(
                sorted(
                    slot_id
                    for slot_id in (resource_to_slot_id.get(action.resource_id) for action in event_actions)
                    if slot_id is not None
                )
            )
            uncovered_slot_ids = tuple(
                sorted(slot.slot_id for slot in event_slots if slot.slot_id not in covered_slot_ids)
            )

            eta_values = [action.eta_seconds for action in event_actions]
            average_eta_seconds = sum(eta_values) / len(eta_values) if eta_values else None
            coverage_score = 100.0 * len(covered_slot_ids) / len(event_slots) if event_slots else 0.0

            demand_result = GlobalResponseOptimizationService._demand_result(
                fire_event_id=fire_event_id,
                event_slots=event_slots,
                covered_slot_ids=set(covered_slot_ids),
                problem=problem,
            )

            results.append(
                GlobalEventOptimizationResult(
                    fire_event_id=fire_event_id,
                    actions=event_actions,
                    covered_slot_ids=covered_slot_ids,
                    uncovered_slot_ids=uncovered_slot_ids,
                    coverage_score=coverage_score,
                    average_eta_seconds=average_eta_seconds,
                    demand_result=demand_result,
                )
            )
        return tuple(results)

    @staticmethod
    def _demand_result(
        *,
        fire_event_id: int,
        event_slots: list[GlobalAllocationSlot],
        covered_slot_ids: set[str],
        problem: GlobalResponseOptimizationProblem,
    ) -> GlobalEventResourceDemandResult:
        # Classified by the OWNING TARGET's type - the authoritative signal
        # the slot factory itself used (Task 9: only ACTIVE_FIRE targets ever
        # carry required=True slots) - not by "shares a response_target_id
        # with a required slot", which would misclassify a suppression
        # target whose demand happens to have minimum_resources == 0.
        def _target_type(slot: GlobalAllocationSlot) -> ResponseTargetType:
            return problem.targets_by_id[slot.response_target_id].target_type

        # Stage 6: minimum/desired come from the AUTHORITATIVE severity-driven
        # GlobalIncidentDemand, never re-derived from slot counts - the slot
        # factory may have generated MORE ACTIVE_FIRE slots than
        # desired_resources calls for, purely to give already-hard-dispatched
        # resources a legal home after a demand decrease (Task 8: never
        # fabricate an implicit recall). Those extra slots must never inflate
        # reported desired_resources - they are reported separately via
        # locked_resources_preserved.
        demand = problem.incident_demands_by_event.get(fire_event_id)
        minimum_resources = demand.minimum_resources if demand is not None else 0
        desired_resources = demand.desired_resources if demand is not None else 0

        active_fire_slots_by_index = sorted(
            (slot for slot in event_slots if _target_type(slot) is ResponseTargetType.ACTIVE_FIRE),
            key=lambda slot: slot.slot_index,
        )
        required_slots = active_fire_slots_by_index[:minimum_resources]
        desired_slots = active_fire_slots_by_index[minimum_resources:desired_resources]
        locked_overflow_slots = active_fire_slots_by_index[desired_resources:]
        predicted_risk_slots = [slot for slot in event_slots if _target_type(slot) is ResponseTargetType.PREDICTED_RISK]

        required_covered = sum(1 for slot in required_slots if slot.slot_id in covered_slot_ids)
        desired_covered = sum(1 for slot in desired_slots if slot.slot_id in covered_slot_ids)
        locked_overflow_covered = sum(1 for slot in locked_overflow_slots if slot.slot_id in covered_slot_ids)
        predicted_risk_covered = sum(1 for slot in predicted_risk_slots if slot.slot_id in covered_slot_ids)

        return GlobalEventResourceDemandResult(
            fire_event_id=fire_event_id,
            minimum_resources=minimum_resources,
            desired_resources=desired_resources,
            suppression_resources_assigned=required_covered + desired_covered + locked_overflow_covered,
            required_slots_covered=required_covered,
            required_slots_uncovered=minimum_resources - required_covered,
            desired_slots_covered=desired_covered,
            desired_slots_uncovered=len(desired_slots) - desired_covered,
            predicted_risk_slots_covered=predicted_risk_covered,
            locked_resources_preserved=locked_overflow_covered,
        )

    @staticmethod
    def _compute_shortage(
        *,
        global_planning_input: GlobalPlanningInput,
        problem: GlobalResponseOptimizationProblem,
        event_results: tuple[GlobalEventOptimizationResult, ...],
    ) -> GlobalResourceShortage:
        total_required = sum(demand.minimum_resources for demand in global_planning_input.incident_demands)
        total_desired = sum(demand.desired_resources for demand in global_planning_input.incident_demands)
        total_assigned = sum(result.demand_result.suppression_resources_assigned for result in event_results)
        unmet_required = sum(result.demand_result.required_slots_uncovered for result in event_results)
        unmet_desired = sum(result.demand_result.unmet_desired for result in event_results)
        locked_resources_preserved = sum(result.demand_result.locked_resources_preserved for result in event_results)

        candidate_assignable_resource_count = sum(1 for resource in global_planning_input.resources if resource.is_assignable)
        committed_resource_count = sum(1 for resource in global_planning_input.resources if resource.is_committed)
        unavailable_resource_count = sum(1 for resource in global_planning_input.resources if not resource.is_assignable)

        return GlobalResourceShortage(
            total_required=total_required,
            total_desired=total_desired,
            total_assigned=total_assigned,
            unmet_required=unmet_required,
            unmet_desired=unmet_desired,
            candidate_assignable_resource_count=candidate_assignable_resource_count,
            committed_resource_count=committed_resource_count,
            unavailable_resource_count=unavailable_resource_count,
            locked_resources_preserved=locked_resources_preserved,
        )
