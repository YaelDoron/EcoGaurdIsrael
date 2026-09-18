"""Pure Global GA (Stage 4 of the Global Multi-Incident Optimizer refactor).

Operates purely on an already-built GlobalPlanningInput (Stage 3): no
sqlalchemy, repositories, database, external providers, DijkstraCalculator,
or agents are imported anywhere in this package (Task 37) - see
tests/architecture/test_global_response_optimization_architecture_guard.py.

`GlobalResponseOptimizationService` is deliberately NOT re-exported here:
it is the one file in this package that imports GlobalOptimizationResult/
GlobalEventOptimizationResult back from src.models, and those two model
modules import GlobalResponseOptimizationConfig from this package - eagerly
aggregating the service here as well would close that into a real import
cycle. Import it directly:
    from src.calculators.global_response_optimization.global_response_optimization_service
        import GlobalResponseOptimizationService
"""

from src.calculators.global_response_optimization.global_allocation_slot_factory import (
    GlobalAllocationSlotFactory,
    GlobalAllocationSlotGenerationError,
)
from src.calculators.global_response_optimization.global_demand_scoring_policy import GlobalDemandScoringPolicy
from src.calculators.global_response_optimization.global_genetic_optimizer import (
    GlobalCandidateEvaluator,
    GlobalEvaluatedChromosome,
    GlobalGeneticOptimizationResult,
    GlobalGeneticResponseOptimizer,
    GlobalTournamentSelector,
)
from src.calculators.global_response_optimization.global_initial_population_generator import (
    GlobalInitialPopulationGenerator,
)
from src.calculators.global_response_optimization.global_response_optimization_config import (
    GlobalResponseOptimizationConfig,
)
from src.calculators.global_response_optimization.global_response_optimization_problem import (
    GlobalResponseOptimizationProblem,
    build_global_optimization_problem,
)
from src.calculators.global_response_optimization.global_response_plan_chromosome_decoder import (
    GlobalResponsePlanChromosomeDecodeError,
    GlobalResponsePlanChromosomeDecoder,
)
from src.calculators.global_response_optimization.global_response_plan_scorer import (
    GlobalResponsePlanScorer,
    GlobalResponsePlanScoringError,
    GlobalScoreBreakdown,
)
from src.calculators.global_response_optimization.severity_demand_policy import SeverityDemandPolicy

__all__ = [
    "GlobalAllocationSlotFactory",
    "GlobalAllocationSlotGenerationError",
    "GlobalDemandScoringPolicy",
    "SeverityDemandPolicy",
    "GlobalResponseOptimizationConfig",
    "GlobalResponseOptimizationProblem",
    "build_global_optimization_problem",
    "GlobalResponsePlanScorer",
    "GlobalResponsePlanScoringError",
    "GlobalScoreBreakdown",
    "GlobalInitialPopulationGenerator",
    "GlobalResponsePlanChromosomeDecoder",
    "GlobalResponsePlanChromosomeDecodeError",
    "GlobalCandidateEvaluator",
    "GlobalEvaluatedChromosome",
    "GlobalGeneticOptimizationResult",
    "GlobalGeneticResponseOptimizer",
    "GlobalTournamentSelector",
]
