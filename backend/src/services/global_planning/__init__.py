"""GlobalPlanningRun + GlobalPlanningOrchestrator (Stage 2 of the Global
Multi-Incident Optimizer refactor).

This is NOT global optimization. It establishes a first-class
"one planning cycle observed all active FireEvents together" concept while
the existing per-FireEvent routing and genetic optimizer remain completely
unchanged underneath it - see global_planning_orchestrator.py's own
docstring for the full flow and global_planning_config.py for why its
methodology is explicitly named "legacy_per_event_orchestration".
"""

from src.services.global_planning.global_candidate_collector import GlobalCandidateCollector
from src.services.global_planning.global_candidate_resource_config import GlobalCandidateResourceConfig
from src.services.global_planning.global_planning_input_builder import GlobalPlanningInputBuilder
from src.services.global_planning.global_planning_input_unstable import GlobalPlanningInputUnstable
from src.services.global_planning.global_planning_orchestrator import GlobalPlanningOrchestrator
from src.services.global_planning.global_planning_result import GlobalPlanningEventResult, GlobalPlanningResult
from src.services.global_planning.global_planning_snapshot import GlobalPlanningSnapshot
from src.services.global_planning.global_route_matrix_builder import (
    GlobalRouteMatrixBuilder,
    GlobalRouteMatrixBuildResult,
)

__all__ = [
    "GlobalPlanningOrchestrator",
    "GlobalPlanningEventResult",
    "GlobalPlanningResult",
    "GlobalPlanningSnapshot",
    "GlobalCandidateCollector",
    "GlobalCandidateResourceConfig",
    "GlobalPlanningInputBuilder",
    "GlobalPlanningInputUnstable",
    "GlobalRouteMatrixBuilder",
    "GlobalRouteMatrixBuildResult",
]
