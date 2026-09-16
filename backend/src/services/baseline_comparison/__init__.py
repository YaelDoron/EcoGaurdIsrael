"""Orchestration for comparing an optimized plan against its deterministic baseline."""

from src.services.baseline_comparison.baseline_comparison_adapters import (
    PersistedOptimizedPlanScore,
    PersistedOptimizedPlanSnapshot,
    PersistedRoutePlanningRunSnapshot,
    PersistedRouteResultSnapshot,
    ResponsePlanRepositoryOptimizedPlanReader,
    ResponsePlanScorerBaselineAdapter,
    RoutePlanningRepositoryRunReader,
)
from src.services.baseline_comparison.baseline_comparison_ports import (
    OptimizedPlanLike,
    OptimizedPlanReader,
    ResponseTargetSetReader,
    RoutePlanningRunLike,
    RoutePlanningRunReader,
    RouteResultLike,
)
from src.services.baseline_comparison.baseline_comparison_service import (
    BaselineComparisonService,
    BaselineComparisonServiceError,
)

__all__ = [
    "BaselineComparisonService",
    "BaselineComparisonServiceError",
    "PersistedOptimizedPlanScore",
    "PersistedOptimizedPlanSnapshot",
    "PersistedRoutePlanningRunSnapshot",
    "PersistedRouteResultSnapshot",
    "ResponsePlanRepositoryOptimizedPlanReader",
    "ResponsePlanScorerBaselineAdapter",
    "RoutePlanningRepositoryRunReader",
    "OptimizedPlanLike",
    "OptimizedPlanReader",
    "ResponseTargetSetReader",
    "RoutePlanningRunLike",
    "RoutePlanningRunReader",
    "RouteResultLike",
]
