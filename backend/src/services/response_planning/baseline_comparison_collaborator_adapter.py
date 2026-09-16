"""Task 6 adapter bridging Task 5's BaselineComparisonCollaborator port to BaselineComparisonService.

BaselineComparisonService.compare(response_plan_id=...) already persists a
PlanComparison row internally (via PlanComparisonRepository.save()) but
returns only the unwrapped PlanComparison domain object, discarding the
persisted row id before returning. Task 5's PlanningRefreshResult needs that
id (comparison_id), so this adapter calls the real service exactly once and
then deterministically recovers the row it just created.

Recovery strategy (see module docstring rationale in the Task 6 report):
PlanComparisonRepository has no "get by response_plan_id" lookup, but
list_for_fire_event(fire_event_id) already exists and is exactly what Task
5's orchestrator itself uses for the same "does a comparison exist for this
plan" question. This adapter reuses that same call, filters to
optimized_plan_id == response_plan_id, and - per the required
newest-DB-id-wins tie-break for any pre-existing historical rows - returns
the one with the highest id. No new repository method is introduced.
"""
from __future__ import annotations

from src.repositories.plan_comparison_repository import PlanComparisonRepository, StoredPlanComparison
from src.services.baseline_comparison.baseline_comparison_service import BaselineComparisonService


class BaselineComparisonAdapterError(Exception):
    """Raised when the comparison just created by BaselineComparisonService cannot be located."""


class BaselineComparisonCollaboratorAdapter:
    """Bridges Task 5's BaselineComparisonCollaborator port to the real BaselineComparisonService."""

    def __init__(
        self,
        baseline_comparison_service: BaselineComparisonService,
        plan_comparison_repository: PlanComparisonRepository | None = None,
    ) -> None:
        self._baseline_comparison_service = baseline_comparison_service
        self._plan_comparison_repository = plan_comparison_repository or PlanComparisonRepository()

    def compare(self, *, response_plan_id: int) -> StoredPlanComparison:
        """Run the real baseline comparison exactly once and return its persisted row."""
        comparison = self._baseline_comparison_service.compare(response_plan_id=response_plan_id)
        return self._find_newest_stored_comparison(comparison.fire_event_id, response_plan_id)

    def _find_newest_stored_comparison(self, fire_event_id: int, response_plan_id: int) -> StoredPlanComparison:
        matches = [
            stored
            for stored in self._plan_comparison_repository.list_for_fire_event(fire_event_id)
            if stored.comparison.optimized_plan_id == response_plan_id
        ]
        if not matches:
            raise BaselineComparisonAdapterError(
                f"BaselineComparisonService reported success for ResponsePlan {response_plan_id!r} "
                "but no matching PlanComparison could be found afterward."
            )
        return max(matches, key=lambda stored: stored.id)
