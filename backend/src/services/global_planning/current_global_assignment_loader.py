"""CurrentGlobalAssignmentLoader: reads each active FireEvent's current
ResponsePlan + ResourceCommitment state and resolves it down to one
CurrentGlobalAssignment per still-committed resource (Stage 6 of the
Global Multi-Incident Optimizer refactor, Task 14).

ResourceCommitment alone answers "who owns this resource and is it
DISPATCHED" but not "which exact response_target_id is it satisfying" -
that only exists on the current ResponsePlan's ResponseActions. This loader
joins the two, once, outside the Global GA (the GA package itself never
queries a repository - Task 60), and hands the result into
GlobalPlanningInput as plain, immutable data.

A commitment with no matching action on the current plan (e.g. a
just-superseded plan mid-transition) is defensively skipped rather than
guessed - GlobalPlanningInputBuilder's mid-build revalidation already
protects against building on stale resource state.
"""
from __future__ import annotations

from collections.abc import Iterable

from src.models.current_global_assignment import CurrentGlobalAssignment
from src.repositories.resource_commitment_repository import ResourceCommitmentRepository
from src.services.response_planning.current_response_plan_resolver import CurrentResponsePlanResolver


class CurrentGlobalAssignmentLoader:
    """Builds the pre-GA snapshot of every active FireEvent's currently-committed resources."""

    def __init__(
        self,
        current_response_plan_resolver: CurrentResponsePlanResolver | None = None,
        resource_commitment_repository: ResourceCommitmentRepository | None = None,
    ) -> None:
        self._resolver = current_response_plan_resolver or CurrentResponsePlanResolver()
        self._resource_commitment_repository = resource_commitment_repository or ResourceCommitmentRepository()

    def load(self, fire_event_ids: Iterable[int]) -> tuple[CurrentGlobalAssignment, ...]:
        assignments: list[CurrentGlobalAssignment] = []
        for fire_event_id in fire_event_ids:
            stored_plan = self._resolver.resolve(fire_event_id=fire_event_id)
            if stored_plan is None:
                continue
            commitments = self._resource_commitment_repository.get_for_fire_event(fire_event_id)
            commitment_by_resource_id = {commitment.resource_id: commitment for commitment in commitments}

            for action in stored_plan.plan.actions:
                resource_id = str(action.resource_id)
                commitment = commitment_by_resource_id.get(resource_id)
                if commitment is None:
                    continue
                assignments.append(
                    CurrentGlobalAssignment(
                        resource_id=resource_id,
                        fire_event_id=fire_event_id,
                        response_target_id=action.response_target_id,
                        response_plan_id=stored_plan.id,
                        dispatch_state=commitment.dispatch_state,
                    )
                )
        return tuple(assignments)
