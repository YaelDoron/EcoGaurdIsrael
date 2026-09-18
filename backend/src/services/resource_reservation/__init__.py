"""Cross-event resource-reservation services (Global Multi-Incident Optimizer refactor).

Stage 0 (CrossEventReservedResourceResolver) prevented the same firefighting
resource from being selected into two different active FireEvents' CURRENT
ResponsePlans by reading already-persisted plan history at candidate-
selection time - a read-side heuristic, not a transaction-safe reservation.

Stage 1 (ResponsePlanActivationService, ResourceCommitmentConflict) adds
real persistent, transaction-safe resource commitment semantics:
FirefightingResource.status stays a pure operational concept; a resource's
CURRENT owning FireEvent is now tracked explicitly via
ResourceCommitmentRepository, written atomically alongside the
ResponsePlanPlanningState sidecar that makes a plan current.
CrossEventReservedResourceResolver now unions both sources - see its own
docstring for exactly why the Stage-0 fallback is still needed and when it
can be removed.
"""

from src.services.resource_reservation.cross_event_reserved_resource_resolver import (
    CrossEventReservedResourceResolver,
)
from src.services.resource_reservation.resource_commitment_conflict import ResourceCommitmentConflict
from src.services.resource_reservation.response_plan_activation_service import ResponsePlanActivationService

__all__ = [
    "CrossEventReservedResourceResolver",
    "ResourceCommitmentConflict",
    "ResponsePlanActivationService",
]
