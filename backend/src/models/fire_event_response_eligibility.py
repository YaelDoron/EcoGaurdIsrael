"""Two different questions about a FireEvent's status (Task 9A) - kept apart on purpose.

    ACTIVE FOR DETECTION / MONITORING   {SUSPECTED, CONFIRMED}
        the event stays persisted, matches later evidence, keeps its history, is visible on the dashboard / in the
        active-event APIs and can be promoted (SUSPECTED -> CONFIRMED) or later closed (RESOLVED / DISMISSED).

    ELIGIBLE FOR EMERGENCY RESPONSE     {CONFIRMED}
        only these events drive severity, spread, response targets, routing (road graph / Dijkstra), resource
        allocation (GA), global response planning and resource commitment.

`ACTIVE_FOR_MONITORING_STATUSES` is deliberately NOT changed to {CONFIRMED}: that would stop SUSPECTED incidents from
accumulating evidence. Modules must import these names instead of writing `status == CONFIRMED` themselves, so the
single definition of "response eligible" lives here (a test forbids scattered CONFIRMED checks in services).

The future AI Hybrid policy (NO_EVENT / SUSPECTED / CONFIRMED) plugs into exactly these two sets; nothing here assumes
that every active event eventually becomes CONFIRMED (a SUSPECTED event may also be dismissed).
"""
from __future__ import annotations

from src.models.fire_event_status import FireEventStatus

ACTIVE_FOR_MONITORING_STATUSES: frozenset[FireEventStatus] = frozenset(
    {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}
)
RESPONSE_ELIGIBLE_STATUSES: frozenset[FireEventStatus] = frozenset({FireEventStatus.CONFIRMED})

ACTIVE_FOR_MONITORING_STATUS_VALUES: tuple[str, ...] = tuple(sorted(s.value for s in ACTIVE_FOR_MONITORING_STATUSES))
RESPONSE_ELIGIBLE_STATUS_VALUES: tuple[str, ...] = tuple(sorted(s.value for s in RESPONSE_ELIGIBLE_STATUSES))


def _as_status(status: FireEventStatus | str) -> FireEventStatus:
    return status if isinstance(status, FireEventStatus) else FireEventStatus(status)


def is_active_for_monitoring(status: FireEventStatus | str) -> bool:
    """SUSPECTED or CONFIRMED: keep matching evidence, keep history, show on the dashboard."""
    return _as_status(status) in ACTIVE_FOR_MONITORING_STATUSES


def is_response_eligible(status: FireEventStatus | str) -> bool:
    """CONFIRMED only: allowed to trigger the emergency-response pipeline."""
    return _as_status(status) in RESPONSE_ELIGIBLE_STATUSES


def event_is_response_eligible(stored_event) -> bool:
    """Convenience for a StoredFireEvent (anything with `.event.status`)."""
    return is_response_eligible(stored_event.event.status)
