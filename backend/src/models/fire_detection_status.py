"""Pure active-wildfire detection decision statuses."""
from __future__ import annotations

from enum import Enum


class FireDetectionStatus(Enum):
    """Detection result for a single wildfire evidence candidate.

    Later persistence/orchestration tasks may map SUSPECTED/CONFIRMED decisions
    into persisted event state and add lifecycle states such as
    RESOLVED/DISMISSED. Those lifecycle states are intentionally not part of
    this pure decision.
    """

    NO_EVENT = "no_event"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
