"""Input preparation status for operational response targets."""
from __future__ import annotations

from enum import Enum


class ResponseTargetInputStatus(Enum):
    """Readiness state for preparing ResponseTargetInput."""

    READY = "ready"
    INACTIVE_EVENT = "inactive_event"
