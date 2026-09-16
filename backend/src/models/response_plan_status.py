"""Response-plan completeness status."""
from __future__ import annotations

from enum import Enum


class ResponsePlanStatus(str, Enum):
    """Completeness of a generated response plan."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NO_FEASIBLE_ASSIGNMENTS = "no_feasible_assignments"
