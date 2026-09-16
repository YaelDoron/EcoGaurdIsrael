"""Build outcome for the current Planning Effective State."""
from __future__ import annotations

from enum import Enum


class PlanningEffectiveStateStatus(Enum):
    """Outcome of attempting to build the current PlanningEffectiveState for a FireEvent."""

    BUILT = "built"
    NO_CURRENT_TARGETS = "no_current_targets"
