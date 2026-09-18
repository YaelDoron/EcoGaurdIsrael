"""Per-FireEvent outcome within one GlobalPlanningRun (Stage 2 of the Global
Multi-Incident Optimizer refactor).

Deliberately mirrors src/services/response_planning/planning_refresh_result.py's
PlanningRefreshStatus rather than inventing parallel semantics (see
GlobalPlanningOrchestrator's own status-mapping docstring for the exact
1:1/near-1:1 mapping): PLANNED<-REFRESHED, NO_OP<-NO_OP,
SKIPPED_INACTIVE<-INACTIVE_EVENT, FAILED<-FAILED.
INSUFFICIENT_DATA is kept as its own value (rather than folded into FAILED
or SKIPPED_INACTIVE) because it is neither: the FireEvent is active and
planning did not error, it simply has no current ResponseTargetSet yet.
"""
from __future__ import annotations

from enum import Enum


class GlobalPlanningRunEventStatus(Enum):
    """One FireEvent's recorded outcome within a GlobalPlanningRun's membership."""

    PLANNED = "planned"
    NO_OP = "no_op"
    INSUFFICIENT_DATA = "insufficient_data"
    SKIPPED_INACTIVE = "skipped_inactive"
    FAILED = "failed"
