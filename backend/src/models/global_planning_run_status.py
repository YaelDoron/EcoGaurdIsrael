"""Status of one GlobalPlanningRun (Stage 2 of the Global Multi-Incident Optimizer refactor)."""
from __future__ import annotations

from enum import Enum


class GlobalPlanningRunStatus(Enum):
    """Final (or in-flight) outcome of one global planning cycle.

    RUNNING is the transient state between run creation and finalization;
    every run persisted long enough to be queried after `run()` returns is
    one of the other four. See GlobalPlanningOrchestrator._determine_final_status
    for the exact aggregation rule from member results to one of these.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    NO_ACTIVE_EVENTS = "no_active_events"
