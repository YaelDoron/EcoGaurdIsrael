"""Exception raised when GlobalPlanningInputBuilder cannot produce an
internally consistent GlobalPlanningInput (Stage 3.1 of the Global
Multi-Incident Optimizer refactor).
"""
from __future__ import annotations


class GlobalPlanningInputUnstable(Exception):
    """Raised when material resource state (operational status /
    ResourceCommitment ownership) kept changing across every rebuild
    attempt while assembling a GlobalPlanningInput, so the builder could
    not produce a snapshot it can vouch for.

    Deliberately carries no repository/SQL detail - only the run id and
    how many attempts were made - so the future Global GA (or any other
    caller) never receives, or has to guess at, a known-stale input.
    """

    def __init__(self, *, global_planning_run_id: int, attempts: int) -> None:
        self.global_planning_run_id = global_planning_run_id
        self.attempts = attempts
        super().__init__(
            f"GlobalPlanningInput for GlobalPlanningRun {global_planning_run_id} could not be "
            f"stabilized after {attempts} attempt(s): resource state kept changing during construction."
        )
