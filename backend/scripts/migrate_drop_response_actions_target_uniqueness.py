"""Drop response_actions' stale (response_plan_id, response_target_id)
uniqueness constraint when present (Stage 6 of the Global Multi-Incident
Optimizer refactor).

This constraint predates Stage 5's demand-aware GlobalAllocationSlotFactory,
which legitimately generates several allocation slots against the SAME
canonical ACTIVE_FIRE response_target_id whenever desired_resources > 1
(e.g. a HIGH/CRITICAL severity FireEvent with more than one suppression
resource assigned) - a real GlobalResponsePlanActivationService.activate()
call for such a FireEvent raises an unhandled ResponsePlanRepositoryError
and the ENTIRE global refresh cycle crashes, with this constraint still in
place. See src/database/models/response_action_db.py's updated docstring:
(response_plan_id, resource_id) remains enough to forbid the real
corruption this table must prevent (the same resource appearing twice in
one plan).

Idempotent: introspects for the constraint before dropping it, so running
this script when it is already absent is a safe no-op.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402

TABLE_NAME = "response_actions"
CONSTRAINT_NAME = "uq_response_actions_plan_target"


def constraint_exists(connection) -> bool:
    existing = {uc["name"] for uc in inspect(connection).get_unique_constraints(TABLE_NAME)}
    return CONSTRAINT_NAME in existing


def migrate() -> bool:
    """Drop response_actions.uq_response_actions_plan_target if present. Returns whether it was dropped."""
    engine = get_engine()
    with engine.begin() as connection:
        if not constraint_exists(connection):
            return False
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} DROP CONSTRAINT {CONSTRAINT_NAME}"))
        return True


def main() -> int:
    dropped = migrate()
    print(f"{TABLE_NAME}.{CONSTRAINT_NAME}: {'dropped' if dropped else 'already absent'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
