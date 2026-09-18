"""Create global_planning_runs/global_planning_run_events and add
response_plans.global_planning_run_id when absent (Stage 2 of the Global
Multi-Incident Optimizer refactor).

Five independent, idempotent steps, each in its own transaction and each
introspected before being applied, so running this script when some/all are
already present is safe, and one step's failure cannot roll back another
step that already succeeded:
  1. CREATE TABLE global_planning_runs (checkfirst)
  2. CREATE TABLE global_planning_run_events (checkfirst)
  3. ALTER TABLE response_plans ADD COLUMN global_planning_run_id
  4. ALTER TABLE response_plans ADD CONSTRAINT ... FOREIGN KEY (...)
  5. CREATE INDEX on response_plans.global_planning_run_id

Step 4 is wrapped in its own try/except (mirroring
migrate_epic_5_foreign_keys.py's established pattern for this exact
project): SQLite cannot execute ALTER TABLE ADD CONSTRAINT ... FOREIGN KEY
at all, so this step is expected to report "failed" in SQLite-based tests
and only actually succeeds against a real PostgreSQL database (Neon). That
failure never blocks steps 1-3/5 from completing.

No backfill: every historical ResponsePlan row keeps
global_planning_run_id = NULL - a plan's persisted actions cannot tell us
retroactively which (if any) global cycle considered it, mirroring Stage
1's migrate_resource_commitments.py precedent of never fabricating this
kind of provenance for pre-existing rows.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402
from src.database.models.global_planning_run_db import GlobalPlanningRunDB  # noqa: E402
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB  # noqa: E402

RUN_TABLE_NAME = GlobalPlanningRunDB.__tablename__
EVENT_TABLE_NAME = GlobalPlanningRunEventDB.__tablename__
RESPONSE_PLANS_TABLE_NAME = "response_plans"
COLUMN_NAME = "global_planning_run_id"
FK_CONSTRAINT_NAME = "response_plans_global_planning_run_id_fkey"
INDEX_NAME = "ix_response_plans_global_planning_run_id"


def create_run_table(connection) -> bool:
    if inspect(connection).has_table(RUN_TABLE_NAME):
        return False
    GlobalPlanningRunDB.__table__.create(bind=connection, checkfirst=True)
    return True


def create_run_event_table(connection) -> bool:
    if inspect(connection).has_table(EVENT_TABLE_NAME):
        return False
    GlobalPlanningRunEventDB.__table__.create(bind=connection, checkfirst=True)
    return True


def add_response_plans_column(connection) -> bool:
    existing_columns = {column["name"] for column in inspect(connection).get_columns(RESPONSE_PLANS_TABLE_NAME)}
    if COLUMN_NAME in existing_columns:
        return False
    connection.execute(text(f"ALTER TABLE {RESPONSE_PLANS_TABLE_NAME} ADD COLUMN {COLUMN_NAME} INTEGER NULL"))
    return True


def add_response_plans_fk(connection) -> bool:
    existing_fks = {fk["name"] for fk in inspect(connection).get_foreign_keys(RESPONSE_PLANS_TABLE_NAME)}
    if FK_CONSTRAINT_NAME in existing_fks:
        return False
    connection.execute(
        text(
            f"ALTER TABLE {RESPONSE_PLANS_TABLE_NAME} ADD CONSTRAINT {FK_CONSTRAINT_NAME} "
            f"FOREIGN KEY ({COLUMN_NAME}) REFERENCES {RUN_TABLE_NAME} (id)"
        )
    )
    return True


def add_response_plans_index(connection) -> bool:
    existing_indexes = {index["name"] for index in inspect(connection).get_indexes(RESPONSE_PLANS_TABLE_NAME)}
    if INDEX_NAME in existing_indexes:
        return False
    connection.execute(text(f"CREATE INDEX {INDEX_NAME} ON {RESPONSE_PLANS_TABLE_NAME} ({COLUMN_NAME})"))
    return True


def migrate() -> dict[str, str]:
    """Apply all five steps independently. Returns {step_name: outcome}.

    outcome is one of "created"/"added", "already present", or
    "failed: <ExceptionClassName>" (step 4 only, see module docstring).
    """
    engine = get_engine()
    results: dict[str, str] = {}

    with engine.begin() as connection:
        results["global_planning_runs_table"] = "created" if create_run_table(connection) else "already present"
    with engine.begin() as connection:
        results["global_planning_run_events_table"] = (
            "created" if create_run_event_table(connection) else "already present"
        )
    with engine.begin() as connection:
        results["response_plans.global_planning_run_id_column"] = (
            "added" if add_response_plans_column(connection) else "already present"
        )

    try:
        with engine.begin() as connection:
            added = add_response_plans_fk(connection)
        results["response_plans.global_planning_run_id_fk"] = "added" if added else "already present"
    except SQLAlchemyError as exc:
        results["response_plans.global_planning_run_id_fk"] = f"failed: {exc.__class__.__name__}"

    with engine.begin() as connection:
        results["response_plans.global_planning_run_id_index"] = (
            "added" if add_response_plans_index(connection) else "already present"
        )

    return results


def main() -> int:
    results = migrate()
    for step_name, outcome in results.items():
        print(f"{step_name}: {outcome}")
    return 1 if any(outcome.startswith("failed") for outcome in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
