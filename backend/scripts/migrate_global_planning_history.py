"""Add the demand/shortage historical-persistence columns to
global_planning_runs and global_planning_run_events when absent (Stage 6 of
the Global Multi-Incident Optimizer refactor).

Every new column is a simple, backend-agnostic ADD COLUMN - unlike
migrate_global_planning_run.py's response_plans FK step, none of this needs
SQLite-vs-Postgres special-casing. One independent, idempotent step per
column (introspected before being applied), each in its own transaction, so
running this script when some/all columns are already present is safe and
one column's failure cannot roll back another that already succeeded.

No backfill: every historical GlobalPlanningRun/GlobalPlanningRunEvent row
predates this snapshot capture entirely - there is no way to reconstruct a
past cycle's exact severity/demand/shortage/GA-config values after the
fact, so every new column stays NULL on existing rows, mirroring this
project's established migration convention (see migrate_global_planning_run.py,
migrate_dispatch_state.py).
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402
from src.database.models.global_planning_run_db import GlobalPlanningRunDB  # noqa: E402
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB  # noqa: E402

RUN_TABLE_NAME = GlobalPlanningRunDB.__tablename__
EVENT_TABLE_NAME = GlobalPlanningRunEventDB.__tablename__

# (column_name, SQL type) - order matches declaration order in the ORM models.
RUN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("raw_input_fingerprint", "VARCHAR"),
    ("optimization_policy_fingerprint", "VARCHAR"),
    ("random_seed", "INTEGER"),
    ("ga_population_size", "INTEGER"),
    ("ga_generation_count", "INTEGER"),
    ("ga_mutation_rate", "FLOAT"),
    ("ga_crossover_rate", "FLOAT"),
    ("demand_scoring_policy_methodology", "VARCHAR"),
    ("demand_scoring_policy_version", "VARCHAR"),
    ("severity_demand_policy_methodology", "VARCHAR"),
    ("severity_demand_policy_version", "VARCHAR"),
    ("stability_policy_methodology", "VARCHAR"),
    ("stability_policy_version", "VARCHAR"),
    ("fitness_score", "FLOAT"),
    ("coverage_score", "FLOAT"),
    ("average_eta_seconds", "FLOAT"),
    ("shortage_total_required", "INTEGER"),
    ("shortage_total_desired", "INTEGER"),
    ("shortage_total_assigned", "INTEGER"),
    ("shortage_unmet_required", "INTEGER"),
    ("shortage_unmet_desired", "INTEGER"),
    ("shortage_candidate_assignable_resource_count", "INTEGER"),
    ("shortage_committed_resource_count", "INTEGER"),
    ("shortage_unavailable_resource_count", "INTEGER"),
    ("shortage_locked_resources_preserved", "INTEGER"),
)

EVENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("severity_assessment_id", "INTEGER"),
    ("severity_level", "VARCHAR"),
    ("severity_score", "FLOAT"),
    ("demand_source", "VARCHAR"),
    ("demand_policy_methodology", "VARCHAR"),
    ("demand_policy_version", "VARCHAR"),
    ("minimum_resources", "INTEGER"),
    ("desired_resources", "INTEGER"),
    ("assigned_resources", "INTEGER"),
    ("required_slots_covered", "INTEGER"),
    ("required_slots_uncovered", "INTEGER"),
    ("desired_slots_covered", "INTEGER"),
    ("desired_slots_uncovered", "INTEGER"),
    ("predicted_risk_slots_covered", "INTEGER"),
    ("locked_resources_preserved", "INTEGER"),
    ("coverage_score", "FLOAT"),
    ("average_eta_seconds", "FLOAT"),
)


def _add_column_if_absent(connection, *, table_name: str, column_name: str, sql_type: str) -> bool:
    existing_columns = {column["name"] for column in inspect(connection).get_columns(table_name)}
    if column_name in existing_columns:
        return False
    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type} NULL"))
    return True


def migrate() -> dict[str, str]:
    """Add every missing column independently. Returns {"<table>.<column>": outcome}.

    outcome is "added" or "already present" - every step here is a plain
    ADD COLUMN, so (unlike migrate_global_planning_run.py's FK step) no
    step is expected to fail on SQLite.
    """
    engine = get_engine()
    results: dict[str, str] = {}

    for column_name, sql_type in RUN_COLUMNS:
        key = f"{RUN_TABLE_NAME}.{column_name}"
        with engine.begin() as connection:
            added = _add_column_if_absent(
                connection, table_name=RUN_TABLE_NAME, column_name=column_name, sql_type=sql_type
            )
        results[key] = "added" if added else "already present"

    for column_name, sql_type in EVENT_COLUMNS:
        key = f"{EVENT_TABLE_NAME}.{column_name}"
        with engine.begin() as connection:
            added = _add_column_if_absent(
                connection, table_name=EVENT_TABLE_NAME, column_name=column_name, sql_type=sql_type
            )
        results[key] = "added" if added else "already present"

    return results


def main() -> int:
    results = migrate()
    for key, outcome in results.items():
        print(f"{key}: {outcome}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
