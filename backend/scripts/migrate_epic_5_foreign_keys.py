"""Add the missing Epic 5 foreign-key constraints when absent (FND-05).

Several relational id columns across the Epic 5 response-planning chain
(`response_plans`, `response_actions`, `route_results`, `plan_comparisons`,
`response_plan_uncovered_targets`) were originally persisted as plain
scalar integers/strings without a `FOREIGN KEY` constraint, even though the
ORM models (`src/database/models/`) reference real parent tables for every
one of them. This script brings the live schema to exact parity with the
ORM by adding each missing constraint.

Each constraint is added independently and idempotently (introspected
before being added via `inspect(connection).get_foreign_keys(table)`), so
running this script when some/all are already present is safe. If a
constraint cannot be added because existing rows already violate it (a
child row referencing a value absent from its parent table), that single
constraint is reported as failed and the script continues on to the rest
rather than aborting - existing/empty tables trivially satisfy every one of
these constraints, so a failure here means real orphaned data that needs
separate investigation before that specific constraint can be added.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402

# (constraint_name, table_name, column_name, referenced_table, referenced_column).
# Constraint names follow PostgreSQL's own default naming convention
# (<table>_<column>_fkey) so a live-migrated database ends up with the same
# constraint names a freshly created one gets from Base.metadata.create_all().
FOREIGN_KEYS = (
    (
        "response_plans_route_planning_run_id_fkey",
        "response_plans",
        "route_planning_run_id",
        "route_planning_runs",
        "id",
    ),
    (
        "response_actions_resource_id_fkey",
        "response_actions",
        "resource_id",
        "firefighting_resources",
        "id",
    ),
    (
        "response_actions_response_target_id_fkey",
        "response_actions",
        "response_target_id",
        "response_targets",
        "id",
    ),
    (
        "response_actions_route_result_id_fkey",
        "response_actions",
        "route_result_id",
        "route_results",
        "id",
    ),
    (
        "route_results_resource_id_fkey",
        "route_results",
        "resource_id",
        "firefighting_resources",
        "id",
    ),
    (
        "plan_comparisons_fire_event_id_fkey",
        "plan_comparisons",
        "fire_event_id",
        "fire_events",
        "id",
    ),
    (
        "plan_comparisons_optimized_plan_id_fkey",
        "plan_comparisons",
        "optimized_plan_id",
        "response_plans",
        "id",
    ),
    (
        "plan_comparisons_route_planning_run_id_fkey",
        "plan_comparisons",
        "route_planning_run_id",
        "route_planning_runs",
        "id",
    ),
    (
        "plan_comparisons_response_target_set_id_fkey",
        "plan_comparisons",
        "response_target_set_id",
        "response_target_sets",
        "id",
    ),
    (
        "response_plan_uncovered_targets_response_target_id_fkey",
        "response_plan_uncovered_targets",
        "response_target_id",
        "response_targets",
        "id",
    ),
)


def _constraint_exists(inspector, table_name: str, constraint_name: str) -> bool:
    return any(fk["name"] == constraint_name for fk in inspector.get_foreign_keys(table_name))


def add_missing_foreign_keys(engine) -> dict[str, str]:
    """Add whichever FOREIGN_KEYS entries are missing.

    Returns {constraint_name: outcome}, outcome one of "added", "already
    present", "skipped (<table> table missing)", or "failed: <reason>".
    Each constraint is applied in its own transaction so one failure (e.g.
    orphaned rows) does not roll back constraints already successfully
    added, and so a constraint check never blocks on a prior failed one.
    """
    results: dict[str, str] = {}
    for constraint_name, table_name, column_name, ref_table, ref_column in FOREIGN_KEYS:
        with engine.connect() as connection:
            inspector = inspect(connection)
            if not inspector.has_table(table_name):
                results[constraint_name] = f"skipped ('{table_name}' table does not exist)"
                continue
            if not inspector.has_table(ref_table):
                results[constraint_name] = f"skipped (referenced table '{ref_table}' does not exist)"
                continue
            if _constraint_exists(inspector, table_name, constraint_name):
                results[constraint_name] = "already present"
                continue

        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                        f"FOREIGN KEY ({column_name}) REFERENCES {ref_table} ({ref_column})"
                    )
                )
            results[constraint_name] = "added"
        except SQLAlchemyError as exc:
            results[constraint_name] = (
                f"failed: {exc.__class__.__name__} - likely existing rows in "
                f"{table_name}.{column_name} that do not reference a real "
                f"{ref_table}.{ref_column}; resolve the data before re-running."
            )
    return results


def main() -> int:
    engine = get_engine()
    results = add_missing_foreign_keys(engine)
    had_failure = False
    for constraint_name, _, _, _, _ in FOREIGN_KEYS:
        outcome = results[constraint_name]
        print(f"{constraint_name}: {outcome}")
        if outcome.startswith("failed"):
            had_failure = True
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
