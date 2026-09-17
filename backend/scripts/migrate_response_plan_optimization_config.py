"""Add the exact GA-configuration snapshot columns to response_plans when absent (FND-04).

Existing rows intentionally keep every one of these columns NULL: a legacy
NULL means "the full optimization configuration used for this historical
run is unavailable" - not "use today's defaults." Persistence-layer
reconstruction (ResponsePlanRepository) already treats an all-NULL row as
optimization_config=None; nothing here fabricates historical values.

Brings the live schema to exact parity with the ORM
(src/database/models/response_plan_db.py): the 8 nullable
_OPTIMIZATION_CONFIG_COLUMNS plus the all-or-nothing CHECK constraint
ck_response_plans_optimization_config_all_or_none. Each column and the
constraint are added independently and idempotently (introspected before
being added), so running this script when some/all are already present is
safe.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402

TABLE_NAME = "response_plans"
CONSTRAINT_NAME = "ck_response_plans_optimization_config_all_or_none"

# (column_name, SQL type) in the exact order declared on ResponsePlanDB.
COLUMNS = (
    ("population_size", "INTEGER"),
    ("generation_count", "INTEGER"),
    ("mutation_rate", "DOUBLE PRECISION"),
    ("crossover_rate", "DOUBLE PRECISION"),
    ("eta_reference_seconds", "DOUBLE PRECISION"),
    ("initial_assignment_probability", "DOUBLE PRECISION"),
    ("tournament_size", "INTEGER"),
    ("elitism_count", "INTEGER"),
)

_CONSTRAINT_SQL = (
    "(" + " AND ".join(f"{name} IS NULL" for name, _ in COLUMNS) + ")"
    " OR ("
    + " AND ".join(f"{name} IS NOT NULL" for name, _ in COLUMNS)
    + ")"
)


def add_missing_columns(connection) -> tuple[str, ...]:
    """Add whichever of the 8 config columns are missing. Returns the names added."""
    existing_columns = {column["name"] for column in inspect(connection).get_columns(TABLE_NAME)}
    columns_added = []
    for column_name, sql_type in COLUMNS:
        if column_name in existing_columns:
            continue
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column_name} {sql_type} NULL"))
        columns_added.append(column_name)
    return tuple(columns_added)


def add_missing_constraint(connection) -> bool:
    """Add the all-or-nothing CHECK constraint if it is not already present. Returns whether it was added."""
    existing_constraints = {
        constraint["name"] for constraint in inspect(connection).get_check_constraints(TABLE_NAME)
    }
    if CONSTRAINT_NAME in existing_constraints:
        return False
    connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD CONSTRAINT {CONSTRAINT_NAME} CHECK ({_CONSTRAINT_SQL})"))
    return True


def migrate() -> tuple[tuple[str, ...], bool]:
    """Apply the safe nullable-columns + constraint migration.

    Returns (columns_added, constraint_added).
    """
    engine = get_engine()
    with engine.begin() as connection:
        columns_added = add_missing_columns(connection)
        constraint_added = add_missing_constraint(connection)
        return columns_added, constraint_added


def main() -> int:
    columns_added, constraint_added = migrate()
    for column_name, _ in COLUMNS:
        status = "added" if column_name in columns_added else "already present"
        print(f"{TABLE_NAME}.{column_name}: {status}")
    print(f"{CONSTRAINT_NAME}: {'added' if constraint_added else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
