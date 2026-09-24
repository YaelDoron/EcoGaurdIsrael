"""Add the Task 9B AI Hybrid V5 audit columns to fire_event_ml_assessments.

Entirely additive: five NULLABLE columns (policy_version, policy_status, history_available, satellite_pass_count,
current_satellite_pixel_count). Existing rows and every other column are untouched; existing decision modes
(rule_only / shadow / hybrid) never read or write these columns, so they keep working whether or not this migration has
been applied. ONLY the ai_hybrid_v5 decision mode needs it.

Idempotent: each column is added only if it is missing; running it again is a safe no-op.

Usage:
    python -m scripts.migrate_add_fire_event_ml_assessment_ai_columns --inspect-only
    python -m scripts.migrate_add_fire_event_ml_assessment_ai_columns --apply

Uses the project's DATABASE_URL and shared engine. Never runs automatically - only when explicitly invoked with --apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import Engine, inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine
from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB

TABLE_NAME = FireEventMLAssessmentDB.__tablename__
# (column name, cross-dialect SQL type)
COLUMNS: tuple[tuple[str, str], ...] = (
    ("policy_version", "VARCHAR"),
    ("policy_status", "VARCHAR"),
    ("history_available", "BOOLEAN"),
    ("satellite_pass_count", "INTEGER"),
    ("current_satellite_pixel_count", "INTEGER"),
)


def missing_columns(engine: Engine) -> tuple[str, ...]:
    inspector = inspect(engine)
    if not inspector.has_table(TABLE_NAME):
        raise RuntimeError(f"{TABLE_NAME} does not exist; run init_db() first.")
    present = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
    return tuple(name for name, _ in COLUMNS if name not in present)


def migrate(engine: Engine | None = None) -> tuple[str, ...]:
    """Add every missing column. Returns the names it added (empty when the schema was already current)."""
    engine = engine or get_engine()
    missing = missing_columns(engine)
    types = dict(COLUMNS)
    with engine.begin() as connection:
        for name in missing:
            connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {name} {types[name]} NULL"))
    return missing


def row_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="Inspect schema without altering it.")
    mode.add_argument("--apply", action="store_true", help="Add the missing nullable columns.")
    args = parser.parse_args(argv)

    engine = get_engine()
    missing = missing_columns(engine)
    count_before = row_count(engine)
    print(f"{TABLE_NAME}: missing columns = {list(missing) or 'none'} (row_count={count_before})")
    if args.inspect_only or not missing:
        return 0

    migrate(engine)
    if row_count(engine) != count_before:
        raise RuntimeError("Row count changed during migration; aborting.")
    if missing_columns(engine):
        raise RuntimeError("Migration completed but some columns are still missing.")
    print("Migration applied successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
