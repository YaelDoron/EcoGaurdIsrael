"""Add a nullable news wildfire-signal-strength column to wildfire_reports.

Task 4 (Fire Detection ML V3 evidence enrichment): NewsMonitoringAgent's LLM
now derives a structured NONE/WEAK/MODERATE/STRONG signal for how strongly a
news article's own text supports an active wildfire
(see src/models/news_wildfire_signal_strength.py). This migration adds the
column WildfireReport/NewsRepository need to persist it.

The column is nullable and NEVER backfilled: every row saved before this
signal existed has no real value to record. NULL means "unknown", not
"none" - fabricating "none" for historical rows would misrepresent an
absence of analysis as a real negative finding.

Usage:
    python -m scripts.migrate_add_news_wildfire_signal_strength --inspect-only
    python -m scripts.migrate_add_news_wildfire_signal_strength --apply

Uses the project's existing DATABASE_URL configuration and shared SQLAlchemy
engine. Never drops/recreates the table and never writes to existing rows.
This script itself does not run automatically against the configured Neon
database - only when explicitly invoked with --apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine

TABLE_NAME = "wildfire_reports"
COLUMN_NAME = "wildfire_signal_strength"


def column_exists(engine: Engine) -> bool:
    """Return whether COLUMN_NAME already exists on TABLE_NAME. Raises if the table itself is missing."""
    inspector = inspect(engine)
    if not inspector.has_table(TABLE_NAME):
        raise RuntimeError(f"{TABLE_NAME} does not exist; run init_db() first.")
    return any(column["name"] == COLUMN_NAME for column in inspector.get_columns(TABLE_NAME))


def migrate() -> bool:
    """Add the nullable column if missing. Returns True if it added the column, False if already present.

    Uses plain, cross-dialect ADD COLUMN syntax (unlengthed VARCHAR, like
    every other String-typed column on this table) so this is safe on both
    SQLite (unit tests) and PostgreSQL (Neon).
    """
    engine = get_engine()
    if column_exists(engine):
        return False
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR NULL"))
    return True


def row_count(engine: Engine) -> int:
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="Inspect schema without altering it.")
    mode.add_argument("--apply", action="store_true", help="Apply the safe in-place migration if needed.")
    args = parser.parse_args(argv)

    engine = get_engine()
    exists_before = column_exists(engine)
    count_before = row_count(engine)
    _print_status(exists_before, count_before)

    if args.inspect_only:
        return 0

    if exists_before:
        print(f"{TABLE_NAME}.{COLUMN_NAME}: already present, nothing to do.")
        return 0

    created = migrate()

    count_after = row_count(engine)
    if count_after != count_before:
        raise RuntimeError(f"Row count changed during migration ({count_before} -> {count_after}); aborting.")

    exists_after = column_exists(engine)
    if not exists_after:
        raise RuntimeError("Migration completed but the column still does not exist.")

    print("Migration applied successfully." if created else "Migration not needed: column already present.")
    _print_status(exists_after, count_after)
    return 0


def _print_status(exists: bool, count: int) -> None:
    state = "present" if exists else "MISSING"
    print(f"{TABLE_NAME}.{COLUMN_NAME}: {state} (row_count={count})")


if __name__ == "__main__":
    raise SystemExit(main())
