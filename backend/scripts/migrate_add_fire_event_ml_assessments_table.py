"""Create the fire_event_ml_assessments table if absent (Task 5 runtime ML integration).

Entirely additive: one new table, one row per FireEvent (upserted on
reevaluation, not a history log), with a unique FK to fire_events.id
(ON DELETE CASCADE). FireEventDB/fire_events itself is not touched by this
migration. Historical FireEvents simply have no matching row here - never
backfilled, never a fabricated "ML not evaluated" row inserted retroactively.

Idempotent: introspects for the table before creating it via SQLAlchemy
Core's own table-creation DDL (matching scripts.migrate_resource_commitments's
precedent), so running this script when the table already exists is a safe
no-op that never alters existing fire_events rows or evidence links.

Usage:
    python -m scripts.migrate_add_fire_event_ml_assessments_table --inspect-only
    python -m scripts.migrate_add_fire_event_ml_assessments_table --apply

Uses the project's existing DATABASE_URL configuration and shared SQLAlchemy
engine. This script does not run automatically against the configured Neon
database - only when explicitly invoked with --apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_ml_assessment_db import FireEventMLAssessmentDB

TABLE_NAME = FireEventMLAssessmentDB.__tablename__
_PARENT_TABLE_NAME = FireEventDB.__tablename__


def table_exists(engine) -> bool:
    with engine.connect() as connection:
        return inspect(connection).has_table(TABLE_NAME)


def migrate() -> bool:
    """Create fire_event_ml_assessments if absent. Returns whether it was created."""
    engine = get_engine()
    with engine.begin() as connection:
        if inspect(connection).has_table(TABLE_NAME):
            return False
        if not inspect(connection).has_table(_PARENT_TABLE_NAME):
            raise RuntimeError(f"{_PARENT_TABLE_NAME} does not exist; run init_db() first.")
        FireEventMLAssessmentDB.__table__.create(bind=connection, checkfirst=True)
        return True


def _row_count(engine) -> int | None:
    if not table_exists(engine):
        return None
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()


def _fire_events_row_count(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT COUNT(*) FROM {_PARENT_TABLE_NAME}")).scalar_one()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="Inspect schema without altering it.")
    mode.add_argument("--apply", action="store_true", help="Create the table if it does not already exist.")
    args = parser.parse_args(argv)

    engine = get_engine()
    fire_events_before = _fire_events_row_count(engine)
    _print_status(engine, fire_events_before)

    if args.inspect_only:
        return 0

    created = migrate()

    fire_events_after = _fire_events_row_count(engine)
    if fire_events_after != fire_events_before:
        raise RuntimeError(
            f"{_PARENT_TABLE_NAME} row count changed during migration "
            f"({fire_events_before} -> {fire_events_after}); aborting."
        )

    print(f"{TABLE_NAME}: {'created' if created else 'already present'}")
    _print_status(engine, fire_events_after)
    return 0


def _print_status(engine, fire_events_row_count: int) -> None:
    count = _row_count(engine)
    state = f"present (row_count={count})" if count is not None else "MISSING"
    print(f"{TABLE_NAME}: {state}")
    print(f"{_PARENT_TABLE_NAME}: row_count={fire_events_row_count} (unaffected by this migration)")


if __name__ == "__main__":
    raise SystemExit(main())
