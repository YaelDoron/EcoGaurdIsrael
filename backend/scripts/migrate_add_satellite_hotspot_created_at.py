"""Add a nullable `created_at` (TIMESTAMPTZ) column to satellite_hotspots.

Task 4 (Activity Feed "available_at" semantics): satellite_hotspots had no
persisted timestamp representing when EcoGuard itself ingested/persisted the
row (only `detected_at`, the satellite's own observation time). This adds
the smallest clean, timezone-aware column for that purpose.

The column is nullable and NEVER backfilled from `detected_at` or from
"now" at migration time: existing rows genuinely have no recorded ingestion
moment, and fabricating one (either by copying detected_at, or by stamping
every legacy row with the migration's own run time) would misrepresent
history. Existing rows simply get NULL; only new rows inserted after this
migration get a real value, written once by SatelliteHotspotRepository on
insert (never derived, never backfilled later).

Usage:
    python scripts/migrate_add_satellite_hotspot_created_at.py --inspect-only
    python scripts/migrate_add_satellite_hotspot_created_at.py --apply

The script uses the project's existing DATABASE_URL configuration and shared
SQLAlchemy engine. It never drops or recreates the table, and never writes
to existing rows.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from sqlalchemy import text
from sqlalchemy.engine import Connection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine

_TABLE_NAME = "satellite_hotspots"
_COLUMN_NAME = "created_at"


@dataclass(frozen=True)
class ColumnInfo:
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="Inspect schema without altering it.")
    mode.add_argument("--apply", action="store_true", help="Apply the safe in-place migration if needed.")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as connection:
        row_count_before = connection.execute(text(f"SELECT COUNT(*) FROM {_TABLE_NAME}")).scalar_one()
        column = inspect_schema(connection)
        print_summary(column, row_count_before)

        if args.inspect_only:
            return

        if schema_is_correct(column):
            print("Migration not needed: column already exists with the intended type.")
            return

        apply_migration(connection, column)
        migrated_column = inspect_schema(connection)
        if not schema_is_correct(migrated_column):
            raise RuntimeError("Migration completed but schema is still not in the intended state.")

        row_count_after = connection.execute(text(f"SELECT COUNT(*) FROM {_TABLE_NAME}")).scalar_one()
        if row_count_after != row_count_before:
            raise RuntimeError(
                f"Row count changed during migration ({row_count_before} -> {row_count_after}); aborting."
            )

        print("Migration applied successfully.")
        print_summary(migrated_column, row_count_after)


def inspect_schema(connection: Connection) -> ColumnInfo | None:
    table_exists = connection.execute(
        text(
            "SELECT EXISTS ("
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :table_name"
            ")"
        ),
        {"table_name": _TABLE_NAME},
    ).scalar_one()
    if not table_exists:
        raise RuntimeError(f"{_TABLE_NAME} does not exist; run init_db() first.")

    row = connection.execute(
        text(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name"
        ),
        {"table_name": _TABLE_NAME, "column_name": _COLUMN_NAME},
    ).mappings().one_or_none()
    if row is None:
        return None
    return ColumnInfo(
        table_name=_TABLE_NAME,
        column_name=_COLUMN_NAME,
        data_type=row["data_type"],
        is_nullable=row["is_nullable"] == "YES",
    )


def schema_is_correct(column: ColumnInfo | None) -> bool:
    return (
        column is not None
        and column.is_nullable
        and column.data_type == "timestamp with time zone"
    )


def apply_migration(connection: Connection, column: ColumnInfo | None) -> None:
    if column is not None:
        return
    connection.execute(text(f"ALTER TABLE {_TABLE_NAME} ADD COLUMN {_COLUMN_NAME} TIMESTAMPTZ NULL"))


def print_summary(column: ColumnInfo | None, row_count: int) -> None:
    if column is None:
        print(f"{_TABLE_NAME}.{_COLUMN_NAME}: MISSING (row_count={row_count})")
        return
    nullable = "YES" if column.is_nullable else "NO"
    print(f"{_TABLE_NAME}.{_COLUMN_NAME}: data_type={column.data_type}, is_nullable={nullable}, row_count={row_count}")


if __name__ == "__main__":
    main()
