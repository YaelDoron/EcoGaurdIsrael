"""Add optional `location_name` columns to satellite_hotspots and fire_events.

Both columns are brand-new, nullable, and require no data backfill/transform:
existing rows simply get NULL (the correct value for "no trusted provenance
existed before this change" - see this task's report on the FireEvent
location-name provenance chain). This is the smallest safe schema change:
a plain `ALTER TABLE ... ADD COLUMN ... NULL`, never touching existing data.

Usage:
    python scripts/migrate_add_location_name_columns.py --inspect-only
    python scripts/migrate_add_location_name_columns.py --apply

The script uses the project's existing DATABASE_URL configuration and shared
SQLAlchemy engine. It never drops or recreates either table.
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

_TARGETS = (
    ("satellite_hotspots", "location_name"),
    ("fire_events", "location_name"),
)


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
        summary = inspect_schema(connection)
        print_summary(summary)

        if args.inspect_only:
            return

        if schema_is_correct(summary):
            print("Migration not needed: both columns already exist.")
            return

        apply_migration(connection, summary)
        migrated_summary = inspect_schema(connection)
        if not schema_is_correct(migrated_summary):
            raise RuntimeError("Migration completed but schema is still not in the intended state.")

        print("Migration applied successfully.")
        print_summary(migrated_summary)


def inspect_schema(connection: Connection) -> dict[str, ColumnInfo | None]:
    result: dict[str, ColumnInfo | None] = {}
    for table_name, column_name in _TARGETS:
        key = f"{table_name}.{column_name}"
        table_exists = connection.execute(
            text(
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :table_name"
                ")"
            ),
            {"table_name": table_name},
        ).scalar_one()
        if not table_exists:
            raise RuntimeError(f"{table_name} does not exist; run init_db() first.")

        row = connection.execute(
            text(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :table_name AND column_name = :column_name"
            ),
            {"table_name": table_name, "column_name": column_name},
        ).mappings().one_or_none()
        result[key] = (
            ColumnInfo(
                table_name=table_name,
                column_name=column_name,
                data_type=row["data_type"],
                is_nullable=row["is_nullable"] == "YES",
            )
            if row is not None
            else None
        )
    return result


def schema_is_correct(summary: dict[str, ColumnInfo | None]) -> bool:
    return all(
        summary[f"{table_name}.{column_name}"] is not None and summary[f"{table_name}.{column_name}"].is_nullable
        for table_name, column_name in _TARGETS
    )


def apply_migration(connection: Connection, summary: dict[str, ColumnInfo | None]) -> None:
    for table_name, column_name in _TARGETS:
        if summary[f"{table_name}.{column_name}"] is not None:
            continue
        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} VARCHAR NULL"))


def print_summary(summary: dict[str, ColumnInfo | None]) -> None:
    for table_name, column_name in _TARGETS:
        column = summary[f"{table_name}.{column_name}"]
        if column is None:
            print(f"{table_name}.{column_name}: MISSING")
            continue
        nullable = "YES" if column.is_nullable else "NO"
        print(f"{table_name}.{column_name}: data_type={column.data_type}, is_nullable={nullable}")


if __name__ == "__main__":
    main()
