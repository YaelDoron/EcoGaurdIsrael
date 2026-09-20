"""Migrate satellite_hotspots.detected_at and weather_observations.timestamp
to TIMESTAMPTZ.

Root cause (see task report): both columns were declared as plain
`DateTime` (Postgres `TIMESTAMP WITHOUT TIME ZONE`), unlike
fire_danger_assessments.assessed_at / wildfire_reports.published_at/
fetched_at, which already use `TIMESTAMPTZ` (the latter migrated by this
script's sibling, migrate_news_timestamps.py). Every simulation-generated
value for both columns is constructed as an aware UTC datetime
(scenario_started_at defaults to datetime.now(timezone.utc)), so a naive
existing row is known to represent a UTC instant - this migration converts
it with `AT TIME ZONE 'UTC'`, never a blind reinterpretation.

Usage:
    python scripts/migrate_satellite_and_weather_timestamps.py --inspect-only
    python scripts/migrate_satellite_and_weather_timestamps.py --apply

The script uses the project's existing DATABASE_URL configuration and shared
SQLAlchemy engine. It never drops or recreates either table.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine

_TARGETS = (
    ("satellite_hotspots", "detected_at", False),
    ("weather_observations", "timestamp", False),
)


@dataclass(frozen=True)
class ColumnInfo:
    table_name: str
    column_name: str
    data_type: str
    udt_name: str
    is_nullable: bool

    @property
    def is_timestamptz(self) -> bool:
        return self.udt_name == "timestamptz"

    @property
    def is_timestamp_without_timezone(self) -> bool:
        return self.udt_name == "timestamp"


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
            print("Migration not needed: both columns already have the intended schema.")
            return

        apply_migration(connection, summary)
        migrated_summary = inspect_schema(connection)
        if not schema_is_correct(migrated_summary):
            raise RuntimeError("Migration completed but schema is still not in the intended state.")

        print("Migration applied successfully.")
        print_summary(migrated_summary)


def inspect_schema(connection: Connection) -> dict[str, ColumnInfo | None]:
    result: dict[str, ColumnInfo | None] = {}
    for table_name, column_name, _ in _TARGETS:
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
            result[key] = None
            continue

        row = connection.execute(
            text(
                "SELECT column_name, data_type, udt_name, is_nullable "
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
                udt_name=row["udt_name"],
                is_nullable=row["is_nullable"] == "YES",
            )
            if row is not None
            else None
        )
    return result


def schema_is_correct(summary: dict[str, ColumnInfo | None]) -> bool:
    return all(
        summary[f"{table_name}.{column_name}"] is not None
        and summary[f"{table_name}.{column_name}"].is_timestamptz
        for table_name, column_name, _ in _TARGETS
    )


def apply_migration(connection: Connection, summary: dict[str, ColumnInfo | None]) -> None:
    for table_name, column_name, nullable in _TARGETS:
        column = summary[f"{table_name}.{column_name}"]
        if column is None:
            raise RuntimeError(f"{table_name} is missing column {column_name}; run init_db() first.")
        if column.is_timestamptz:
            continue
        using_expression = _using_expression(column, column_name)
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ALTER COLUMN {column_name} TYPE TIMESTAMP WITH TIME ZONE "
                f"USING {using_expression}"
            )
        )
        if not nullable:
            connection.execute(text(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} SET NOT NULL"))


def _using_expression(column_info: ColumnInfo, column_name: str) -> str:
    if column_info.is_timestamp_without_timezone:
        return f"{column_name} AT TIME ZONE 'UTC'"
    return f"{column_name}::timestamptz"


def print_summary(summary: dict[str, ColumnInfo | None]) -> None:
    for table_name, column_name, _ in _TARGETS:
        column = summary[f"{table_name}.{column_name}"]
        if column is None:
            print(f"{table_name}.{column_name}: MISSING")
            continue
        nullable = "YES" if column.is_nullable else "NO"
        print(
            f"{table_name}.{column_name}: data_type={column.data_type}, "
            f"udt_name={column.udt_name}, is_nullable={nullable}"
        )


if __name__ == "__main__":
    main()
