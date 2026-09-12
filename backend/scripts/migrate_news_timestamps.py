"""Inspect and migrate wildfire_reports timestamp columns to TIMESTAMPTZ.

Usage:
    python scripts/migrate_news_timestamps.py --inspect-only
    python scripts/migrate_news_timestamps.py --apply

The script uses the project's existing DATABASE_URL configuration and shared
SQLAlchemy engine. It never drops or recreates wildfire_reports.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
import sys
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine

TABLE_NAME = "wildfire_reports"
TIMESTAMP_COLUMNS = ("published_at", "fetched_at")
_ISO_OFFSET_WITHOUT_MINUTES_RE = re.compile(r"([+-]\d{2})$")


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    udt_name: str
    is_nullable: bool

    @property
    def is_timestamptz(self) -> bool:
        return self.udt_name == "timestamptz"

    @property
    def is_timestamp_without_timezone(self) -> bool:
        return self.udt_name == "timestamp"


@dataclass(frozen=True)
class InvalidTimestampValue:
    row_id: Any
    source_url: str | None
    column_name: str
    value: str
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-only", action="store_true", help="Inspect schema/data without altering it.")
    mode.add_argument("--apply", action="store_true", help="Apply the safe in-place migration if needed.")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as connection:
        summary = inspect_schema_and_data(connection)
        print_summary(summary)

        if args.inspect_only:
            return

        validate_migration_is_safe(summary)
        if schema_is_correct(summary["columns"]):
            print("Migration not needed: timestamp columns already have the intended schema.")
            return

        apply_migration(connection, summary["columns"])
        migrated_summary = inspect_schema_and_data(connection)
        validate_migration_is_safe(migrated_summary)
        if not schema_is_correct(migrated_summary["columns"]):
            raise RuntimeError("Migration completed but schema is still not in the intended state.")

        print("Migration applied successfully.")
        print_summary(migrated_summary)


def inspect_schema_and_data(connection: Connection) -> dict[str, Any]:
    table_exists = connection.execute(
        text(
            "SELECT EXISTS ("
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = :table_name"
            ")"
        ),
        {"table_name": TABLE_NAME},
    ).scalar_one()

    if not table_exists:
        return {
            "table_exists": False,
            "columns": {},
            "row_count": 0,
            "null_counts": {},
            "empty_string_counts": {},
            "format_counts": {},
            "invalid_values": [],
        }

    column_rows = connection.execute(
        text(
            "SELECT column_name, data_type, udt_name, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "AND table_name = :table_name "
            "AND column_name IN ('published_at', 'fetched_at')"
        ),
        {"table_name": TABLE_NAME},
    ).mappings()
    columns = {
        row["column_name"]: ColumnInfo(
            name=row["column_name"],
            data_type=row["data_type"],
            udt_name=row["udt_name"],
            is_nullable=row["is_nullable"] == "YES",
        )
        for row in column_rows
    }

    row_count = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar_one()
    values = connection.execute(
        text(
            "SELECT id, source_url, published_at::text AS published_at, fetched_at::text AS fetched_at "
            f"FROM {TABLE_NAME} "
            "ORDER BY id"
        )
    ).mappings()

    null_counts = {column: 0 for column in TIMESTAMP_COLUMNS}
    empty_string_counts = {column: 0 for column in TIMESTAMP_COLUMNS}
    format_counts = {column: {} for column in TIMESTAMP_COLUMNS}
    invalid_values: list[InvalidTimestampValue] = []

    for row in values:
        for column in TIMESTAMP_COLUMNS:
            value = row[column]
            if value is None:
                null_counts[column] += 1
                continue

            if value == "":
                empty_string_counts[column] += 1
                invalid_values.append(
                    InvalidTimestampValue(
                        row_id=row["id"],
                        source_url=row["source_url"],
                        column_name=column,
                        value=value,
                        reason="empty string is not a safe timestamp",
                    )
                )
                continue

            format_name = classify_timestamp_format(value)
            format_counts[column][format_name] = format_counts[column].get(format_name, 0) + 1

            if not can_parse_timestamp(value):
                invalid_values.append(
                    InvalidTimestampValue(
                        row_id=row["id"],
                        source_url=row["source_url"],
                        column_name=column,
                        value=value,
                        reason="not parseable by the migration validator",
                    )
                )

    return {
        "table_exists": True,
        "columns": columns,
        "row_count": row_count,
        "null_counts": null_counts,
        "empty_string_counts": empty_string_counts,
        "format_counts": format_counts,
        "invalid_values": invalid_values,
    }


def classify_timestamp_format(value: str) -> str:
    if "T" in value and ("+" in value or value.endswith("Z")):
        return "iso8601_with_timezone"
    if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}([+-]\d{2}(:\d{2})?)?$", value):
        return "postgres_timestamp_text"
    if "," in value:
        return "rfc2822_like"
    return "other"


def can_parse_timestamp(value: str) -> bool:
    try:
        parse_timestamp(value)
        return True
    except ValueError:
        return False


def parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise ValueError("empty timestamp")

    iso_candidate = _ISO_OFFSET_WITHOUT_MINUTES_RE.sub(r"\1:00", normalized.replace("Z", "+00:00"))
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def validate_migration_is_safe(summary: dict[str, Any]) -> None:
    if not summary["table_exists"]:
        raise RuntimeError("wildfire_reports table does not exist; run init_db() first.")

    columns = summary["columns"]
    missing_columns = sorted(set(TIMESTAMP_COLUMNS) - set(columns))
    if missing_columns:
        raise RuntimeError(f"wildfire_reports is missing timestamp columns: {missing_columns}")

    invalid_values = summary["invalid_values"]
    if invalid_values:
        details = "\n".join(
            f"- id={value.row_id} source_url={value.source_url!r} "
            f"{value.column_name}={value.value!r}: {value.reason}"
            for value in invalid_values[:20]
        )
        raise RuntimeError(f"Unsafe timestamp values block migration:\n{details}")

    if summary["null_counts"]["fetched_at"] > 0:
        raise RuntimeError(
            f"Cannot enforce fetched_at NOT NULL: {summary['null_counts']['fetched_at']} rows are NULL."
        )


def schema_is_correct(columns: dict[str, ColumnInfo]) -> bool:
    published_at = columns.get("published_at")
    fetched_at = columns.get("fetched_at")
    return (
        published_at is not None
        and fetched_at is not None
        and published_at.is_timestamptz
        and fetched_at.is_timestamptz
        and published_at.is_nullable
        and not fetched_at.is_nullable
    )


def apply_migration(connection: Connection, columns: dict[str, ColumnInfo]) -> None:
    published_at = columns["published_at"]
    fetched_at = columns["fetched_at"]

    if not published_at.is_timestamptz:
        published_using = _using_expression(published_at, "published_at")
        connection.execute(
            text(
                f"ALTER TABLE {TABLE_NAME} "
                f"ALTER COLUMN published_at TYPE TIMESTAMP WITH TIME ZONE "
                f"USING {published_using}"
            )
        )

    connection.execute(text(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN published_at DROP NOT NULL"))

    if not fetched_at.is_timestamptz:
        fetched_using = _using_expression(fetched_at, "fetched_at")
        connection.execute(
            text(
                f"ALTER TABLE {TABLE_NAME} "
                f"ALTER COLUMN fetched_at TYPE TIMESTAMP WITH TIME ZONE "
                f"USING {fetched_using}"
            )
        )

    connection.execute(text(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN fetched_at SET NOT NULL"))


def _using_expression(column_info: ColumnInfo, column_name: str) -> str:
    if column_info.is_timestamp_without_timezone:
        return f"{column_name} AT TIME ZONE 'UTC'"
    return f"{column_name}::timestamptz"


def print_summary(summary: dict[str, Any]) -> None:
    print(f"table_exists={summary['table_exists']}")
    if not summary["table_exists"]:
        return

    print(f"row_count={summary['row_count']}")
    for column_name in TIMESTAMP_COLUMNS:
        column = summary["columns"].get(column_name)
        if column is None:
            print(f"{column_name}: MISSING")
        else:
            nullable = "YES" if column.is_nullable else "NO"
            print(
                f"{column_name}: data_type={column.data_type}, "
                f"udt_name={column.udt_name}, is_nullable={nullable}"
            )
        print(f"{column_name}: null_count={summary['null_counts'].get(column_name)}")
        print(f"{column_name}: empty_string_count={summary['empty_string_counts'].get(column_name)}")
        print(f"{column_name}: format_counts={summary['format_counts'].get(column_name)}")

    print(f"invalid_value_count={len(summary['invalid_values'])}")
    for invalid in summary["invalid_values"][:20]:
        print(
            f"invalid: id={invalid.row_id} source_url={invalid.source_url!r} "
            f"{invalid.column_name}={invalid.value!r} reason={invalid.reason}"
        )


if __name__ == "__main__":
    main()
