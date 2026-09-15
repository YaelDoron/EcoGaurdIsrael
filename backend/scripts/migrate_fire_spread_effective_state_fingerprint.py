"""Add nullable fire-spread effective-state fingerprint column when absent.

Existing rows intentionally remain NULL. A legacy NULL means "previous
effective state unknown"; refresh orchestration will create one new prediction
with a fingerprint before normal no-op comparison is possible.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402

TABLE_NAME = "fire_spread_predictions"
COLUMN_NAME = "effective_state_fingerprint"


def migrate() -> bool:
    """Apply the safe nullable-column migration. Returns True if changed."""
    engine = get_engine()
    with engine.begin() as connection:
        columns = {column["name"] for column in inspect(connection).get_columns(TABLE_NAME)}
        if COLUMN_NAME in columns:
            return False
        connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR(64) NULL"))
        return True


def main() -> int:
    changed = migrate()
    print(f"{TABLE_NAME}.{COLUMN_NAME}: {'added' if changed else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
