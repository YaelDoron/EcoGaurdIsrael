"""Add nullable fire-spread effective-state fingerprint column when absent.

Existing rows intentionally remain NULL. A legacy NULL means "previous
effective state unknown"; refresh orchestration will create one new prediction
with a fingerprint before normal no-op comparison is possible.

Brings the live schema to exact parity with the ORM
(src/database/models/fire_spread_prediction_db.py):
`effective_state_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)`
-- both the VARCHAR(64) NULL column and its declared index. Column and index
are added independently and idempotently (each is introspected before being
added), so running this script when only one of the two is already present
(or when both already are) is safe.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402

TABLE_NAME = "fire_spread_predictions"
COLUMN_NAME = "effective_state_fingerprint"
INDEX_NAME = "ix_fire_spread_predictions_effective_state_fingerprint"


def migrate() -> tuple[bool, bool]:
    """Apply the safe nullable-column + index migration.

    Returns (column_added, index_added).
    """
    engine = get_engine()
    with engine.begin() as connection:
        inspector = inspect(connection)
        columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
        column_added = COLUMN_NAME not in columns
        if column_added:
            connection.execute(text(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR(64) NULL"))

        # Re-inspect after a possible ALTER rather than trusting the
        # pre-ALTER index snapshot, matching the ORM's index=True exactly.
        indexes = {index["name"] for index in inspect(connection).get_indexes(TABLE_NAME)}
        index_added = INDEX_NAME not in indexes
        if index_added:
            connection.execute(text(f"CREATE INDEX {INDEX_NAME} ON {TABLE_NAME} ({COLUMN_NAME})"))

        return column_added, index_added


def main() -> int:
    column_added, index_added = migrate()
    print(f"{TABLE_NAME}.{COLUMN_NAME}: {'added' if column_added else 'already present'}")
    print(f"{INDEX_NAME}: {'added' if index_added else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
