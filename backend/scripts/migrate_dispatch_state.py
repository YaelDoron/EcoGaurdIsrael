"""Add resource_commitments.dispatch_state when absent (Stage 6 of the
Global Multi-Incident Optimizer refactor).

Idempotent: introspects for the column before adding it, so running this
script when it already exists is a safe no-op.

Backfill: existing (pre-Stage-6) commitment rows predate the dispatch
concept entirely - there is no historical record of whether a legacy
per-event activation "really" represents a resource already physically
dispatched. Per this project's established migration convention (no
fabricated backfill), this migration does not invent a false historical
fact. Instead it applies one explicit, documented, CONSERVATIVE policy
default: existing rows are backfilled to DISPATCHED, not PLANNED. This is
the safe direction - a legacy-activated commitment already represents this
project's most operationally-committed prior state (it is the resource set
a real ResponsePlan was actually activated with), so treating it as a hard
lock means the new global optimizer can never silently reassign an
already-committed resource elsewhere on its very first run, purely as a
side effect of this migration. See src/database/models/resource_commitment_db.py
(DEFAULT_DISPATCH_STATE) for the same rationale at the ORM-default level.

Column added NOT NULL with a server-side default so the single ALTER TABLE
statement both adds the column and backfills every existing row in one
step (safe on Postgres; matches how DEFAULT_DISPATCH_STATE is expressed at
the ORM layer, so newly-created rows via raw INSERTs would also default
consistently even without an application-level value).
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402
from src.database.models.resource_commitment_db import DEFAULT_DISPATCH_STATE  # noqa: E402

TABLE_NAME = "resource_commitments"
COLUMN_NAME = "dispatch_state"


def column_exists(connection) -> bool:
    existing_columns = {column["name"] for column in inspect(connection).get_columns(TABLE_NAME)}
    return COLUMN_NAME in existing_columns


def migrate() -> bool:
    """Add resource_commitments.dispatch_state if missing. Returns whether it was added."""
    engine = get_engine()
    with engine.begin() as connection:
        if column_exists(connection):
            return False
        connection.execute(
            text(
                f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} VARCHAR NOT NULL "
                f"DEFAULT '{DEFAULT_DISPATCH_STATE.value}'"
            )
        )
        return True


def main() -> int:
    added = migrate()
    print(f"{TABLE_NAME}.{COLUMN_NAME}: {'added' if added else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
