"""Create the resource_commitments table if absent (Stage 1 of the Global
Multi-Incident Optimizer refactor).

Idempotent: introspects for the table before creating it, so running this
script when it already exists is a safe no-op. Brings the live schema to
parity with the ORM (src/database/models/resource_commitment_db.py) via
SQLAlchemy Core's own table-creation DDL (`Table.create(checkfirst=True)`),
matching how every OTHER new table in this project is created (via
init_db()'s Base.metadata.create_all(), which this table is already
registered for) - this script exists only to create THIS ONE table
directly, for a focused, reviewable, single-purpose migration step,
without touching any other table init_db() would also ensure.

No backfill: resource_commitments starts empty. FireEvents with an
already-current ResponsePlan predating this migration get no synthetic
commitment rows - see CrossEventReservedResourceResolver's own docstring
for why a blind backfill was deliberately not attempted (a plan's
persisted actions cannot tell us whether the resources they reference are
still operationally valid "right now"; fabricating commitments from stale
history risks reserving resources that are actually free, or missing ones
that should be reserved, either of which is worse than the documented,
intentional temporary gap the Stage-0 compatibility fallback already
covers). Historical ResponsePlans/ResponseActions are never touched.
"""
from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import get_engine  # noqa: E402
from src.database.models.resource_commitment_db import ResourceCommitmentDB  # noqa: E402

TABLE_NAME = ResourceCommitmentDB.__tablename__


def table_exists(connection) -> bool:
    return inspect(connection).has_table(TABLE_NAME)


def migrate() -> bool:
    """Create resource_commitments if it does not already exist. Returns whether it was created."""
    engine = get_engine()
    with engine.begin() as connection:
        if table_exists(connection):
            return False
        ResourceCommitmentDB.__table__.create(bind=connection, checkfirst=True)
        return True


def main() -> int:
    created = migrate()
    print(f"{TABLE_NAME}: {'created' if created else 'already present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
