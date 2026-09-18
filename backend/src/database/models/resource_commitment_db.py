"""SQLAlchemy ORM model for persistent resource ownership (Stage 1).

`resource_id` is the PRIMARY KEY (not a surrogate id): this is the exact
mechanism that guarantees, at the database level, "one resource -> at most
one current commitment" - it is a final invariant that holds even if
application-level row locking (see ResponsePlanActivationService) were ever
bypassed or buggy. Deliberately global-ready (Stage 2+): nothing here scopes
one commitment row to belonging to a single-FireEvent-only transaction -
`fire_event_id` is just a column like any other, so a future global
activation could write commitment rows for several different FireEvents
within one transaction without any schema change.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.database.base import Base
from src.models.dispatch_state import DispatchState

# Existing (pre-Stage-6) commitment rows predate the dispatch concept. The
# migration backfills them to DISPATCHED (see scripts/migrate_dispatch_state.py)
# rather than PLANNED: a legacy-activated commitment already represents this
# project's most operationally-committed prior state, so treating it as a
# hard lock is the conservative/safe direction - it never silently makes an
# already-committed resource freely reassignable by the new global optimizer.
DEFAULT_DISPATCH_STATE = DispatchState.DISPATCHED


class ResourceCommitmentDB(Base):
    """Current ownership: this resource is committed to this FireEvent's current ResponsePlan."""

    __tablename__ = "resource_commitments"

    resource_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("firefighting_resources.id"),
        primary_key=True,
    )
    fire_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("fire_events.id"),
        nullable=False,
        index=True,
    )
    response_plan_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("response_plans.id"),
        nullable=False,
        index=True,
    )
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dispatch_state: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=DEFAULT_DISPATCH_STATE.value,
        server_default=DEFAULT_DISPATCH_STATE.value,
    )
