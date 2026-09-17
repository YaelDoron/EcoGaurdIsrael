"""API transport contract for US 6.1 (`GET /api/v1/fire-events/active`).

These Pydantic models are the HTTP response shape only - distinct from the
read-layer dataclasses in `src.models.active_fire_events`
(`ActiveFireEventsResult` etc.), which `ActiveFireEventsService` (Task 2)
already returns. `to_active_fire_events_response` is the one place that
converts between the two; it is a pure field-by-field copy (tuple -> list,
enum -> its `.value` via Pydantic's own enum serialization) and performs no
calculation of its own.

`Optional[X]` is used instead of `X | None` throughout because Pydantic
resolves annotations at class-definition time and the `X | None` (PEP 604)
syntax is not evaluable on this project's Python 3.9 runtime for arbitrary
types (unlike the plain dataclasses in src.models, whose annotations are
never resolved at runtime).

There is deliberately no `area_name` field - see src/models/active_fire_events.py
for why (FireEvent persists no trustworthy area/location label); this layer
does not fabricate one just to match an old draft contract.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.models.active_fire_events import (
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel


class ActiveFireEventSeverityResponse(BaseModel):
    """A FireEvent's latest persisted severity assessment, as sent over HTTP."""

    assessment_id: int
    status: FireSeverityAssessmentStatus
    score: Optional[float]
    level: Optional[FireSeverityLevel]
    assessed_at: datetime


class ActiveFireEventResponse(BaseModel):
    """One currently-active FireEvent, as sent over HTTP."""

    fire_event_id: int
    status: FireEventStatus
    latitude: float
    longitude: float
    detection_confidence: float
    detected_at: datetime
    updated_at: datetime
    severity: Optional[ActiveFireEventSeverityResponse]


class ActiveFireEventsResponse(BaseModel):
    """Response body for `GET /api/v1/fire-events/active`."""

    as_of: datetime
    items: list[ActiveFireEventResponse]


def to_active_fire_events_response(result: ActiveFireEventsResult) -> ActiveFireEventsResponse:
    """Map the Task 2 read model to this endpoint's response contract, preserving order."""
    return ActiveFireEventsResponse(
        as_of=result.as_of,
        items=[_to_event_response(item) for item in result.items],
    )


def _to_event_response(item: ActiveFireEventSummary) -> ActiveFireEventResponse:
    return ActiveFireEventResponse(
        fire_event_id=item.fire_event_id,
        status=item.status,
        latitude=item.latitude,
        longitude=item.longitude,
        detection_confidence=item.detection_confidence,
        detected_at=item.detected_at,
        updated_at=item.updated_at,
        severity=_to_severity_response(item.severity) if item.severity is not None else None,
    )


def _to_severity_response(severity: ActiveFireEventSeveritySummary) -> ActiveFireEventSeverityResponse:
    return ActiveFireEventSeverityResponse(
        assessment_id=severity.assessment_id,
        status=severity.status,
        score=severity.score,
        level=severity.level,
        assessed_at=severity.assessed_at,
    )
