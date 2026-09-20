"""Tests for the US 6.1 Task 3 API schema/mapper (`src.api.schemas.active_fire_events`).

Exercises `to_active_fire_events_response` directly - no HTTP, no service,
no DB - to isolate the mapping step itself from the router tests in
tests/api/test_fire_events_router.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.api.schemas.active_fire_events import (
    ActiveFireEventResponse,
    ActiveFireEventSeverityResponse,
    ActiveFireEventsResponse,
    to_active_fire_events_response,
)
from src.models.active_fire_events import (
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel

DETECTED_AT = datetime(2026, 9, 17, 13, 20, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=8)
ASSESSED_AT = DETECTED_AT + timedelta(minutes=7)
AS_OF = DETECTED_AT + timedelta(minutes=40)


def make_severity(**overrides) -> ActiveFireEventSeveritySummary:
    values = dict(
        assessment_id=44,
        status=FireSeverityAssessmentStatus.VALID,
        score=81.4,
        level=FireSeverityLevel.CRITICAL,
        assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return ActiveFireEventSeveritySummary(**values)


def make_event(**overrides) -> ActiveFireEventSummary:
    values = dict(
        fire_event_id=12,
        status=FireEventStatus.CONFIRMED,
        latitude=32.731,
        longitude=35.046,
        detection_confidence=0.91,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        created_at=UPDATED_AT,
        severity=None,
    )
    values.update(overrides)
    return ActiveFireEventSummary(**values)


def test_maps_empty_result():
    result = ActiveFireEventsResult(as_of=AS_OF, items=())

    response = to_active_fire_events_response(result)

    assert isinstance(response, ActiveFireEventsResponse)
    assert response.as_of == AS_OF
    assert response.items == []


def test_maps_event_fields_one_to_one():
    event = make_event()
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    item = response.items[0]
    assert isinstance(item, ActiveFireEventResponse)
    assert item.fire_event_id == event.fire_event_id
    assert item.status is event.status
    assert item.latitude == event.latitude
    assert item.longitude == event.longitude
    assert item.detection_confidence == event.detection_confidence
    assert item.detected_at == event.detected_at
    assert item.updated_at == event.updated_at
    assert item.created_at == event.created_at


def test_maps_severity_fields_one_to_one():
    severity = make_severity()
    event = make_event(severity=severity)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    mapped = response.items[0].severity
    assert isinstance(mapped, ActiveFireEventSeverityResponse)
    assert mapped.assessment_id == severity.assessment_id
    assert mapped.status is severity.status
    assert mapped.score == severity.score
    assert mapped.level is severity.level
    assert mapped.assessed_at == severity.assessed_at


def test_maps_missing_severity_to_none():
    event = make_event(severity=None)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert response.items[0].severity is None


def test_converts_items_tuple_to_list():
    result = ActiveFireEventsResult(as_of=AS_OF, items=(make_event(),))

    response = to_active_fire_events_response(result)

    assert isinstance(response.items, list)


def test_preserves_result_item_order():
    first = make_event(fire_event_id=9)
    second = make_event(fire_event_id=1)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(first, second))

    response = to_active_fire_events_response(result)

    assert [item.fire_event_id for item in response.items] == [9, 1]


def test_status_enum_serializes_to_stable_lowercase_value():
    event = make_event(status=FireEventStatus.SUSPECTED)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert response.model_dump(mode="json")["items"][0]["status"] == "suspected"


def test_severity_level_enum_serializes_to_stable_lowercase_value():
    event = make_event(severity=make_severity(level=FireSeverityLevel.LOW))
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert response.model_dump(mode="json")["items"][0]["severity"]["level"] == "low"


def test_no_area_name_field_on_response():
    event = make_event()
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert "area_name" not in response.items[0].model_dump()


def test_maps_location_name_when_present():
    event = make_event(location_name="Carmel Demo Area")
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert response.items[0].location_name == "Carmel Demo Area"


def test_maps_location_name_none_when_absent():
    event = make_event(location_name=None)
    result = ActiveFireEventsResult(as_of=AS_OF, items=(event,))

    response = to_active_fire_events_response(result)

    assert response.items[0].location_name is None
