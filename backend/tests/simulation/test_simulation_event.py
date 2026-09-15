"""Tests for simulation event definitions."""

import pytest

from src.models.resource_status import ResourceStatus
from src.simulation import SimulationEvent, SimulationEventType, SimulationResourceStatusChange


def test_valid_event_construction():
    event = SimulationEvent(
        offset_seconds=20,
        event_type=SimulationEventType.SATELLITE,
        incident_id="incident-carmel-01",
        source_event_index=0,
    )

    assert event.offset_seconds == 20
    assert event.event_type is SimulationEventType.SATELLITE
    assert event.incident_id == "incident-carmel-01"
    assert event.source_event_index == 0


def test_event_types_are_represented():
    assert SimulationEventType.WEATHER.value == "weather"
    assert SimulationEventType.SATELLITE.value == "satellite"
    assert SimulationEventType.NEWS.value == "news"
    assert SimulationEventType.RESOURCE_STATUS.value == "resource_status"


def test_resource_status_event_requires_resource_payload():
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=60,
            event_type=SimulationEventType.RESOURCE_STATUS,
            incident_id="incident-carmel-01",
            source_event_index=0,
        )


def test_resource_status_event_accepts_existing_resource_status_enum():
    event = SimulationEvent(
        offset_seconds=60,
        event_type=SimulationEventType.RESOURCE_STATUS,
        incident_id="incident-carmel-01",
        source_event_index=0,
        resource_status_change=SimulationResourceStatusChange(
            new_status=ResourceStatus.UNAVAILABLE,
            resource_id="TRUCK-A",
        ),
    )

    assert event.resource_status_change.new_status is ResourceStatus.UNAVAILABLE
    assert event.resource_status_change.resource_id == "TRUCK-A"


def test_non_resource_event_rejects_resource_payload():
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=0,
            event_type=SimulationEventType.WEATHER,
            incident_id="incident-carmel-01",
            source_event_index=0,
            resource_status_change=SimulationResourceStatusChange(
                new_status=ResourceStatus.UNAVAILABLE,
                resource_id="TRUCK-A",
            ),
        )


@pytest.mark.parametrize("offset_seconds", [-1, None, "10", True])
def test_negative_or_invalid_offsets_are_rejected(offset_seconds):
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=offset_seconds,
            event_type=SimulationEventType.WEATHER,
            incident_id="incident-carmel-01",
            source_event_index=0,
        )


def test_invalid_event_type_is_rejected():
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=0,
            event_type="weather",
            incident_id="incident-carmel-01",
            source_event_index=0,
        )


@pytest.mark.parametrize("incident_id", ["", " ", None, 123])
def test_invalid_incident_ids_are_rejected(incident_id):
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=0,
            event_type=SimulationEventType.WEATHER,
            incident_id=incident_id,
            source_event_index=0,
        )


@pytest.mark.parametrize("source_event_index", [-1, None, "0", True])
def test_invalid_source_event_indexes_are_rejected(source_event_index):
    with pytest.raises(ValueError):
        SimulationEvent(
            offset_seconds=0,
            event_type=SimulationEventType.WEATHER,
            incident_id="incident-carmel-01",
            source_event_index=source_event_index,
        )
