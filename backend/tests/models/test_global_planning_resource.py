"""Tests for GlobalPlanningResource (Stage 3 of the Global Multi-Incident Optimizer refactor)."""
from __future__ import annotations

import pytest

from src.models.global_planning_resource import GlobalPlanningResource
from src.models.resource_status import ResourceStatus


def make_resource(**overrides) -> GlobalPlanningResource:
    defaults = dict(
        resource_id="R1",
        station_id="S1",
        station_name="Station 1",
        station_latitude=32.7,
        station_longitude=35.0,
        operational_status=ResourceStatus.AVAILABLE,
        current_commitment_fire_event_id=None,
        current_commitment_response_plan_id=None,
    )
    defaults.update(overrides)
    return GlobalPlanningResource(**defaults)


def test_freely_available_resource_is_assignable_and_uncommitted():
    resource = make_resource()
    assert resource.is_assignable is True
    assert resource.is_committed is False


def test_committed_resource_remains_assignable():
    resource = make_resource(
        operational_status=ResourceStatus.ASSIGNED,
        current_commitment_fire_event_id=5,
        current_commitment_response_plan_id=50,
    )
    assert resource.is_committed is True
    assert resource.is_assignable is True
    assert resource.current_commitment_fire_event_id == 5
    assert resource.current_commitment_response_plan_id == 50


def test_unavailable_resource_is_not_assignable_even_if_committed():
    resource = make_resource(
        operational_status=ResourceStatus.UNAVAILABLE,
        current_commitment_fire_event_id=5,
        current_commitment_response_plan_id=50,
    )
    assert resource.is_assignable is False
    assert resource.is_committed is True  # still represented for audit, per Task 4


def test_rejects_mismatched_commitment_fields():
    with pytest.raises(ValueError):
        make_resource(current_commitment_fire_event_id=5, current_commitment_response_plan_id=None)
    with pytest.raises(ValueError):
        make_resource(current_commitment_fire_event_id=None, current_commitment_response_plan_id=50)


def test_rejects_invalid_ids():
    with pytest.raises(ValueError):
        make_resource(resource_id="")
    with pytest.raises(ValueError):
        make_resource(station_id="")
    with pytest.raises(ValueError):
        make_resource(current_commitment_fire_event_id=0, current_commitment_response_plan_id=1)


def test_rejects_invalid_operational_status():
    with pytest.raises(ValueError):
        make_resource(operational_status="available")


def test_rejects_out_of_range_station_coordinates():
    with pytest.raises(ValueError):
        make_resource(station_latitude=91.0)
    with pytest.raises(ValueError):
        make_resource(station_longitude=-181.0)


def test_two_resources_with_the_same_id_are_equal_dataclass_instances():
    """Uniqueness itself is enforced at the collection level
    (GlobalPlanningInput) - this only confirms the model's own equality
    semantics are structural, which that validation relies on."""
    first = make_resource()
    second = make_resource()
    assert first == second
