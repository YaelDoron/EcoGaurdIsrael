"""Tests for Planning Effective State domain models (Epic 5, US 5.4, Tasks 1 and 3)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import math

import pytest

from src.models import (
    PlanningEffectiveState,
    PlanningResourceState,
    PlanningTargetState,
    ResponseTargetType,
)


def make_active_target(**overrides) -> PlanningTargetState:
    defaults = dict(
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return PlanningTargetState(**defaults)


def make_predicted_target(**overrides) -> PlanningTargetState:
    defaults = dict(
        target_type=ResponseTargetType.PREDICTED_RISK,
        latitude=32.74,
        longitude=35.06,
        priority_score=80.0,
        prediction_horizon_minutes=30,
    )
    defaults.update(overrides)
    return PlanningTargetState(**defaults)


def make_resource(**overrides) -> PlanningResourceState:
    defaults = dict(
        resource_id="resource-1",
        station_id="station-1",
        station_latitude=32.7,
        station_longitude=35.05,
    )
    defaults.update(overrides)
    return PlanningResourceState(**defaults)


def make_state(**overrides) -> PlanningEffectiveState:
    defaults = dict(
        fire_event_id=1,
        targets=(make_active_target(),),
        resources=(make_resource(),),
        routing_methodology="ECOGUARD_ROUTING_DIJKSTRA",
        routing_methodology_version="1.0",
        optimization_methodology="GENETIC_RESOURCE_ALLOCATION",
        optimization_methodology_version="1.0",
    )
    defaults.update(overrides)
    return PlanningEffectiveState(**defaults)


def test_valid_planning_target_state_can_be_created():
    target = make_active_target()

    assert target.target_type is ResponseTargetType.ACTIVE_FIRE
    assert target.latitude == 32.731
    assert target.longitude == 35.046
    assert target.priority_score == 150.0


def test_active_fire_target_can_have_no_prediction_horizon():
    target = make_active_target()

    assert target.prediction_horizon_minutes is None


def test_predicted_risk_target_can_contain_horizon():
    target = make_predicted_target()

    assert target.target_type is ResponseTargetType.PREDICTED_RISK
    assert target.prediction_horizon_minutes == 30


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_target_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_active_target(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_target_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_active_target(longitude=longitude)


@pytest.mark.parametrize("priority_score", [math.nan, math.inf, -math.inf, True, "100"])
def test_non_finite_priority_score_rejected(priority_score):
    with pytest.raises(ValueError):
        make_active_target(priority_score=priority_score)


@pytest.mark.parametrize("prediction_horizon_minutes", [-1, True, "30"])
def test_invalid_prediction_horizon_rejected(prediction_horizon_minutes):
    with pytest.raises(ValueError):
        make_predicted_target(prediction_horizon_minutes=prediction_horizon_minutes)


def test_invalid_target_type_rejected():
    with pytest.raises(ValueError):
        make_active_target(target_type="active_fire")


def test_planning_target_state_is_immutable():
    target = make_active_target()

    with pytest.raises(FrozenInstanceError):
        target.priority_score = 10.0


def test_valid_planning_resource_state_can_be_created():
    resource = make_resource()

    assert resource.resource_id == "resource-1"
    assert resource.station_id == "station-1"
    assert resource.station_latitude == 32.7
    assert resource.station_longitude == 35.05


@pytest.mark.parametrize("station_latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_resource_station_latitude_rejected(station_latitude):
    with pytest.raises(ValueError):
        make_resource(station_latitude=station_latitude)


@pytest.mark.parametrize("station_longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_resource_station_longitude_rejected(station_longitude):
    with pytest.raises(ValueError):
        make_resource(station_longitude=station_longitude)


@pytest.mark.parametrize("resource_id", ["", "   "])
def test_invalid_resource_id_rejected(resource_id):
    with pytest.raises(ValueError):
        make_resource(resource_id=resource_id)


@pytest.mark.parametrize("station_id", ["", "   "])
def test_invalid_station_id_rejected(station_id):
    with pytest.raises(ValueError):
        make_resource(station_id=station_id)


def test_planning_resource_state_is_immutable():
    resource = make_resource()

    with pytest.raises(FrozenInstanceError):
        resource.resource_id = "other"


def test_planning_effective_state_can_contain_multiple_targets_and_resources():
    state = make_state(
        targets=(make_active_target(), make_predicted_target()),
        resources=(make_resource(), make_resource(resource_id="resource-2", station_id="station-2")),
    )

    assert len(state.targets) == 2
    assert len(state.resources) == 2


def test_planning_effective_state_targets_and_resources_are_tuples():
    state = make_state(
        targets=[make_active_target()],
        resources=[make_resource()],
    )

    assert isinstance(state.targets, tuple)
    assert isinstance(state.resources, tuple)


def test_two_states_with_identical_semantic_fields_compare_equal():
    first = make_state()
    second = make_state()

    assert first == second


def test_states_with_different_targets_compare_unequal():
    first = make_state()
    second = make_state(targets=(make_predicted_target(),))

    assert first != second


def test_invalid_fire_event_id_rejected():
    with pytest.raises(ValueError):
        make_state(fire_event_id=0)


@pytest.mark.parametrize(
    "field_name",
    [
        "routing_methodology",
        "routing_methodology_version",
        "optimization_methodology",
        "optimization_methodology_version",
    ],
)
def test_blank_methodology_identity_field_rejected(field_name):
    with pytest.raises(ValueError):
        make_state(**{field_name: "   "})


def test_targets_must_contain_planning_target_state_items():
    with pytest.raises(ValueError):
        make_state(targets=("not-a-target",))


def test_resources_must_contain_planning_resource_state_items():
    with pytest.raises(ValueError):
        make_state(resources=("not-a-resource",))


def test_planning_effective_state_is_immutable():
    state = make_state()

    with pytest.raises(FrozenInstanceError):
        state.fire_event_id = 2


def test_provenance_fields_are_not_part_of_planning_target_state():
    field_names = {field.name for field in fields(PlanningTargetState)}

    assert "response_target_id" not in field_names
    assert "response_target_set_id" not in field_names
    assert "spread_prediction_id" not in field_names
    assert "spread_prediction_cell_id" not in field_names


def test_provenance_fields_are_not_part_of_planning_effective_state():
    field_names = {field.name for field in fields(PlanningEffectiveState)}

    assert "response_target_set_id" not in field_names
    assert "response_target_id" not in field_names
    assert "route_planning_run_id" not in field_names
    assert "response_plan_id" not in field_names
    assert "generated_at" not in field_names
    assert "as_of" not in field_names
    assert "spread_prediction_id" not in field_names
    assert "severity_assessment_id" not in field_names


# ---------------------------------------------------------------------------
# Task 3: fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_across_repeated_calls():
    state = make_state()

    assert state.fingerprint == state.fingerprint


def test_fingerprint_is_deterministic_across_equivalent_separately_created_states():
    first = make_state()
    second = make_state()

    assert first is not second
    assert first.fingerprint == second.fingerprint


def test_fingerprint_is_order_independent_for_targets_and_resources():
    active = make_active_target()
    predicted = make_predicted_target()
    resource_a = make_resource(resource_id="resource-1", station_id="station-1")
    resource_b = make_resource(
        resource_id="resource-2", station_id="station-2", station_latitude=32.8, station_longitude=35.2
    )

    first = make_state(targets=(active, predicted), resources=(resource_a, resource_b))
    second = make_state(targets=(predicted, active), resources=(resource_b, resource_a))

    assert first.fingerprint == second.fingerprint


def test_fingerprint_is_order_independent_when_resources_share_station_and_resource_id():
    """Even with tied station_id/resource_id, station_latitude/longitude totally order the sort key."""
    resource_x = make_resource(
        resource_id="resource-1", station_id="station-1", station_latitude=32.7, station_longitude=35.0
    )
    resource_y = make_resource(
        resource_id="resource-1", station_id="station-1", station_latitude=32.9, station_longitude=35.3
    )

    state_a = make_state(resources=(resource_x, resource_y))
    state_b = make_state(resources=(resource_y, resource_x))

    assert state_a.fingerprint == state_b.fingerprint


def test_fingerprint_output_is_stable_sha256_hex_format():
    fingerprint = make_state().fingerprint

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64
    assert fingerprint == fingerprint.lower()
    int(fingerprint, 16)


def test_fingerprint_requires_no_external_dependencies():
    """Pure function of already-constructed state: no repository, DB, or as_of/timestamp involved."""
    state = PlanningEffectiveState(
        fire_event_id=7,
        targets=(make_active_target(),),
        resources=(make_resource(),),
    )

    assert isinstance(state.fingerprint, str)


def test_two_effective_states_built_from_different_persisted_snapshots_share_fingerprint():
    """PlanningEffectiveState structurally excludes response_target_set_id/response_target_id,

    so two states that would have come from different persisted ResponseTargetSet rows
    (e.g. set #501 vs #999) but identical semantic content always fingerprint equal.
    """
    first = make_state()
    second = make_state()

    assert first.fingerprint == second.fingerprint


def test_changed_target_priority_changes_fingerprint():
    baseline = make_state()
    changed = make_state(targets=(make_active_target(priority_score=999.0),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_target_coordinates_change_fingerprint():
    baseline = make_state()
    changed = make_state(targets=(make_active_target(latitude=10.0, longitude=10.0),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_target_type_changes_fingerprint():
    baseline = make_state()
    changed = make_state(targets=(make_active_target(target_type=ResponseTargetType.PREDICTED_RISK),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_prediction_horizon_changes_fingerprint():
    baseline = make_state(targets=(make_predicted_target(),))
    changed = make_state(targets=(make_predicted_target(prediction_horizon_minutes=60),))

    assert changed.fingerprint != baseline.fingerprint


def test_adding_target_changes_fingerprint():
    baseline = make_state()
    changed = make_state(targets=(make_active_target(), make_predicted_target()))

    assert changed.fingerprint != baseline.fingerprint


def test_removing_target_changes_fingerprint():
    baseline = make_state(targets=(make_active_target(), make_predicted_target()))
    changed = make_state(targets=(make_active_target(),))

    assert changed.fingerprint != baseline.fingerprint


def test_adding_available_resource_changes_fingerprint():
    baseline = make_state()
    changed = make_state(
        resources=(make_resource(), make_resource(resource_id="resource-2", station_id="station-2"))
    )

    assert changed.fingerprint != baseline.fingerprint


def test_removing_resource_changes_fingerprint():
    baseline = make_state(
        resources=(make_resource(), make_resource(resource_id="resource-2", station_id="station-2"))
    )
    changed = make_state(resources=(make_resource(),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_resource_id_changes_fingerprint():
    baseline = make_state()
    changed = make_state(resources=(make_resource(resource_id="resource-other"),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_resource_station_id_changes_fingerprint():
    baseline = make_state()
    changed = make_state(resources=(make_resource(station_id="station-other"),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_resource_station_coordinates_change_fingerprint():
    baseline = make_state()
    changed = make_state(resources=(make_resource(station_latitude=10.0, station_longitude=10.0),))

    assert changed.fingerprint != baseline.fingerprint


def test_changed_routing_methodology_changes_fingerprint():
    baseline = make_state()
    changed = make_state(routing_methodology="OTHER_ROUTING_METHOD")

    assert changed.fingerprint != baseline.fingerprint


def test_changed_routing_methodology_version_changes_fingerprint():
    baseline = make_state()
    changed = make_state(routing_methodology_version="2.0")

    assert changed.fingerprint != baseline.fingerprint


def test_changed_optimization_methodology_changes_fingerprint():
    baseline = make_state()
    changed = make_state(optimization_methodology="OTHER_OPTIMIZATION_METHOD")

    assert changed.fingerprint != baseline.fingerprint


def test_changed_optimization_methodology_version_changes_fingerprint():
    baseline = make_state()
    changed = make_state(optimization_methodology_version="2.0")

    assert changed.fingerprint != baseline.fingerprint


def test_changed_fire_event_id_changes_fingerprint():
    baseline = make_state(fire_event_id=1)
    changed = make_state(fire_event_id=2)

    assert changed.fingerprint != baseline.fingerprint
