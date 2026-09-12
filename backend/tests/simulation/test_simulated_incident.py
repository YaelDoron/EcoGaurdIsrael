"""Tests for simulated incident definitions."""

from dataclasses import FrozenInstanceError

import pytest

from src.simulation import CARMEL_LOCATION, ScenarioType, SimulatedIncident


def test_valid_incident_construction():
    incident = SimulatedIncident(
        incident_id="incident-carmel-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=CARMEL_LOCATION,
    )

    assert incident.incident_id == "incident-carmel-01"
    assert incident.scenario_type is ScenarioType.ACTIVE_FIRE
    assert incident.location == CARMEL_LOCATION


@pytest.mark.parametrize("incident_id", ["", " ", None, 123])
def test_invalid_incident_id_is_rejected(incident_id):
    with pytest.raises(ValueError):
        SimulatedIncident(
            incident_id=incident_id,
            scenario_type=ScenarioType.ACTIVE_FIRE,
            location=CARMEL_LOCATION,
        )


def test_invalid_scenario_type_is_rejected():
    with pytest.raises(ValueError):
        SimulatedIncident(
            incident_id="incident-carmel-01",
            scenario_type="active_fire",
            location=CARMEL_LOCATION,
        )


def test_invalid_location_is_rejected():
    with pytest.raises(ValueError):
        SimulatedIncident(
            incident_id="incident-carmel-01",
            scenario_type=ScenarioType.ACTIVE_FIRE,
            location="carmel",
        )


def test_incident_is_immutable():
    incident = SimulatedIncident(
        incident_id="incident-carmel-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=CARMEL_LOCATION,
    )

    with pytest.raises(FrozenInstanceError):
        incident.incident_id = "incident-golan-01"
