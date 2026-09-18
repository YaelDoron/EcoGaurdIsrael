"""Tests for GlobalIncidentDemand (Stage 5 of the Global Multi-Incident
Optimizer refactor, Task 4)."""
from __future__ import annotations

import pytest

from src.models.demand_source import DemandSource
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_incident_demand import GlobalIncidentDemand


def make_demand(**overrides) -> GlobalIncidentDemand:
    defaults = dict(
        fire_event_id=1,
        severity_assessment_id=10,
        severity_level=FireSeverityLevel.HIGH,
        severity_score=70.0,
        minimum_resources=2,
        desired_resources=3,
        demand_source=DemandSource.SEVERITY_ASSESSMENT,
        policy_methodology="ecoguard_demo_severity_demand_policy",
        policy_version="1.0",
    )
    defaults.update(overrides)
    return GlobalIncidentDemand(**defaults)


def test_valid_severity_assessment_demand_round_trips():
    demand = make_demand()
    assert demand.fire_event_id == 1
    assert demand.severity_level is FireSeverityLevel.HIGH
    assert demand.minimum_resources == 2
    assert demand.desired_resources == 3


def test_valid_insufficient_severity_demand_carries_no_severity_fields():
    demand = make_demand(
        severity_assessment_id=None,
        severity_level=None,
        severity_score=None,
        minimum_resources=1,
        desired_resources=1,
        demand_source=DemandSource.INSUFFICIENT_SEVERITY,
    )
    assert demand.severity_level is None
    assert demand.demand_source is DemandSource.INSUFFICIENT_SEVERITY


def test_rejects_desired_below_minimum():
    with pytest.raises(ValueError):
        make_demand(minimum_resources=3, desired_resources=2)


def test_rejects_negative_minimum():
    with pytest.raises(ValueError):
        make_demand(minimum_resources=-1, desired_resources=2)


def test_rejects_severity_assessment_source_without_severity_level():
    with pytest.raises(ValueError):
        make_demand(demand_source=DemandSource.SEVERITY_ASSESSMENT, severity_level=None)


def test_rejects_insufficient_severity_source_carrying_a_severity_level():
    with pytest.raises(ValueError):
        make_demand(
            demand_source=DemandSource.INSUFFICIENT_SEVERITY,
            severity_level=FireSeverityLevel.LOW,
        )


def test_rejects_insufficient_severity_source_carrying_a_severity_score():
    with pytest.raises(ValueError):
        make_demand(
            demand_source=DemandSource.INSUFFICIENT_SEVERITY,
            severity_level=None,
            severity_score=10.0,
        )


def test_rejects_severity_score_out_of_range():
    with pytest.raises(ValueError):
        make_demand(severity_score=10_000.0)


def test_rejects_non_positive_fire_event_id():
    with pytest.raises(ValueError):
        make_demand(fire_event_id=0)


def test_rejects_wrong_type_severity_level():
    with pytest.raises(ValueError):
        make_demand(severity_level="HIGH")


def test_rejects_empty_policy_methodology():
    with pytest.raises(ValueError):
        make_demand(policy_methodology="")


def test_rejects_wrong_type_demand_source():
    with pytest.raises(ValueError):
        make_demand(demand_source="severity_assessment")
