"""Tests for GlobalEventResourceDemandResult (Stage 5 of the Global
Multi-Incident Optimizer refactor, Tasks 18/19)."""
from __future__ import annotations

import pytest

from src.models.global_event_resource_demand_result import GlobalEventResourceDemandResult


def make_result(**overrides) -> GlobalEventResourceDemandResult:
    defaults = dict(
        fire_event_id=1,
        minimum_resources=2,
        desired_resources=3,
        suppression_resources_assigned=2,
        required_slots_covered=2,
        required_slots_uncovered=0,
        desired_slots_covered=0,
        desired_slots_uncovered=1,
        predicted_risk_slots_covered=0,
    )
    defaults.update(overrides)
    return GlobalEventResourceDemandResult(**defaults)


def test_valid_result_round_trips_and_exposes_unmet_properties():
    result = make_result()
    assert result.unmet_minimum == 0
    assert result.unmet_desired == 1


def test_unmet_minimum_reflects_uncovered_required_slots():
    result = make_result(
        required_slots_covered=1,
        required_slots_uncovered=1,
        suppression_resources_assigned=1,
        desired_slots_covered=0,
        desired_slots_uncovered=1,
    )
    assert result.unmet_minimum == 1
    assert result.unmet_desired == 2


def test_rejects_desired_below_minimum():
    with pytest.raises(ValueError):
        make_result(minimum_resources=3, desired_resources=2)


def test_rejects_required_slot_arithmetic_mismatch():
    with pytest.raises(ValueError):
        make_result(minimum_resources=2, required_slots_covered=1, required_slots_uncovered=0)


def test_rejects_desired_slot_arithmetic_mismatch():
    with pytest.raises(ValueError):
        make_result(
            minimum_resources=2,
            desired_resources=3,
            required_slots_covered=2,
            required_slots_uncovered=0,
            desired_slots_covered=1,
            desired_slots_uncovered=1,
        )


def test_rejects_suppression_assigned_inconsistent_with_covered_slots():
    with pytest.raises(ValueError):
        make_result(suppression_resources_assigned=5)


def test_rejects_negative_field():
    with pytest.raises(ValueError):
        make_result(predicted_risk_slots_covered=-1)


def test_rejects_non_positive_fire_event_id():
    with pytest.raises(ValueError):
        make_result(fire_event_id=0)
