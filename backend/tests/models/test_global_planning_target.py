"""Tests for GlobalPlanningTarget (Stage 3 of the Global Multi-Incident Optimizer refactor)."""
from __future__ import annotations

import pytest

from src.models.global_planning_target import GlobalPlanningTarget
from src.models.response_target_type import ResponseTargetType


def make_target(**overrides) -> GlobalPlanningTarget:
    defaults = dict(
        fire_event_id=1,
        response_target_id=10,
        target_order=0,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.7,
        longitude=35.0,
        priority_score=100.0,
        prediction_horizon_minutes=None,
    )
    defaults.update(overrides)
    return GlobalPlanningTarget(**defaults)


def test_valid_target_retains_owning_fire_event_id():
    target = make_target(fire_event_id=7)
    assert target.fire_event_id == 7


def test_rejects_invalid_fire_event_id():
    with pytest.raises(ValueError):
        make_target(fire_event_id=0)
    with pytest.raises(ValueError):
        make_target(fire_event_id=-1)
    with pytest.raises(ValueError):
        make_target(fire_event_id=True)


def test_rejects_invalid_response_target_id():
    with pytest.raises(ValueError):
        make_target(response_target_id=0)


def test_rejects_negative_target_order():
    with pytest.raises(ValueError):
        make_target(target_order=-1)


def test_rejects_invalid_target_type():
    with pytest.raises(ValueError):
        make_target(target_type="active_fire")


def test_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        make_target(latitude=91.0)
    with pytest.raises(ValueError):
        make_target(longitude=181.0)


def test_predicted_risk_target_with_prediction_horizon():
    target = make_target(
        target_type=ResponseTargetType.PREDICTED_RISK, prediction_horizon_minutes=30
    )
    assert target.prediction_horizon_minutes == 30


def test_rejects_negative_prediction_horizon():
    with pytest.raises(ValueError):
        make_target(prediction_horizon_minutes=-1)
