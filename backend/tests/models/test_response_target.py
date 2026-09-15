"""Tests for operational response target domain model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from src.models import ResponseTarget, ResponseTargetType


def make_active_target(**overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=1,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def make_predicted_target(**overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=1,
        target_type=ResponseTargetType.PREDICTED_RISK,
        latitude=32.74,
        longitude=35.06,
        priority_score=80.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=10,
        spread_prediction_cell_id=100,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def test_valid_active_fire_target():
    target = make_active_target()

    assert target.target_type is ResponseTargetType.ACTIVE_FIRE
    assert target.prediction_horizon_minutes is None
    assert target.spread_prediction_id is None
    assert target.spread_prediction_cell_id is None


def test_valid_predicted_risk_target():
    target = make_predicted_target()

    assert target.target_type is ResponseTargetType.PREDICTED_RISK
    assert target.prediction_horizon_minutes == 30
    assert target.spread_prediction_id == 10
    assert target.spread_prediction_cell_id == 100


def test_response_target_is_immutable():
    target = make_active_target()

    with pytest.raises(FrozenInstanceError):
        target.priority_score = 10.0


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_active_target(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_active_target(longitude=longitude)


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_active_target(fire_event_id=fire_event_id)


@pytest.mark.parametrize("priority_score", [math.nan, math.inf, -math.inf, True, "100"])
def test_invalid_priority_score_rejected(priority_score):
    with pytest.raises(ValueError):
        make_active_target(priority_score=priority_score)


def test_invalid_target_type_rejected():
    with pytest.raises(ValueError):
        make_active_target(target_type="active_fire")


@pytest.mark.parametrize(
    "field",
    ["prediction_horizon_minutes", "spread_prediction_id", "spread_prediction_cell_id"],
)
def test_predicted_target_requires_prediction_metadata(field):
    with pytest.raises(ValueError):
        make_predicted_target(**{field: None})


@pytest.mark.parametrize(
    "field",
    ["prediction_horizon_minutes", "spread_prediction_id", "spread_prediction_cell_id"],
)
@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_predicted_target_requires_positive_prediction_metadata(field, value):
    with pytest.raises(ValueError):
        make_predicted_target(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_horizon_minutes", 30),
        ("spread_prediction_id", 10),
        ("spread_prediction_cell_id", 100),
    ],
)
def test_active_target_rejects_prediction_only_metadata(field, value):
    with pytest.raises(ValueError):
        make_active_target(**{field: value})
