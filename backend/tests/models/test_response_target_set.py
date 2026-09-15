"""Tests for response-target set domain model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.models import ResponseTarget, ResponseTargetSet, ResponseTargetType

GENERATED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_active(**overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=1,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def make_predicted(**overrides) -> ResponseTarget:
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


def make_set(**overrides) -> ResponseTargetSet:
    defaults = dict(
        fire_event_id=1,
        generated_at=GENERATED_AT,
        methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
        methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
        targets=(make_active(),),
    )
    defaults.update(overrides)
    return ResponseTargetSet(**defaults)


def test_valid_set_with_one_active_fire_target():
    target_set = make_set()

    assert target_set.fire_event_id == 1
    assert target_set.targets == (make_active(),)


def test_valid_set_with_active_fire_and_predicted_targets():
    target_set = make_set(targets=(make_active(), make_predicted()))

    assert len(target_set.targets) == 2
    assert target_set.targets[1].target_type is ResponseTargetType.PREDICTED_RISK


def test_target_set_is_immutable():
    target_set = make_set()

    with pytest.raises(FrozenInstanceError):
        target_set.methodology = "changed"


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_non_positive_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_set(fire_event_id=fire_event_id)


def test_naive_generated_at_rejected():
    with pytest.raises(ValueError):
        make_set(generated_at=datetime(2026, 9, 14, 12, 0))


@pytest.mark.parametrize("methodology", ["", "   ", None])
def test_empty_methodology_rejected(methodology):
    with pytest.raises(ValueError):
        make_set(methodology=methodology)


@pytest.mark.parametrize("methodology_version", ["", "   ", None])
def test_empty_methodology_version_rejected(methodology_version):
    with pytest.raises(ValueError):
        make_set(methodology_version=methodology_version)


def test_empty_targets_rejected():
    with pytest.raises(ValueError):
        make_set(targets=())


def test_target_belonging_to_another_fire_event_rejected():
    with pytest.raises(ValueError):
        make_set(targets=(make_active(fire_event_id=1), make_predicted(fire_event_id=2)))


def test_set_without_active_fire_rejected():
    with pytest.raises(ValueError):
        make_set(targets=(make_predicted(),))


def test_set_with_two_active_fire_targets_rejected():
    with pytest.raises(ValueError):
        make_set(targets=(make_active(), make_active(latitude=32.732)))


def test_target_order_in_tuple_is_preserved():
    first = make_predicted(spread_prediction_id=10, spread_prediction_cell_id=100)
    second = make_predicted(latitude=32.75, spread_prediction_id=11, spread_prediction_cell_id=101)
    target_set = make_set(targets=(first, make_active(), second))

    assert target_set.targets == (first, make_active(), second)


def test_identical_input_constructs_identical_domain_value():
    first = make_set(targets=(make_active(), make_predicted()))
    second = make_set(targets=(make_active(), make_predicted()))

    assert first == second
