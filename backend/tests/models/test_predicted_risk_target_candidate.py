"""Tests for predicted response-target candidate input model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from src.models import PredictedRiskTargetCandidate


def make_candidate(**overrides) -> PredictedRiskTargetCandidate:
    defaults = dict(
        fire_event_id=1,
        latitude=32.731,
        longitude=35.046,
        risk_score=72.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=10,
        spread_prediction_cell_id=100,
    )
    defaults.update(overrides)
    return PredictedRiskTargetCandidate(**defaults)


def test_valid_candidate_construction():
    candidate = make_candidate()

    assert candidate.fire_event_id == 1
    assert candidate.latitude == 32.731
    assert candidate.risk_score == 72.0
    assert candidate.prediction_horizon_minutes == 30
    assert candidate.spread_prediction_id == 10
    assert candidate.spread_prediction_cell_id == 100


def test_candidate_is_immutable():
    candidate = make_candidate()

    with pytest.raises(FrozenInstanceError):
        candidate.risk_score = 80.0


@pytest.mark.parametrize("field", ["fire_event_id", "spread_prediction_id", "spread_prediction_cell_id"])
@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_positive_ids_required(field, value):
    with pytest.raises(ValueError):
        make_candidate(**{field: value})


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_candidate(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_candidate(longitude=longitude)


@pytest.mark.parametrize("risk_score", [-0.1, 100.1, math.nan, math.inf, -math.inf, True])
def test_invalid_risk_score_rejected(risk_score):
    with pytest.raises(ValueError):
        make_candidate(risk_score=risk_score)


@pytest.mark.parametrize("prediction_horizon_minutes", [0, -30, True, 30.5, "30"])
def test_horizon_must_be_positive_integer(prediction_horizon_minutes):
    with pytest.raises(ValueError):
        make_candidate(prediction_horizon_minutes=prediction_horizon_minutes)
