"""Tests for response-target input domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from src.models import (
    PredictedRiskTargetCandidate,
    ResponseTargetInput,
    ResponseTargetInputResult,
    ResponseTargetInputStatus,
)


def make_candidate(**overrides) -> PredictedRiskTargetCandidate:
    defaults = dict(
        fire_event_id=1,
        latitude=32.74,
        longitude=35.06,
        risk_score=55.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=10,
        spread_prediction_cell_id=100,
    )
    defaults.update(overrides)
    return PredictedRiskTargetCandidate(**defaults)


def make_input(**overrides) -> ResponseTargetInput:
    defaults = dict(
        fire_event_id=1,
        fire_latitude=32.731,
        fire_longitude=35.046,
        severity_score=None,
        predicted_candidates=(make_candidate(),),
    )
    defaults.update(overrides)
    return ResponseTargetInput(**defaults)


def test_valid_input_with_optional_severity_and_candidates():
    input_data = make_input(severity_score=72.5)

    assert input_data.fire_event_id == 1
    assert input_data.severity_score == 72.5
    assert len(input_data.predicted_candidates) == 1


def test_input_is_immutable():
    input_data = make_input()

    with pytest.raises(FrozenInstanceError):
        input_data.severity_score = 10.0


def test_candidates_are_stored_as_sorted_tuple():
    later = make_candidate(prediction_horizon_minutes=60, spread_prediction_id=20, spread_prediction_cell_id=200)
    earlier = make_candidate(prediction_horizon_minutes=30, spread_prediction_id=10, spread_prediction_cell_id=100)

    input_data = make_input(predicted_candidates=[later, earlier])

    assert isinstance(input_data.predicted_candidates, tuple)
    assert input_data.predicted_candidates == (earlier, later)


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_input(fire_event_id=fire_event_id, predicted_candidates=())


@pytest.mark.parametrize("latitude", [-90.1, 90.1, math.nan, math.inf, -math.inf, True])
def test_invalid_fire_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_input(fire_latitude=latitude, predicted_candidates=())


@pytest.mark.parametrize("longitude", [-180.1, 180.1, math.nan, math.inf, -math.inf, True])
def test_invalid_fire_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_input(fire_longitude=longitude, predicted_candidates=())


@pytest.mark.parametrize("severity_score", [-0.1, 100.1, math.nan, math.inf, -math.inf, True])
def test_invalid_severity_score_rejected(severity_score):
    with pytest.raises(ValueError):
        make_input(severity_score=severity_score, predicted_candidates=())


def test_candidate_from_another_fire_event_rejected():
    with pytest.raises(ValueError):
        make_input(predicted_candidates=(make_candidate(fire_event_id=2),))


def test_non_candidate_rejected():
    with pytest.raises(ValueError):
        make_input(predicted_candidates=("not-a-candidate",))


def test_ready_result_requires_input_data():
    with pytest.raises(ValueError):
        ResponseTargetInputResult(
            status=ResponseTargetInputStatus.READY,
            input_data=None,
            fire_event_id=1,
        )


def test_ready_result_accepts_matching_input_data():
    input_data = make_input()

    result = ResponseTargetInputResult(
        status=ResponseTargetInputStatus.READY,
        input_data=input_data,
        fire_event_id=1,
    )

    assert result.input_data == input_data


def test_ready_result_rejects_mismatched_input_fire_event_id():
    with pytest.raises(ValueError):
        ResponseTargetInputResult(
            status=ResponseTargetInputStatus.READY,
            input_data=make_input(fire_event_id=2, predicted_candidates=(make_candidate(fire_event_id=2),)),
            fire_event_id=1,
        )


def test_inactive_result_must_not_include_input_data():
    with pytest.raises(ValueError):
        ResponseTargetInputResult(
            status=ResponseTargetInputStatus.INACTIVE_EVENT,
            input_data=make_input(),
            fire_event_id=1,
        )


def test_inactive_result_without_input_data_is_valid():
    result = ResponseTargetInputResult(
        status=ResponseTargetInputStatus.INACTIVE_EVENT,
        input_data=None,
        fire_event_id=1,
    )

    assert result.status is ResponseTargetInputStatus.INACTIVE_EVENT


def test_invalid_result_status_rejected():
    with pytest.raises(ValueError):
        ResponseTargetInputResult(status="ready", input_data=make_input(), fire_event_id=1)
