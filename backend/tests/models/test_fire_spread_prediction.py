"""Tests for wildfire-spread prediction domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from src.models import FireSpreadPrediction, FireSpreadPredictionCell, FireSpreadPredictionStatus
from src.models.fire_spread_prediction import CA_TIME_STEP_MINUTES

PREDICTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_cell(**overrides) -> FireSpreadPredictionCell:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        spread_probability=0.72,
        spread_risk_score=72.0,
        reached_step=2,
        reached_minutes=10,
    )
    defaults.update(overrides)
    return FireSpreadPredictionCell(**defaults)


def make_prediction(**overrides) -> FireSpreadPrediction:
    defaults = dict(
        fire_event_id=1,
        severity_assessment_id=1,
        predicted_at=PREDICTED_AT,
        horizon_minutes=30,
        status=FireSpreadPredictionStatus.VALID,
        methodology="ECOGUARD_PROPAGATOR_CA",
        methodology_version="1.0",
        cells=(make_cell(),),
    )
    defaults.update(overrides)
    return FireSpreadPrediction(**defaults)


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------


def test_valid_cell_construction():
    cell = make_cell()

    assert cell.latitude == 32.731
    assert cell.spread_probability == 0.72
    assert cell.spread_risk_score == 72.0
    assert cell.reached_step == 2
    assert cell.reached_minutes == 10


def test_valid_30_minute_prediction():
    prediction = make_prediction(horizon_minutes=30)

    assert prediction.horizon_minutes == 30
    assert prediction.status is FireSpreadPredictionStatus.VALID
    assert len(prediction.cells) == 1


def test_valid_60_minute_prediction():
    prediction = make_prediction(
        horizon_minutes=60,
        cells=(make_cell(reached_step=12, reached_minutes=60),),
    )

    assert prediction.horizon_minutes == 60
    assert prediction.cells[0].reached_minutes == 60


def test_prediction_is_immutable():
    prediction = make_prediction()

    with pytest.raises(FrozenInstanceError):
        prediction.horizon_minutes = 60


def test_cell_is_immutable():
    cell = make_cell()

    with pytest.raises(FrozenInstanceError):
        cell.spread_probability = 0.1


# ---------------------------------------------------------------------------
# Cell validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latitude", [-90.1, 90.1])
def test_cell_invalid_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_cell(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1])
def test_cell_invalid_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_cell(longitude=longitude)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_cell_nan_and_infinite_coordinates_rejected(coordinate):
    with pytest.raises(ValueError):
        make_cell(latitude=coordinate)
    with pytest.raises(ValueError):
        make_cell(longitude=coordinate)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_cell_probability_out_of_range_rejected(probability):
    with pytest.raises(ValueError):
        make_cell(spread_probability=probability, spread_risk_score=max(0.0, min(100.0, probability * 100)))


@pytest.mark.parametrize("probability", [math.nan, math.inf, -math.inf])
def test_cell_nan_and_infinite_probability_rejected(probability):
    with pytest.raises(ValueError):
        make_cell(spread_probability=probability)


@pytest.mark.parametrize("risk_score", [-0.1, 100.1])
def test_cell_risk_score_out_of_range_rejected(risk_score):
    with pytest.raises(ValueError):
        make_cell(spread_risk_score=risk_score)


@pytest.mark.parametrize("risk_score", [math.nan, math.inf, -math.inf])
def test_cell_nan_and_infinite_risk_score_rejected(risk_score):
    with pytest.raises(ValueError):
        make_cell(spread_risk_score=risk_score)


@pytest.mark.parametrize("field", ["latitude", "longitude", "spread_probability", "spread_risk_score"])
def test_cell_bool_rejected_for_numeric_fields(field):
    with pytest.raises(ValueError):
        make_cell(**{field: True})


def test_cell_risk_score_inconsistent_with_probability_rejected():
    with pytest.raises(ValueError):
        make_cell(spread_probability=0.72, spread_risk_score=50.0)


def test_cell_risk_score_within_tolerance_accepted():
    cell = make_cell(spread_probability=0.7234, spread_risk_score=72.34)

    assert cell.spread_probability == 0.7234


@pytest.mark.parametrize("reached_step", [-1, True, 1.5, "2"])
def test_cell_invalid_reached_step_rejected(reached_step):
    with pytest.raises(ValueError):
        make_cell(reached_step=reached_step, reached_minutes=10)


@pytest.mark.parametrize("reached_minutes", [-1, True, 1.5, "10"])
def test_cell_invalid_reached_minutes_rejected(reached_minutes):
    with pytest.raises(ValueError):
        make_cell(reached_minutes=reached_minutes)


def test_cell_reached_minutes_inconsistent_with_step_rejected():
    with pytest.raises(ValueError):
        make_cell(reached_step=2, reached_minutes=11)


def test_cell_reached_minutes_matches_step_times_time_step():
    cell = make_cell(reached_step=3, reached_minutes=3 * CA_TIME_STEP_MINUTES)

    assert cell.reached_minutes == 15


# ---------------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_prediction(fire_event_id=fire_event_id)


@pytest.mark.parametrize("severity_assessment_id", [0, -1, True, "1"])
def test_invalid_severity_assessment_id_rejected(severity_assessment_id):
    with pytest.raises(ValueError):
        make_prediction(severity_assessment_id=severity_assessment_id)


def test_valid_status_without_severity_assessment_id_rejected():
    """A VALID prediction always went through a real severity assessment
    (Task 5's READY path), so it must always be traceable to one."""
    with pytest.raises(ValueError):
        make_prediction(status=FireSpreadPredictionStatus.VALID, severity_assessment_id=None)


@pytest.mark.parametrize(
    "status", [FireSpreadPredictionStatus.INSUFFICIENT_DATA, FireSpreadPredictionStatus.INACTIVE_EVENT]
)
def test_non_valid_status_without_severity_assessment_id_accepted(status):
    """Task 5 showed INSUFFICIENT_DATA/INACTIVE_EVENT can occur before a
    severity assessment was ever retrieved (e.g. missing FireEvent, or an
    inactive FireEvent short-circuiting before severity retrieval) -- there
    is no id to fabricate, so None must be allowed for these statuses."""
    prediction = make_prediction(status=status, cells=(), severity_assessment_id=None)

    assert prediction.severity_assessment_id is None


@pytest.mark.parametrize(
    "status", [FireSpreadPredictionStatus.INSUFFICIENT_DATA, FireSpreadPredictionStatus.INACTIVE_EVENT]
)
def test_non_valid_status_with_severity_assessment_id_still_accepted(status):
    """A non-VALID prediction may still know which severity assessment it
    inspected (e.g. an assessment existed but was stale) -- None is allowed,
    not required."""
    prediction = make_prediction(status=status, cells=(), severity_assessment_id=500)

    assert prediction.severity_assessment_id == 500


def test_naive_predicted_at_rejected():
    with pytest.raises(ValueError):
        make_prediction(predicted_at=datetime(2026, 9, 14, 12, 0))


@pytest.mark.parametrize("horizon_minutes", [0, -30, 45, 90, True])
def test_invalid_horizon_rejected(horizon_minutes):
    with pytest.raises(ValueError):
        make_prediction(horizon_minutes=horizon_minutes)


def test_invalid_status_type_rejected():
    with pytest.raises(ValueError):
        make_prediction(status="valid")


@pytest.mark.parametrize("methodology", ["", "   ", None])
def test_empty_methodology_rejected(methodology):
    with pytest.raises(ValueError):
        make_prediction(methodology=methodology)


@pytest.mark.parametrize("methodology_version", ["", "   ", None])
def test_empty_methodology_version_rejected(methodology_version):
    with pytest.raises(ValueError):
        make_prediction(methodology_version=methodology_version)


def test_non_tuple_cells_rejected():
    with pytest.raises(ValueError):
        make_prediction(cells=[make_cell()])


def test_cells_tuple_with_wrong_type_rejected():
    with pytest.raises(ValueError):
        make_prediction(cells=(make_cell(), "not-a-cell"))


# ---------------------------------------------------------------------------
# Status consistency
# ---------------------------------------------------------------------------


def test_valid_status_with_no_cells_accepted():
    """A scientifically valid prediction may legitimately find no cell crosses
    the propagation threshold within the horizon -- that is a valid "no
    predicted spread" result, not an error."""
    prediction = make_prediction(status=FireSpreadPredictionStatus.VALID, cells=())

    assert prediction.status is FireSpreadPredictionStatus.VALID
    assert prediction.cells == ()


def test_insufficient_data_with_cells_rejected():
    with pytest.raises(ValueError):
        make_prediction(status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=(make_cell(),))


def test_inactive_event_with_cells_rejected():
    with pytest.raises(ValueError):
        make_prediction(status=FireSpreadPredictionStatus.INACTIVE_EVENT, cells=(make_cell(),))


def test_insufficient_data_with_no_cells_accepted():
    prediction = make_prediction(status=FireSpreadPredictionStatus.INSUFFICIENT_DATA, cells=())

    assert prediction.status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
    assert prediction.cells == ()


def test_inactive_event_with_no_cells_accepted():
    prediction = make_prediction(status=FireSpreadPredictionStatus.INACTIVE_EVENT, cells=())

    assert prediction.status is FireSpreadPredictionStatus.INACTIVE_EVENT
    assert prediction.cells == ()
