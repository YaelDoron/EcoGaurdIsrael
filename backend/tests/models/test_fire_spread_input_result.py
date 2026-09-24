"""Tests for the FireSpreadInputResult traceability model."""
from __future__ import annotations

import pytest

from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_input_result import FireSpreadInputResult
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason

_REASON = FireSpreadInsufficientDataReason.MISSING_VEGETATION


def make_input(**overrides) -> FireSpreadInput:
    defaults = dict(
        origin_latitude=32.75,
        origin_longitude=35.0,
        wind_speed_kmh=20.0,
        wind_direction_deg=270.0,
        fuel_moisture_percent=12.0,
        fuel_class=FireSpreadFuelClass.SHRUBS,
        horizon_minutes=30,
    )
    defaults.update(overrides)
    return FireSpreadInput(**defaults)


def make_result(**overrides) -> FireSpreadInputResult:
    defaults = dict(
        status=FireSpreadInputStatus.READY,
        input_data=make_input(),
        fire_event_id=10,
        severity_assessment_id=500,
        weather_observation_id=101,
    )
    defaults.update(overrides)
    return FireSpreadInputResult(**defaults)


def test_valid_ready_result():
    result = make_result()
    assert result.status is FireSpreadInputStatus.READY
    assert result.input_data is not None


@pytest.mark.parametrize(
    ("status", "reason"),
    [(FireSpreadInputStatus.INSUFFICIENT_DATA, _REASON), (FireSpreadInputStatus.INACTIVE_EVENT, None)],
)
def test_valid_non_ready_result_without_input_data(status, reason):
    result = make_result(
        status=status,
        input_data=None,
        severity_assessment_id=None,
        weather_observation_id=None,
        insufficient_data_reason=reason,
    )
    assert result.status is status
    assert result.input_data is None
    assert result.insufficient_data_reason is reason


def test_non_ready_result_may_still_carry_partial_traceability():
    result = make_result(
        status=FireSpreadInputStatus.INSUFFICIENT_DATA,
        input_data=None,
        weather_observation_id=None,
        insufficient_data_reason=_REASON,
    )
    assert result.severity_assessment_id == 500
    assert result.weather_observation_id is None


def test_ready_result_has_no_reason_by_default():
    assert make_result().insufficient_data_reason is None


def test_ready_result_with_reason_rejected():
    with pytest.raises(ValueError):
        make_result(insufficient_data_reason=_REASON)


def test_inactive_event_result_with_reason_rejected():
    with pytest.raises(ValueError):
        make_result(status=FireSpreadInputStatus.INACTIVE_EVENT, input_data=None, insufficient_data_reason=_REASON)


@pytest.mark.parametrize("reason", [None, "missing_vegetation"])
def test_insufficient_data_result_requires_a_reason_enum(reason):
    with pytest.raises(ValueError):
        make_result(status=FireSpreadInputStatus.INSUFFICIENT_DATA, input_data=None, insufficient_data_reason=reason)


def test_ready_without_input_data_rejected():
    with pytest.raises(ValueError):
        make_result(input_data=None)


def test_ready_without_severity_assessment_id_rejected():
    with pytest.raises(ValueError):
        make_result(severity_assessment_id=None)


def test_ready_without_weather_observation_id_rejected():
    with pytest.raises(ValueError):
        make_result(weather_observation_id=None)


def test_non_ready_with_input_data_rejected():
    with pytest.raises(ValueError):
        make_result(status=FireSpreadInputStatus.INSUFFICIENT_DATA, insufficient_data_reason=_REASON)


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "10"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_result(fire_event_id=fire_event_id)


@pytest.mark.parametrize("severity_assessment_id", [0, -1, True])
def test_invalid_severity_assessment_id_rejected(severity_assessment_id):
    with pytest.raises(ValueError):
        make_result(severity_assessment_id=severity_assessment_id)


@pytest.mark.parametrize("weather_observation_id", [0, -1, True])
def test_invalid_weather_observation_id_rejected(weather_observation_id):
    with pytest.raises(ValueError):
        make_result(weather_observation_id=weather_observation_id)


def test_invalid_status_type_rejected():
    with pytest.raises(ValueError):
        make_result(status="ready")
