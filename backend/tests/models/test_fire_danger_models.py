"""Tests for fire-danger domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from src.models import (
    AssessmentArea,
    FireDangerCalculation,
    FireDangerInput,
    FireDangerInputResult,
    FireDangerInputStatus,
    FireDangerLevel,
)


def test_valid_fire_danger_input_construction():
    input_data = FireDangerInput(
        temperature_c=32.5,
        relative_humidity_pct=20.0,
        wind_speed_kmh=15.0,
    )

    assert input_data.temperature_c == 32.5
    assert input_data.relative_humidity_pct == 20.0
    assert input_data.wind_speed_kmh == 15.0


def test_fire_danger_input_is_immutable():
    input_data = FireDangerInput(temperature_c=20, relative_humidity_pct=50, wind_speed_kmh=5)

    with pytest.raises(FrozenInstanceError):
        input_data.temperature_c = 21


@pytest.mark.parametrize("relative_humidity_pct", [-1, 101])
def test_invalid_relative_humidity_is_rejected(relative_humidity_pct):
    with pytest.raises(ValueError):
        FireDangerInput(
            temperature_c=20,
            relative_humidity_pct=relative_humidity_pct,
            wind_speed_kmh=5,
        )


def test_invalid_wind_speed_is_rejected():
    with pytest.raises(ValueError):
        FireDangerInput(temperature_c=20, relative_humidity_pct=50, wind_speed_kmh=-1)


@pytest.mark.parametrize("non_finite_value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field_name", ["temperature_c", "relative_humidity_pct", "wind_speed_kmh"])
def test_non_finite_values_are_rejected(field_name, non_finite_value):
    values = {
        "temperature_c": 20,
        "relative_humidity_pct": 50,
        "wind_speed_kmh": 5,
    }
    values[field_name] = non_finite_value

    with pytest.raises(ValueError):
        FireDangerInput(**values)


def test_fire_danger_level_values():
    assert [level.value for level in FireDangerLevel] == [
        "low",
        "moderate",
        "high",
        "very_high",
        "extreme",
    ]


def test_fire_danger_calculation_construction():
    calculation = FireDangerCalculation(score=42.5, level=FireDangerLevel.VERY_HIGH)

    assert calculation.score == 42.5
    assert calculation.level is FireDangerLevel.VERY_HIGH


def test_fire_danger_calculation_is_immutable():
    calculation = FireDangerCalculation(score=10.0, level=FireDangerLevel.LOW)

    with pytest.raises(FrozenInstanceError):
        calculation.score = 11.0


def test_valid_assessment_area_construction():
    area = AssessmentArea(
        id="carmel",
        name="Carmel Forest",
        latitude=32.731,
        longitude=35.046,
        radius_km=7.5,
    )

    assert area.id == "carmel"
    assert area.name == "Carmel Forest"
    assert area.latitude == 32.731
    assert area.longitude == 35.046
    assert area.radius_km == 7.5


def test_assessment_area_is_immutable():
    area = AssessmentArea(id="area-1", name="Area 1", latitude=32.0, longitude=35.0, radius_km=5.0)

    with pytest.raises(FrozenInstanceError):
        area.radius_km = 10.0


@pytest.mark.parametrize("field_name", ["id", "name"])
def test_assessment_area_rejects_blank_text_fields(field_name):
    values = {
        "id": "area-1",
        "name": "Area 1",
        "latitude": 32.0,
        "longitude": 35.0,
        "radius_km": 5.0,
    }
    values[field_name] = " "

    with pytest.raises(ValueError):
        AssessmentArea(**values)


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("latitude", -90.1),
        ("latitude", 90.1),
        ("longitude", -180.1),
        ("longitude", 180.1),
        ("radius_km", 0),
        ("radius_km", -1),
        ("latitude", math.nan),
        ("longitude", math.inf),
        ("radius_km", -math.inf),
    ],
)
def test_assessment_area_rejects_invalid_coordinates_and_radius(field_name, invalid_value):
    values = {
        "id": "area-1",
        "name": "Area 1",
        "latitude": 32.0,
        "longitude": 35.0,
        "radius_km": 5.0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        AssessmentArea(**values)


def test_fire_danger_input_status_values():
    assert FireDangerInputStatus.READY.value == "ready"
    assert FireDangerInputStatus.INSUFFICIENT_DATA.value == "insufficient_data"


def test_ready_fire_danger_input_result_requires_input_and_traceability_ids():
    input_data = FireDangerInput(temperature_c=30, relative_humidity_pct=25, wind_speed_kmh=12)

    result = FireDangerInputResult(
        status=FireDangerInputStatus.READY,
        input_data=input_data,
        observation_ids=[101, 102],
        station_ids=[1, 2],
    )

    assert result.input_data == input_data
    assert result.observation_ids == (101, 102)
    assert result.station_ids == (1, 2)


@pytest.mark.parametrize(
    "observation_ids, station_ids",
    [
        ((), (1,)),
        ((101,), ()),
    ],
)
def test_ready_fire_danger_input_result_rejects_missing_traceability_ids(
    observation_ids,
    station_ids,
):
    with pytest.raises(ValueError):
        FireDangerInputResult(
            status=FireDangerInputStatus.READY,
            input_data=FireDangerInput(temperature_c=30, relative_humidity_pct=25, wind_speed_kmh=12),
            observation_ids=observation_ids,
            station_ids=station_ids,
        )


def test_ready_fire_danger_input_result_rejects_missing_input_data():
    with pytest.raises(ValueError):
        FireDangerInputResult(
            status=FireDangerInputStatus.READY,
            input_data=None,
            observation_ids=(101,),
            station_ids=(1,),
        )


def test_insufficient_fire_danger_input_result_requires_no_input_data():
    result = FireDangerInputResult(
        status=FireDangerInputStatus.INSUFFICIENT_DATA,
        input_data=None,
        observation_ids=(),
        station_ids=(),
    )

    assert result.input_data is None


def test_insufficient_fire_danger_input_result_rejects_input_data():
    with pytest.raises(ValueError):
        FireDangerInputResult(
            status=FireDangerInputStatus.INSUFFICIENT_DATA,
            input_data=FireDangerInput(temperature_c=30, relative_humidity_pct=25, wind_speed_kmh=12),
            observation_ids=(),
            station_ids=(),
        )
