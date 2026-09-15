"""Tests for active wildfire severity domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from src.models import (
    FireSeverityCalculation,
    FireSeverityInput,
    FireSeverityInputResult,
    FireSeverityInputStatus,
    FireSeverityLevel,
    VegetationData,
)


def test_fire_severity_level_values():
    assert [level.value for level in FireSeverityLevel] == [
        "low",
        "moderate",
        "high",
        "critical",
    ]


def test_fire_severity_input_status_values():
    assert [status.value for status in FireSeverityInputStatus] == [
        "ready",
        "insufficient_data",
        "inactive_event",
    ]


def test_valid_fire_severity_input_construction():
    input_data = FireSeverityInput(
        frp_mw=25.0,
        wind_speed_kmh=12.5,
        relative_humidity_pct=40.0,
        vegetation_fuel_score=0.7,
    )

    assert input_data.frp_mw == 25.0
    assert input_data.wind_speed_kmh == 12.5
    assert input_data.relative_humidity_pct == 40.0
    assert input_data.vegetation_fuel_score == 0.7


def test_fire_severity_input_allows_missing_vegetation():
    input_data = FireSeverityInput(
        frp_mw=25.0,
        wind_speed_kmh=12.5,
        relative_humidity_pct=40.0,
    )

    assert input_data.vegetation_fuel_score is None


def test_valid_vegetation_data_construction_sorts_distribution_to_tuple():
    data = VegetationData(
        fuel_score=0.7,
        dominant_land_cover="Herbaceous vegetation",
        source="COPERNICUS_GLOBAL_LAND_COVER_100M",
        dataset_year=2019,
        radius_km=1.0,
        land_cover_distribution=[("Herbaceous vegetation", 1.0)],
    )

    assert data.fuel_score == 0.7
    assert data.land_cover_distribution == (("Herbaceous vegetation", 1.0),)


def test_vegetation_data_is_immutable():
    data = VegetationData(
        fuel_score=0.7,
        dominant_land_cover="Herbaceous vegetation",
        source="COPERNICUS_GLOBAL_LAND_COVER_100M",
        dataset_year=2019,
        radius_km=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        data.fuel_score = 0.8


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("fuel_score", -0.1),
        ("fuel_score", 1.1),
        ("fuel_score", math.nan),
        ("fuel_score", math.inf),
        ("dataset_year", 0),
        ("dataset_year", -1),
        ("radius_km", 0),
        ("radius_km", -1),
        ("radius_km", math.nan),
    ],
)
def test_vegetation_data_rejects_invalid_values(field_name, invalid_value):
    values = {
        "fuel_score": 0.7,
        "dominant_land_cover": "Herbaceous vegetation",
        "source": "COPERNICUS_GLOBAL_LAND_COVER_100M",
        "dataset_year": 2019,
        "radius_km": 1.0,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        VegetationData(**values)


def test_ready_fire_severity_input_result_requires_input_and_traceability():
    input_data = FireSeverityInput(
        frp_mw=50,
        wind_speed_kmh=20,
        relative_humidity_pct=30,
        vegetation_fuel_score=None,
    )
    result = FireSeverityInputResult(
        status=FireSeverityInputStatus.READY,
        input_data=input_data,
        fire_event_id=10,
        weather_observation_ids=(102, 101),
        satellite_hotspot_ids=(2, 1),
        selected_frp_hotspot_id=2,
    )

    assert result.input_data == input_data
    assert result.weather_observation_ids == (101, 102)
    assert result.satellite_hotspot_ids == (1, 2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_data": None},
        {"weather_observation_ids": ()},
        {"satellite_hotspot_ids": ()},
        {"selected_frp_hotspot_id": None},
        {"selected_frp_hotspot_id": 999},
    ],
)
def test_ready_fire_severity_input_result_rejects_missing_mandatory_traceability(overrides):
    values = {
        "status": FireSeverityInputStatus.READY,
        "input_data": FireSeverityInput(frp_mw=50, wind_speed_kmh=20, relative_humidity_pct=30),
        "fire_event_id": 10,
        "weather_observation_ids": (101,),
        "satellite_hotspot_ids": (1,),
        "selected_frp_hotspot_id": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        FireSeverityInputResult(**values)


@pytest.mark.parametrize(
    "status",
    [FireSeverityInputStatus.INSUFFICIENT_DATA, FireSeverityInputStatus.INACTIVE_EVENT],
)
def test_not_ready_fire_severity_input_result_requires_no_input_data(status):
    result = FireSeverityInputResult(
        status=status,
        input_data=None,
        fire_event_id=10,
    )

    assert result.input_data is None


def test_not_ready_fire_severity_input_result_rejects_input_data():
    with pytest.raises(ValueError):
        FireSeverityInputResult(
            status=FireSeverityInputStatus.INSUFFICIENT_DATA,
            input_data=FireSeverityInput(frp_mw=50, wind_speed_kmh=20, relative_humidity_pct=30),
            fire_event_id=10,
        )


def test_fire_severity_input_is_immutable():
    input_data = FireSeverityInput(frp_mw=1, wind_speed_kmh=2, relative_humidity_pct=3)

    with pytest.raises(FrozenInstanceError):
        input_data.frp_mw = 2


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("frp_mw", -0.1),
        ("frp_mw", math.nan),
        ("frp_mw", math.inf),
        ("frp_mw", -math.inf),
        ("wind_speed_kmh", -0.1),
        ("wind_speed_kmh", math.nan),
        ("wind_speed_kmh", math.inf),
        ("wind_speed_kmh", -math.inf),
        ("relative_humidity_pct", -0.1),
        ("relative_humidity_pct", 100.1),
        ("relative_humidity_pct", math.nan),
        ("relative_humidity_pct", math.inf),
        ("relative_humidity_pct", -math.inf),
        ("vegetation_fuel_score", -0.1),
        ("vegetation_fuel_score", 1.1),
        ("vegetation_fuel_score", math.nan),
        ("vegetation_fuel_score", math.inf),
        ("vegetation_fuel_score", -math.inf),
    ],
)
def test_fire_severity_input_rejects_invalid_values(field_name, invalid_value):
    values = {
        "frp_mw": 10,
        "wind_speed_kmh": 20,
        "relative_humidity_pct": 30,
        "vegetation_fuel_score": 0.5,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        FireSeverityInput(**values)


@pytest.mark.parametrize("field_name", ["frp_mw", "wind_speed_kmh", "relative_humidity_pct"])
def test_fire_severity_input_rejects_bool_required_values(field_name):
    values = {
        "frp_mw": 10,
        "wind_speed_kmh": 20,
        "relative_humidity_pct": 30,
    }
    values[field_name] = True

    with pytest.raises(ValueError):
        FireSeverityInput(**values)


def test_fire_severity_input_rejects_bool_vegetation():
    with pytest.raises(ValueError):
        FireSeverityInput(
            frp_mw=10,
            wind_speed_kmh=20,
            relative_humidity_pct=30,
            vegetation_fuel_score=True,
        )


def test_valid_fire_severity_calculation_construction():
    calculation = FireSeverityCalculation(
        score=42.5,
        level=FireSeverityLevel.MODERATE,
        frp_factor=0.4,
        wind_factor=0.5,
        dryness_factor=0.6,
        vegetation_factor=None,
        available_weight=0.9,
        methodology="method",
        methodology_version="1.0",
    )

    assert calculation.score == 42.5
    assert calculation.level is FireSeverityLevel.MODERATE
    assert calculation.vegetation_factor is None


def test_fire_severity_calculation_is_immutable():
    calculation = FireSeverityCalculation(
        score=42.5,
        level=FireSeverityLevel.MODERATE,
        frp_factor=0.4,
        wind_factor=0.5,
        dryness_factor=0.6,
        vegetation_factor=0.7,
        available_weight=1.0,
        methodology="method",
        methodology_version="1.0",
    )

    with pytest.raises(FrozenInstanceError):
        calculation.score = 43


@pytest.mark.parametrize("score", [-0.1, 100.1, math.nan, math.inf, -math.inf])
def test_fire_severity_calculation_rejects_invalid_score(score):
    with pytest.raises(ValueError):
        FireSeverityCalculation(
            score=score,
            level=FireSeverityLevel.MODERATE,
            frp_factor=0.4,
            wind_factor=0.5,
            dryness_factor=0.6,
            vegetation_factor=0.7,
            available_weight=1.0,
            methodology="method",
            methodology_version="1.0",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("frp_factor", -0.1),
        ("frp_factor", 1.1),
        ("frp_factor", math.nan),
        ("frp_factor", math.inf),
        ("wind_factor", -0.1),
        ("wind_factor", 1.1),
        ("wind_factor", math.nan),
        ("wind_factor", math.inf),
        ("dryness_factor", -0.1),
        ("dryness_factor", 1.1),
        ("dryness_factor", math.nan),
        ("dryness_factor", math.inf),
        ("vegetation_factor", -0.1),
        ("vegetation_factor", 1.1),
        ("vegetation_factor", math.nan),
        ("vegetation_factor", math.inf),
    ],
)
def test_fire_severity_calculation_rejects_invalid_factors(field_name, invalid_value):
    values = {
        "score": 42.5,
        "level": FireSeverityLevel.MODERATE,
        "frp_factor": 0.4,
        "wind_factor": 0.5,
        "dryness_factor": 0.6,
        "vegetation_factor": 0.7,
        "available_weight": 1.0,
        "methodology": "method",
        "methodology_version": "1.0",
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        FireSeverityCalculation(**values)


@pytest.mark.parametrize("available_weight", [0, -0.1, math.nan, math.inf, -math.inf])
def test_fire_severity_calculation_rejects_invalid_available_weight(available_weight):
    with pytest.raises(ValueError):
        FireSeverityCalculation(
            score=42.5,
            level=FireSeverityLevel.MODERATE,
            frp_factor=0.4,
            wind_factor=0.5,
            dryness_factor=0.6,
            vegetation_factor=0.7,
            available_weight=available_weight,
            methodology="method",
            methodology_version="1.0",
        )


def test_fire_severity_calculation_rejects_invalid_level():
    with pytest.raises(ValueError):
        FireSeverityCalculation(
            score=42.5,
            level="moderate",
            frp_factor=0.4,
            wind_factor=0.5,
            dryness_factor=0.6,
            vegetation_factor=0.7,
            available_weight=1.0,
            methodology="method",
            methodology_version="1.0",
        )


@pytest.mark.parametrize("field_name", ["methodology", "methodology_version"])
def test_fire_severity_calculation_rejects_blank_metadata(field_name):
    values = {
        "score": 42.5,
        "level": FireSeverityLevel.MODERATE,
        "frp_factor": 0.4,
        "wind_factor": 0.5,
        "dryness_factor": 0.6,
        "vegetation_factor": 0.7,
        "available_weight": 1.0,
        "methodology": "method",
        "methodology_version": "1.0",
    }
    values[field_name] = " "

    with pytest.raises(ValueError):
        FireSeverityCalculation(**values)
