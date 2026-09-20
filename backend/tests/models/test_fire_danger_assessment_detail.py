"""Tests for Task A4 Fire Danger assessment-detail read models (DTOs).

Pure, read-only presentation contract: must not recompute FFWI or import
calculation/persistence/HTTP modules, matching test_active_fire_events.py's
own precedent for the same kind of read model.
"""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.fire_danger_assessment_detail import (
    FireDangerAssessmentDetail,
    FireDangerAssessmentWeatherInputSummary,
)
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
OBSERVED_AT = datetime(2026, 9, 19, 11, 45)


def weather_input(**overrides) -> FireDangerAssessmentWeatherInputSummary:
    values = {
        "observation_id": 101,
        "station_external_id": 1001,
        "station_name": "Station 1001",
        "observed_at": OBSERVED_AT,
    }
    values.update(overrides)
    return FireDangerAssessmentWeatherInputSummary(**values)


def detail(**overrides) -> FireDangerAssessmentDetail:
    values = {
        "assessment_id": 5,
        "area_id": "area-carmel",
        "area_name": "Carmel",
        "area_latitude": 32.731,
        "area_longitude": 35.046,
        "area_radius_km": 5.0,
        "status": FireDangerAssessmentStatus.VALID,
        "score": 42.5,
        "level": FireDangerLevel.VERY_HIGH,
        "assessed_at": ASSESSED_AT,
        "methodology": "FOSBERG_FFWI",
        "methodology_version": "1.0",
        "weather_inputs": (),
    }
    values.update(overrides)
    return FireDangerAssessmentDetail(**values)


# ---------------------------------------------------------------------------
# FireDangerAssessmentWeatherInputSummary
# ---------------------------------------------------------------------------


def test_weather_input_accepts_naive_observed_at():
    result = weather_input(observed_at=OBSERVED_AT)

    assert result.observed_at.tzinfo is None


def test_weather_input_accepts_aware_observed_at():
    result = weather_input(observed_at=OBSERVED_AT.replace(tzinfo=timezone.utc))

    assert result.observed_at.tzinfo is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_id": 0},
        {"station_external_id": 0},
        {"station_name": ""},
        {"observed_at": "2026-09-19"},
    ],
)
def test_weather_input_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        weather_input(**overrides)


def test_weather_input_is_frozen():
    result = weather_input()
    with pytest.raises(FrozenInstanceError):
        result.observation_id = 999


# ---------------------------------------------------------------------------
# FireDangerAssessmentDetail
# ---------------------------------------------------------------------------


def test_detail_accepts_valid_result_with_weather_inputs():
    result = detail(weather_inputs=(weather_input(),))

    assert result.score == pytest.approx(42.5)
    assert len(result.weather_inputs) == 1


def test_detail_accepts_insufficient_data_with_no_weather_inputs():
    result = detail(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None, weather_inputs=())

    assert result.score is None
    assert result.level is None
    assert result.weather_inputs == ()


def test_detail_coerces_iterable_weather_inputs_to_tuple():
    result = detail(weather_inputs=[weather_input()])

    assert isinstance(result.weather_inputs, tuple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"assessment_id": 0},
        {"area_id": ""},
        {"area_name": ""},
        {"area_latitude": 91.0},
        {"area_longitude": -181.0},
        {"area_radius_km": 0},
        {"status": "valid"},
        {"score": float("nan")},
        {"level": "very_high"},
        {"assessed_at": ASSESSED_AT.replace(tzinfo=None)},
        {"methodology": ""},
        {"methodology_version": ""},
        {"weather_inputs": (object(),)},
    ],
)
def test_detail_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        detail(**overrides)


def test_detail_is_frozen():
    result = detail()
    with pytest.raises(FrozenInstanceError):
        result.score = 1.0


# ---------------------------------------------------------------------------
# Architecture guard: pure read/presentation module only
# ---------------------------------------------------------------------------


def test_fire_danger_assessment_detail_does_not_import_calculation_persistence_or_http_modules():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDangerInputService",
        "fastapi",
        "pydantic",
        "simulation",
        "external",
        "sqlalchemy",
        "repositories",
        "database",
        "calculators",
        "services",
    )
    path = Path(__file__).resolve().parents[2] / "src" / "models" / "fire_danger_assessment_detail.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []
