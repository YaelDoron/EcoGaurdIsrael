"""Tests for fire-danger assessment domain model invariants."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from src.calculators.fire_danger.ffwi_config import (
    FFWI_MAX_SCORE,
    FFWI_METHODOLOGY_NAME,
    FFWI_METHODOLOGY_VERSION,
    FFWI_MIN_SCORE,
)
from src.models import FireDangerAssessment, FireDangerAssessmentStatus, FireDangerLevel

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_assessment(**overrides) -> FireDangerAssessment:
    defaults = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=ASSESSED_AT,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireDangerAssessment(**defaults)


def test_valid_assessment_construction():
    assessment = make_assessment()

    assert assessment.area_id == "area-carmel"
    assert assessment.score == 42.5
    assert assessment.level is FireDangerLevel.VERY_HIGH
    assert assessment.status is FireDangerAssessmentStatus.VALID
    assert assessment.methodology == FFWI_METHODOLOGY_NAME
    assert assessment.methodology_version == FFWI_METHODOLOGY_VERSION


def test_valid_assessment_with_score_none_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(score=None)


def test_valid_assessment_with_level_none_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(level=None)


def test_valid_assessment_score_below_valid_range_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(score=FFWI_MIN_SCORE - 0.01)


def test_valid_assessment_score_above_valid_range_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(score=FFWI_MAX_SCORE + 0.01)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_valid_assessment_non_finite_score_is_rejected(score):
    with pytest.raises(ValueError):
        make_assessment(score=score)


def test_insufficient_data_assessment_with_none_score_and_level_is_valid():
    assessment = make_assessment(
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    assert assessment.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA
    assert assessment.score is None
    assert assessment.level is None


def test_insufficient_data_assessment_with_score_present_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=1.0, level=None)


def test_insufficient_data_assessment_with_level_present_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(
            status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
            score=None,
            level=FireDangerLevel.LOW,
        )


def test_naive_assessed_at_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(assessed_at=datetime(2026, 9, 14, 12, 0))


@pytest.mark.parametrize(
    "field_name, invalid_value",
    [
        ("area_latitude", -90.1),
        ("area_latitude", 90.1),
        ("area_latitude", math.nan),
        ("area_longitude", -180.1),
        ("area_longitude", 180.1),
        ("area_longitude", math.inf),
    ],
)
def test_invalid_coordinates_are_rejected(field_name, invalid_value):
    with pytest.raises(ValueError):
        make_assessment(**{field_name: invalid_value})


@pytest.mark.parametrize("radius_km", [0, -1, math.nan, math.inf])
def test_invalid_radius_is_rejected(radius_km):
    with pytest.raises(ValueError):
        make_assessment(area_radius_km=radius_km)


def test_empty_methodology_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(methodology=" ")


def test_empty_methodology_version_is_rejected():
    with pytest.raises(ValueError):
        make_assessment(methodology_version="")


def test_assessment_is_immutable():
    assessment = make_assessment()

    with pytest.raises(FrozenInstanceError):
        assessment.score = 50.0
