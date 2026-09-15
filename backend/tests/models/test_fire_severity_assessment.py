"""Tests for fire-severity assessment domain models."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math

import pytest

from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models import FireSeverityAssessment, FireSeverityAssessmentStatus, FireSeverityLevel

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def make_assessment(**overrides) -> FireSeverityAssessment:
    defaults = dict(
        fire_event_id=1,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=55.0,
        level=FireSeverityLevel.HIGH,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        vegetation_dataset_year=2019,
        vegetation_radius_km=1.0,
        vegetation_dominant_land_cover="Tree cover",
        vegetation_fuel_score=0.9,
    )
    defaults.update(overrides)
    return FireSeverityAssessment(**defaults)


def test_fire_severity_assessment_status_values():
    assert [status.value for status in FireSeverityAssessmentStatus] == [
        "valid",
        "insufficient_data",
        "inactive_event",
    ]


def test_valid_assessment_with_score_level_and_vegetation():
    assessment = make_assessment()

    assert assessment.status is FireSeverityAssessmentStatus.VALID
    assert assessment.score == 55.0
    assert assessment.level is FireSeverityLevel.HIGH
    assert assessment.vegetation_fuel_score == 0.9


def test_valid_assessment_without_vegetation():
    assessment = make_assessment(
        vegetation_source=None,
        vegetation_dataset_year=None,
        vegetation_radius_km=None,
        vegetation_dominant_land_cover=None,
        vegetation_fuel_score=None,
    )

    assert assessment.status is FireSeverityAssessmentStatus.VALID
    assert assessment.vegetation_fuel_score is None


def test_valid_insufficient_data_assessment():
    assessment = make_assessment(
        status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    assert assessment.status is FireSeverityAssessmentStatus.INSUFFICIENT_DATA


def test_valid_inactive_event_assessment():
    assessment = make_assessment(
        status=FireSeverityAssessmentStatus.INACTIVE_EVENT,
        score=None,
        level=None,
        vegetation_source=None,
        vegetation_dataset_year=None,
        vegetation_radius_km=None,
        vegetation_dominant_land_cover=None,
        vegetation_fuel_score=None,
    )

    assert assessment.status is FireSeverityAssessmentStatus.INACTIVE_EVENT


def test_valid_without_score_rejected():
    with pytest.raises(ValueError):
        make_assessment(score=None)


def test_valid_without_level_rejected():
    with pytest.raises(ValueError):
        make_assessment(level=None)


def test_insufficient_data_with_score_rejected():
    with pytest.raises(ValueError):
        make_assessment(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA, level=None)


def test_inactive_event_with_level_rejected():
    with pytest.raises(ValueError):
        make_assessment(status=FireSeverityAssessmentStatus.INACTIVE_EVENT, score=None)


@pytest.mark.parametrize("fire_event_id", [0, -1, True, "1"])
def test_invalid_fire_event_id_rejected(fire_event_id):
    with pytest.raises(ValueError):
        make_assessment(fire_event_id=fire_event_id)


def test_naive_assessed_at_rejected():
    with pytest.raises(ValueError):
        make_assessment(assessed_at=datetime(2026, 9, 14, 12, 0))


@pytest.mark.parametrize("score", [-0.1, 100.1, math.nan, math.inf, -math.inf])
def test_invalid_score_rejected(score):
    with pytest.raises(ValueError):
        make_assessment(score=score)


@pytest.mark.parametrize("vegetation_fuel_score", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_invalid_vegetation_fuel_score_rejected(vegetation_fuel_score):
    with pytest.raises(ValueError):
        make_assessment(vegetation_fuel_score=vegetation_fuel_score)


@pytest.mark.parametrize("vegetation_radius_km", [0, -1, math.nan, math.inf, -math.inf])
def test_invalid_vegetation_radius_rejected(vegetation_radius_km):
    with pytest.raises(ValueError):
        make_assessment(vegetation_radius_km=vegetation_radius_km)


@pytest.mark.parametrize("vegetation_dataset_year", [0, -1, True, "2019"])
def test_invalid_dataset_year_rejected(vegetation_dataset_year):
    with pytest.raises(ValueError):
        make_assessment(vegetation_dataset_year=vegetation_dataset_year)


def test_empty_methodology_rejected():
    with pytest.raises(ValueError):
        make_assessment(methodology=" ")


def test_empty_methodology_version_rejected():
    with pytest.raises(ValueError):
        make_assessment(methodology_version=" ")


def test_assessment_is_immutable():
    assessment = make_assessment()

    with pytest.raises(FrozenInstanceError):
        assessment.score = 60.0
