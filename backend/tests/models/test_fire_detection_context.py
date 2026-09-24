"""Tests for the FireDetectionContext domain model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_detection_context import FireDetectionContext

ASSESSED_AT = datetime(2026, 9, 14, 11, 40, tzinfo=timezone.utc)


def available_context(**overrides) -> FireDetectionContext:
    values = dict(
        fire_danger_available=True,
        fire_danger_score=42.5,
        fire_danger_age_minutes=20.0,
        fire_danger_level=FireDangerLevel.VERY_HIGH,
        fire_danger_assessment_id=7,
        fire_danger_assessed_at=ASSESSED_AT,
    )
    values.update(overrides)
    return FireDetectionContext(**values)


def test_unavailable_context_carries_no_fire_danger_values():
    context = FireDetectionContext.unavailable()

    assert context.fire_danger_available is False
    assert context.fire_danger_score is None
    assert context.fire_danger_age_minutes is None
    assert context.fire_danger_level is None
    assert context.fire_danger_assessment_id is None
    assert context.fire_danger_assessed_at is None


def test_available_context_exposes_score_age_and_traceability():
    context = available_context()

    assert context.fire_danger_available is True
    assert context.fire_danger_score == 42.5
    assert context.fire_danger_age_minutes == 20.0
    assert context.fire_danger_level is FireDangerLevel.VERY_HIGH
    assert context.fire_danger_assessment_id == 7
    assert context.fire_danger_assessed_at == ASSESSED_AT


def test_zero_score_is_a_real_available_observation_distinct_from_unavailable():
    context = available_context(fire_danger_score=0.0, fire_danger_level=FireDangerLevel.LOW)

    assert context.fire_danger_available is True
    assert context.fire_danger_score == 0.0
    assert context != FireDetectionContext.unavailable()


def test_zero_age_is_allowed():
    assert available_context(fire_danger_age_minutes=0.0).fire_danger_age_minutes == 0.0


def test_context_is_immutable():
    with pytest.raises(FrozenInstanceError):
        available_context().fire_danger_score = 1.0


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("fire_danger_score", 0.0),
        ("fire_danger_age_minutes", 5.0),
        ("fire_danger_level", FireDangerLevel.LOW),
        ("fire_danger_assessment_id", 3),
        ("fire_danger_assessed_at", ASSESSED_AT),
    ],
)
def test_unavailable_context_rejects_any_fire_danger_value(field_name, value):
    with pytest.raises(ValueError):
        FireDetectionContext(fire_danger_available=False, **{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "fire_danger_score",
        "fire_danger_age_minutes",
        "fire_danger_level",
        "fire_danger_assessment_id",
        "fire_danger_assessed_at",
    ],
)
def test_available_context_requires_every_fire_danger_field(field_name):
    with pytest.raises(ValueError):
        available_context(**{field_name: None})


@pytest.mark.parametrize(
    "overrides",
    [
        {"fire_danger_score": float("nan")},
        {"fire_danger_score": float("inf")},
        {"fire_danger_score": -0.1},
        {"fire_danger_score": 1000.0},
        {"fire_danger_score": True},
        {"fire_danger_age_minutes": -1.0},
        {"fire_danger_age_minutes": float("nan")},
        {"fire_danger_level": "high"},
        {"fire_danger_assessment_id": 0},
        {"fire_danger_assessment_id": True},
        {"fire_danger_assessed_at": datetime(2026, 9, 14, 11, 40)},
    ],
)
def test_available_context_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        available_context(**overrides)


def test_fire_danger_available_must_be_a_bool():
    with pytest.raises(ValueError):
        FireDetectionContext(fire_danger_available=1)
