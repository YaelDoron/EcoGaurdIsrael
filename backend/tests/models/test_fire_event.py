"""Tests for the FireEvent domain model."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math

import pytest

from src.models import FireEvent, FireEventStatus

DETECTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
UPDATED_AT = DETECTED_AT + timedelta(minutes=10)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=DETECTED_AT,
        updated_at=UPDATED_AT,
        status=FireEventStatus.SUSPECTED,
        detection_confidence=0.6,
        methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version="1.0",
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


@pytest.mark.parametrize(
    "status",
    [
        FireEventStatus.SUSPECTED,
        FireEventStatus.CONFIRMED,
        FireEventStatus.RESOLVED,
        FireEventStatus.DISMISSED,
    ],
)
def test_valid_fire_event_statuses(status):
    event = make_event(status=status)

    assert event.status is status


@pytest.mark.parametrize("latitude", [-90.1, 90.1])
def test_invalid_latitude_rejected(latitude):
    with pytest.raises(ValueError):
        make_event(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180.1, 180.1])
def test_invalid_longitude_rejected(longitude):
    with pytest.raises(ValueError):
        make_event(longitude=longitude)


@pytest.mark.parametrize("coordinate", [math.nan, math.inf, -math.inf])
def test_nan_and_infinite_coordinates_rejected(coordinate):
    with pytest.raises(ValueError):
        make_event(latitude=coordinate)
    with pytest.raises(ValueError):
        make_event(longitude=coordinate)


def test_naive_detected_at_rejected():
    with pytest.raises(ValueError):
        make_event(detected_at=datetime(2026, 9, 14, 12, 0))


def test_naive_updated_at_rejected():
    with pytest.raises(ValueError):
        make_event(updated_at=datetime(2026, 9, 14, 12, 10))


def test_updated_at_before_detected_at_rejected():
    with pytest.raises(ValueError):
        make_event(updated_at=DETECTED_AT - timedelta(seconds=1))


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        make_event(detection_confidence=confidence)


@pytest.mark.parametrize("methodology", ["", "   ", None])
def test_empty_methodology_rejected(methodology):
    with pytest.raises(ValueError):
        make_event(methodology=methodology)


@pytest.mark.parametrize("methodology_version", ["", "   ", None])
def test_empty_methodology_version_rejected(methodology_version):
    with pytest.raises(ValueError):
        make_event(methodology_version=methodology_version)


def test_fire_event_is_immutable():
    event = make_event()

    with pytest.raises(FrozenInstanceError):
        event.latitude = 33.0
