"""Unit tests for the SatelliteHotspot model."""
from datetime import datetime

import pytest

from src.models.satellite_hotspot import SatelliteHotspot

FIXED_DETECTED_AT = datetime(2026, 9, 5, 14, 32, 0)


def make_hotspot(**overrides):
    defaults = dict(
        latitude=32.7001,
        longitude=35.0123,
        detected_at=FIXED_DETECTED_AT,
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite="N20",
        instrument="VIIRS",
        day_night="D",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


def test_valid_satellite_hotspot_is_created_successfully():
    hotspot = make_hotspot()

    assert hotspot.latitude == 32.7001
    assert hotspot.longitude == 35.0123
    assert hotspot.detected_at == FIXED_DETECTED_AT
    assert hotspot.confidence == "n"
    assert hotspot.frp == 12.4
    assert hotspot.brightness == 341.2
    assert hotspot.satellite == "N20"
    assert hotspot.instrument == "VIIRS"
    assert hotspot.day_night == "D"


@pytest.mark.parametrize("latitude", [-90, 90])
def test_latitude_boundaries_are_allowed(latitude):
    assert make_hotspot(latitude=latitude).latitude == latitude


@pytest.mark.parametrize("latitude", [-90.1, 90.1])
def test_latitude_outside_range_fails(latitude):
    with pytest.raises(ValueError):
        make_hotspot(latitude=latitude)


@pytest.mark.parametrize("longitude", [-180, 180])
def test_longitude_boundaries_are_allowed(longitude):
    assert make_hotspot(longitude=longitude).longitude == longitude


@pytest.mark.parametrize("longitude", [-180.1, 180.1])
def test_longitude_outside_range_fails(longitude):
    with pytest.raises(ValueError):
        make_hotspot(longitude=longitude)


@pytest.mark.parametrize("detected_at", ["2026-09-05T14:32:00", None, 12345])
def test_detected_at_must_be_datetime(detected_at):
    with pytest.raises(ValueError):
        make_hotspot(detected_at=detected_at)


def test_frp_none_is_allowed():
    assert make_hotspot(frp=None).frp is None


def test_frp_zero_is_allowed():
    assert make_hotspot(frp=0).frp == 0


def test_negative_frp_fails():
    with pytest.raises(ValueError):
        make_hotspot(frp=-0.1)


def test_brightness_none_is_allowed():
    assert make_hotspot(brightness=None).brightness is None


def test_brightness_zero_is_allowed():
    assert make_hotspot(brightness=0).brightness == 0


def test_negative_brightness_fails():
    with pytest.raises(ValueError):
        make_hotspot(brightness=-0.1)


def test_all_optional_string_fields_may_be_none():
    hotspot = make_hotspot(confidence=None, satellite=None, instrument=None, day_night=None)

    assert hotspot.confidence is None
    assert hotspot.satellite is None
    assert hotspot.instrument is None
    assert hotspot.day_night is None
