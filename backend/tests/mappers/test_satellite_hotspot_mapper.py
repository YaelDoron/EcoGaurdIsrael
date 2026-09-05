"""Unit tests for SatelliteHotspotMapper.

Inputs are deterministic raw FIRMS dictionaries. These tests perform no
network access, database access, or environment-variable reads.
"""
from datetime import datetime

import pytest

from src.mappers.exceptions import (
    InvalidSatelliteDetectionTimeError,
    MissingRequiredSatelliteFieldError,
    SatelliteHotspotMappingError,
)
from src.mappers.satellite_hotspot_mapper import SatelliteHotspotMapper
from src.models.satellite_hotspot import SatelliteHotspot

RAW_DETECTION = {
    "latitude": "32.7001",
    "longitude": "35.0123",
    "bright_ti4": "341.2",
    "scan": "0.39",
    "track": "0.36",
    "acq_date": "2026-09-05",
    "acq_time": "1432",
    "satellite": "N20",
    "instrument": "VIIRS",
    "confidence": "n",
    "version": "2.0NRT",
    "bright_ti5": "298.5",
    "frp": "12.4",
    "daynight": "D",
}


def map_detection(**overrides):
    raw = dict(RAW_DETECTION)
    raw.update(overrides)
    return SatelliteHotspotMapper.map_detection(raw)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_complete_valid_firms_dict_maps_to_satellite_hotspot():
    hotspot = map_detection()

    assert hotspot == SatelliteHotspot(
        latitude=32.7001,
        longitude=35.0123,
        detected_at=datetime(2026, 9, 5, 14, 32),
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite="N20",
        instrument="VIIRS",
        day_night="D",
    )


def test_latitude_becomes_float():
    assert isinstance(map_detection().latitude, float)


def test_longitude_becomes_float():
    assert isinstance(map_detection().longitude, float)


def test_frp_becomes_float():
    assert map_detection().frp == 12.4


def test_bright_ti4_becomes_brightness_float():
    assert map_detection().brightness == 341.2


def test_acq_date_and_acq_time_become_detected_at():
    assert map_detection().detected_at == datetime(2026, 9, 5, 14, 32)


def test_satellite_maps_correctly():
    assert map_detection().satellite == "N20"


def test_instrument_maps_correctly():
    assert map_detection().instrument == "VIIRS"


def test_confidence_remains_raw_string():
    assert map_detection(confidence="h").confidence == "h"


def test_daynight_maps_to_day_night():
    assert map_detection(daynight="N").day_night == "N"


def test_unknown_firms_fields_are_ignored():
    hotspot = map_detection(unused_firms_column="ignored")

    assert not hasattr(hotspot, "unused_firms_column")


def test_map_detections_preserves_order():
    raw_detections = [
        {**RAW_DETECTION, "latitude": "32.1"},
        {**RAW_DETECTION, "latitude": "32.2"},
    ]

    hotspots = SatelliteHotspotMapper.map_detections(raw_detections)

    assert [hotspot.latitude for hotspot in hotspots] == [32.1, 32.2]


def test_map_detections_surfaces_invalid_item_error():
    raw_detections = [RAW_DETECTION, {**RAW_DETECTION, "latitude": "bad"}]

    with pytest.raises(MissingRequiredSatelliteFieldError):
        SatelliteHotspotMapper.map_detections(raw_detections)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_time", "expected"),
    [
        ("1432", datetime(2026, 9, 5, 14, 32)),
        ("0037", datetime(2026, 9, 5, 0, 37)),
        ("0000", datetime(2026, 9, 5, 0, 0)),
        ("2359", datetime(2026, 9, 5, 23, 59)),
    ],
)
def test_acq_time_parses_expected_times(raw_time, expected):
    assert map_detection(acq_time=raw_time).detected_at == expected


def test_invalid_hour_fails():
    with pytest.raises(InvalidSatelliteDetectionTimeError):
        map_detection(acq_time="2430")


def test_invalid_minute_fails():
    with pytest.raises(InvalidSatelliteDetectionTimeError):
        map_detection(acq_time="2360")


@pytest.mark.parametrize("acq_time", ["37", "123", "12345", "12:30", "abcd"])
def test_malformed_acq_time_fails(acq_time):
    with pytest.raises(InvalidSatelliteDetectionTimeError):
        map_detection(acq_time=acq_time)


def test_malformed_acq_date_fails():
    with pytest.raises(InvalidSatelliteDetectionTimeError):
        map_detection(acq_date="09/05/2026")


def test_missing_acq_date_fails():
    raw = dict(RAW_DETECTION)
    del raw["acq_date"]

    with pytest.raises(MissingRequiredSatelliteFieldError):
        SatelliteHotspotMapper.map_detection(raw)


def test_missing_acq_time_fails():
    raw = dict(RAW_DETECTION)
    del raw["acq_time"]

    with pytest.raises(MissingRequiredSatelliteFieldError):
        SatelliteHotspotMapper.map_detection(raw)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


def test_missing_latitude_raises_mapping_error():
    raw = dict(RAW_DETECTION)
    del raw["latitude"]

    with pytest.raises(MissingRequiredSatelliteFieldError):
        SatelliteHotspotMapper.map_detection(raw)


def test_missing_longitude_raises_mapping_error():
    raw = dict(RAW_DETECTION)
    del raw["longitude"]

    with pytest.raises(MissingRequiredSatelliteFieldError):
        SatelliteHotspotMapper.map_detection(raw)


def test_empty_latitude_raises_mapping_error():
    with pytest.raises(MissingRequiredSatelliteFieldError):
        map_detection(latitude="")


def test_non_numeric_latitude_raises_mapping_error():
    with pytest.raises(MissingRequiredSatelliteFieldError):
        map_detection(latitude="north")


def test_non_numeric_longitude_raises_mapping_error():
    with pytest.raises(MissingRequiredSatelliteFieldError):
        map_detection(longitude="east")


@pytest.mark.parametrize(("field_name", "value"), [("latitude", "120.0"), ("longitude", "220.0")])
def test_out_of_range_coordinates_raise_mapping_error(field_name, value):
    with pytest.raises(SatelliteHotspotMappingError):
        map_detection(**{field_name: value})


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_missing_frp_becomes_none():
    raw = dict(RAW_DETECTION)
    del raw["frp"]

    assert SatelliteHotspotMapper.map_detection(raw).frp is None


def test_empty_frp_becomes_none():
    assert map_detection(frp="").frp is None


def test_non_numeric_frp_becomes_none():
    assert map_detection(frp="abc").frp is None


def test_missing_bright_ti4_becomes_brightness_none():
    raw = dict(RAW_DETECTION)
    del raw["bright_ti4"]

    assert SatelliteHotspotMapper.map_detection(raw).brightness is None


def test_invalid_bright_ti4_becomes_none():
    assert map_detection(bright_ti4="abc").brightness is None


def test_missing_confidence_becomes_none():
    raw = dict(RAW_DETECTION)
    del raw["confidence"]

    assert SatelliteHotspotMapper.map_detection(raw).confidence is None


def test_empty_confidence_becomes_none():
    assert map_detection(confidence=" ").confidence is None


def test_missing_satellite_becomes_none():
    raw = dict(RAW_DETECTION)
    del raw["satellite"]

    assert SatelliteHotspotMapper.map_detection(raw).satellite is None


def test_missing_instrument_becomes_none():
    raw = dict(RAW_DETECTION)
    del raw["instrument"]

    assert SatelliteHotspotMapper.map_detection(raw).instrument is None


def test_missing_daynight_becomes_none():
    raw = dict(RAW_DETECTION)
    del raw["daynight"]

    assert SatelliteHotspotMapper.map_detection(raw).day_night is None
