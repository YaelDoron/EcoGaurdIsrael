"""Unit tests for SatelliteHotspotAgent.

All dependencies are mocked. These tests require no FIRMS MAP key, NASA access,
internet, Neon connection, or current clock time.
"""
from datetime import datetime
from unittest.mock import Mock

import pytest

from src.agents.collection.satellite_hotspot_agent import SatelliteHotspotAgent
from src.external.firms.exceptions import FIRMSAuthenticationError, FIRMSServiceUnavailableError
from src.external.firms.firms_client import FIRMSClient
from src.mappers.exceptions import SatelliteHotspotMappingError
from src.mappers.satellite_hotspot_mapper import SatelliteHotspotMapper
from src.models.satellite_hotspot import SatelliteHotspot
from src.repositories.exceptions import SatelliteHotspotRepositoryError
from src.repositories.satellite_hotspot_repository import SaveHotspotResult, SatelliteHotspotRepository

DETECTED_AT = datetime(2026, 9, 5, 14, 32, 0)


def make_raw_detection(latitude: str = "32.7", longitude: str = "35.0", **overrides) -> dict:
    raw = {
        "latitude": latitude,
        "longitude": longitude,
        "acq_date": "2026-09-05",
        "acq_time": "1432",
        "confidence": "n",
        "frp": "12.4",
        "bright_ti4": "341.2",
        "satellite": "N20",
        "instrument": "VIIRS",
        "daynight": "D",
    }
    raw.update(overrides)
    return raw


def make_hotspot(latitude: float = 32.7, longitude: float = 35.0, **overrides) -> SatelliteHotspot:
    defaults = dict(
        latitude=latitude,
        longitude=longitude,
        detected_at=DETECTED_AT,
        confidence="n",
        frp=12.4,
        brightness=341.2,
        satellite="N20",
        instrument="VIIRS",
        day_night="D",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


@pytest.fixture
def firms_client() -> Mock:
    return Mock(spec=FIRMSClient)


@pytest.fixture
def hotspot_mapper() -> Mock:
    return Mock(spec=SatelliteHotspotMapper)


@pytest.fixture
def hotspot_repository() -> Mock:
    return Mock(spec=SatelliteHotspotRepository)


@pytest.fixture
def agent(firms_client: Mock, hotspot_mapper: Mock, hotspot_repository: Mock) -> SatelliteHotspotAgent:
    return SatelliteHotspotAgent(
        firms_client=firms_client,
        hotspot_mapper=hotspot_mapper,
        hotspot_repository=hotspot_repository,
        west=34.0,
        south=29.4,
        east=35.9,
        north=33.4,
    )


def _wire_successful_processing(hotspot_mapper: Mock, hotspot_repository: Mock) -> None:
    hotspot_mapper.map_detection.side_effect = lambda raw: make_hotspot(
        latitude=float(raw["latitude"]),
        longitude=float(raw["longitude"]),
        satellite=raw.get("satellite", "N20"),
    )
    hotspot_repository.save_hotspot.side_effect = lambda hotspot: SaveHotspotResult(
        hotspot=hotspot,
        is_duplicate=False,
    )


def test_collect_happy_path_calls_firms_once_and_saves_three_hotspots(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    raw_detections = [
        make_raw_detection("32.1", "34.5"),
        make_raw_detection("32.2", "34.6"),
        make_raw_detection("32.3", "34.7"),
    ]
    firms_client.get_area_hotspots.return_value = raw_detections
    _wire_successful_processing(hotspot_mapper, hotspot_repository)

    result = agent.collect()

    firms_client.get_area_hotspots.assert_called_once()
    assert hotspot_mapper.map_detection.call_count == 3
    assert hotspot_repository.save_hotspot.call_count == 3
    assert result.success is True
    assert result.detections_received == 3
    assert result.hotspots_saved == 3
    assert result.duplicates_skipped == 0
    assert result.ignored_outside_area == 0
    assert result.detections_failed == 0


def test_collect_zero_detections_is_success_without_mapper_or_repository_calls(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    firms_client.get_area_hotspots.return_value = []

    result = agent.collect()

    firms_client.get_area_hotspots.assert_called_once()
    hotspot_mapper.map_detection.assert_not_called()
    hotspot_repository.save_hotspot.assert_not_called()
    assert result.success is True
    assert result.detections_received == 0
    assert result.hotspots_saved == 0
    assert result.duplicates_skipped == 0
    assert result.ignored_outside_area == 0
    assert result.detections_failed == 0


def test_collect_ignores_outside_area_detection_without_mapping_or_persisting_it(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    inside = make_raw_detection("32.7", "35.0")
    outside = make_raw_detection("32.7", "36.0")
    firms_client.get_area_hotspots.return_value = [inside, outside]
    _wire_successful_processing(hotspot_mapper, hotspot_repository)

    result = agent.collect()

    hotspot_mapper.map_detection.assert_called_once_with(inside)
    assert hotspot_repository.save_hotspot.call_count == 1
    assert result.success is True
    assert result.hotspots_saved == 1
    assert result.ignored_outside_area == 1
    assert result.detections_failed == 0


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        ("29.4", "34.0"),
        ("33.4", "35.9"),
    ],
)
def test_collect_includes_configured_bounding_edges(agent, firms_client, hotspot_mapper, hotspot_repository, latitude, longitude):
    raw_detection = make_raw_detection(latitude, longitude)
    firms_client.get_area_hotspots.return_value = [raw_detection]
    _wire_successful_processing(hotspot_mapper, hotspot_repository)

    result = agent.collect()

    hotspot_mapper.map_detection.assert_called_once_with(raw_detection)
    assert result.hotspots_saved == 1
    assert result.ignored_outside_area == 0


@pytest.mark.parametrize(
    "raw_detection",
    [
        make_raw_detection(latitude="abc"),
        {"longitude": "35.0", "acq_date": "2026-09-05", "acq_time": "1432"},
        make_raw_detection(longitude=""),
        {"latitude": "32.7", "acq_date": "2026-09-05", "acq_time": "1432"},
    ],
)
def test_collect_invalid_raw_coordinates_count_as_failed_not_ignored(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
    raw_detection,
):
    firms_client.get_area_hotspots.return_value = [raw_detection]

    result = agent.collect()

    hotspot_mapper.map_detection.assert_not_called()
    hotspot_repository.save_hotspot.assert_not_called()
    assert result.detections_failed == 1
    assert result.ignored_outside_area == 0


def test_collect_mapping_failure_is_isolated_and_later_detection_continues(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    first = make_raw_detection("32.1", "34.5")
    second = make_raw_detection("32.2", "34.6")
    second_hotspot = make_hotspot(latitude=32.2, longitude=34.6)
    firms_client.get_area_hotspots.return_value = [first, second]
    hotspot_mapper.map_detection.side_effect = [
        SatelliteHotspotMappingError("bad detection"),
        second_hotspot,
    ]
    hotspot_repository.save_hotspot.return_value = SaveHotspotResult(
        hotspot=second_hotspot,
        is_duplicate=False,
    )

    result = agent.collect()

    assert hotspot_mapper.map_detection.call_count == 2
    hotspot_repository.save_hotspot.assert_called_once_with(second_hotspot)
    assert result.success is True
    assert result.detections_failed == 1
    assert result.hotspots_saved == 1


def test_collect_missing_optional_fields_still_saves_valid_hotspot(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    raw_detection = {
        "latitude": "32.7",
        "longitude": "35.0",
        "acq_date": "2026-09-05",
        "acq_time": "1432",
    }
    hotspot = make_hotspot(
        confidence=None,
        frp=None,
        brightness=None,
        satellite=None,
        instrument=None,
        day_night=None,
    )
    firms_client.get_area_hotspots.return_value = [raw_detection]
    hotspot_mapper.map_detection.return_value = hotspot
    hotspot_repository.save_hotspot.return_value = SaveHotspotResult(hotspot=hotspot, is_duplicate=False)

    result = agent.collect()

    hotspot_repository.save_hotspot.assert_called_once_with(hotspot)
    assert result.hotspots_saved == 1
    assert result.detections_failed == 0


def test_collect_duplicate_increments_duplicates_not_saved_or_failed(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    raw_detection = make_raw_detection()
    hotspot = make_hotspot()
    firms_client.get_area_hotspots.return_value = [raw_detection]
    hotspot_mapper.map_detection.return_value = hotspot
    hotspot_repository.save_hotspot.return_value = SaveHotspotResult(hotspot=hotspot, is_duplicate=True)

    result = agent.collect()

    assert result.success is True
    assert result.hotspots_saved == 0
    assert result.duplicates_skipped == 1
    assert result.detections_failed == 0


def test_collect_repository_failure_is_isolated_and_later_detection_continues(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    first = make_raw_detection("32.1", "34.5")
    second = make_raw_detection("32.2", "34.6")
    first_hotspot = make_hotspot(latitude=32.1, longitude=34.5)
    second_hotspot = make_hotspot(latitude=32.2, longitude=34.6)
    firms_client.get_area_hotspots.return_value = [first, second]
    hotspot_mapper.map_detection.side_effect = [first_hotspot, second_hotspot]
    hotspot_repository.save_hotspot.side_effect = [
        SatelliteHotspotRepositoryError("DB write failed"),
        SaveHotspotResult(hotspot=second_hotspot, is_duplicate=False),
    ]

    result = agent.collect()

    assert hotspot_repository.save_hotspot.call_count == 2
    assert result.success is True
    assert result.detections_failed == 1
    assert result.hotspots_saved == 1


def test_collect_firms_unavailable_stops_before_mapper_or_repository_calls(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    firms_client.get_area_hotspots.side_effect = FIRMSServiceUnavailableError("FIRMS service returned HTTP 503.")

    result = agent.collect()

    hotspot_mapper.map_detection.assert_not_called()
    hotspot_repository.save_hotspot.assert_not_called()
    assert result.success is False
    assert result.detections_received == 0
    assert result.error_message is not None


def test_collect_firms_auth_failure_has_same_top_level_failure_semantics(
    agent,
    firms_client,
    hotspot_mapper,
    hotspot_repository,
):
    firms_client.get_area_hotspots.side_effect = FIRMSAuthenticationError("FIRMS authentication failed.")

    result = agent.collect()

    hotspot_mapper.map_detection.assert_not_called()
    hotspot_repository.save_hotspot.assert_not_called()
    assert result.success is False
    assert result.error_message == "FIRMS authentication failed."
