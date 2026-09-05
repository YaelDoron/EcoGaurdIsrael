"""Unit tests for FIRMSClient.

All HTTP calls are mocked. No real NASA access, MAP key, or database connection
is required to run these tests.
"""
from unittest.mock import patch

import pytest
import requests

from src.external.firms.exceptions import (
    FIRMSAuthenticationError,
    FIRMSClientError,
    FIRMSConfigurationError,
    FIRMSInvalidResponseError,
    FIRMSServiceUnavailableError,
)
from src.external.firms.firms_client import FIRMSClient

BASE_URL = "https://firms.example.test/api"
FAKE_MAP_KEY = "TEST_FIRMS_MAP_KEY_123"
SOURCE = "VIIRS_NOAA20_NRT"

ONE_DETECTION_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "32.7,35.0,330.1,0.5,0.6,2026-09-05,1012,N20,VIIRS,n,2.0,290.0,3.4,D\n"
)
MULTIPLE_DETECTIONS_CSV = (
    "latitude,longitude,acq_date,acq_time\n"
    "32.7,35.0,2026-09-05,1012\n"
    "31.8,34.8,2026-09-05,1018\n"
)
HEADER_ONLY_CSV = "latitude,longitude,acq_date,acq_time\n"
UNKNOWN_COLUMNS_CSV = "latitude,longitude,unexpected_column\n32.7,35.0,kept\n"
MINIMAL_CSV = "latitude,longitude\n32.7,35.0\n"


class FakeResponse:
    """Minimal stand-in for `requests.Response`, used to mock FIRMS HTTP replies."""

    def __init__(self, status_code: int, text: str = ONE_DETECTION_CSV):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text


def make_client(**overrides) -> FIRMSClient:
    config = {
        "base_url": BASE_URL,
        "map_key": FAKE_MAP_KEY,
        "source": SOURCE,
        "day_range": 1,
        "timeout": 10,
        "west": 34.0,
        "south": 29.4,
        "east": 35.9,
        "north": 33.4,
    }
    config.update(overrides)
    return FIRMSClient(**config)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_missing_map_key_raises_configuration_error():
    client = make_client(map_key="")

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_empty_source_raises_configuration_error():
    client = make_client(source="")

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_day_range_zero_fails():
    client = make_client(day_range=0)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_day_range_six_fails():
    client = make_client(day_range=6)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_non_integer_day_range_fails():
    client = make_client(day_range="1")

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_west_greater_than_or_equal_to_east_fails():
    client = make_client(west=35.9, east=35.9)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_south_greater_than_or_equal_to_north_fails():
    client = make_client(south=33.4, north=33.4)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_invalid_latitude_fails():
    client = make_client(south=-91.0)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


def test_invalid_longitude_fails():
    client = make_client(west=-181.0)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        with pytest.raises(FIRMSConfigurationError):
            client.get_area_hotspots()

    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def test_correct_area_endpoint_is_constructed():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200)
        client.get_area_hotspots()

    called_url = mock_get.call_args.args[0]
    assert called_url == f"{BASE_URL}/area/csv/{FAKE_MAP_KEY}/{SOURCE}/34.0,29.4,35.9,33.4/1"


def test_correct_source_is_used():
    client = make_client(source="MODIS_NRT")

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200)
        client.get_area_hotspots()

    assert f"/{client.source}/" in mock_get.call_args.args[0]


def test_coordinate_order_is_west_south_east_north():
    client = make_client(west=1.0, south=2.0, east=3.0, north=4.0)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200)
        client.get_area_hotspots()

    assert "/1.0,2.0,3.0,4.0/" in mock_get.call_args.args[0]


def test_correct_day_range_is_used():
    client = make_client(day_range=5)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200)
        client.get_area_hotspots()

    assert mock_get.call_args.args[0].endswith("/5")


def test_configured_timeout_is_passed_to_requests():
    client = make_client(timeout=7)

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200)
        client.get_area_hotspots()

    assert mock_get.call_args.kwargs["timeout"] == 7


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_valid_csv_with_one_detection_returns_one_dict():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, ONE_DETECTION_CSV)
        result = client.get_area_hotspots()

    assert result == [
        {
            "latitude": "32.7",
            "longitude": "35.0",
            "bright_ti4": "330.1",
            "scan": "0.5",
            "track": "0.6",
            "acq_date": "2026-09-05",
            "acq_time": "1012",
            "satellite": "N20",
            "instrument": "VIIRS",
            "confidence": "n",
            "version": "2.0",
            "bright_ti5": "290.0",
            "frp": "3.4",
            "daynight": "D",
        }
    ]


def test_valid_csv_with_multiple_detections_returns_all_rows():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, MULTIPLE_DETECTIONS_CSV)
        result = client.get_area_hotspots()

    assert len(result) == 2
    assert result[0]["latitude"] == "32.7"
    assert result[1]["longitude"] == "34.8"


def test_valid_csv_header_with_no_detections_returns_empty_list():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, HEADER_ONLY_CSV)
        result = client.get_area_hotspots()

    assert result == []


def test_additional_unknown_firms_columns_are_preserved():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, UNKNOWN_COLUMNS_CSV)
        result = client.get_area_hotspots()

    assert result[0]["unexpected_column"] == "kept"


def test_missing_optional_fields_do_not_make_client_fail():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, MINIMAL_CSV)
        result = client.get_area_hotspots()

    assert result == [{"latitude": "32.7", "longitude": "35.0"}]


def test_html_response_raises_invalid_response_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, "<html>not csv</html>")
        with pytest.raises(FIRMSInvalidResponseError):
            client.get_area_hotspots()


def test_malformed_unrecognizable_csv_raises_invalid_response_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, "not,a,firms,response\n1,2,3\n")
        with pytest.raises(FIRMSInvalidResponseError):
            client.get_area_hotspots()


def test_malformed_csv_row_raises_invalid_response_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, "latitude,longitude\n32.7,35.0,extra\n")
        with pytest.raises(FIRMSInvalidResponseError):
            client.get_area_hotspots()


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_error_status_codes_raise_authentication_error(status_code):
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code)
        with pytest.raises(FIRMSAuthenticationError):
            client.get_area_hotspots()


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_server_error_status_codes_raise_service_unavailable_error(status_code):
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code)
        with pytest.raises(FIRMSServiceUnavailableError):
            client.get_area_hotspots()


def test_timeout_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout()
        with pytest.raises(FIRMSServiceUnavailableError):
            client.get_area_hotspots()


def test_connection_error_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(FIRMSServiceUnavailableError):
            client.get_area_hotspots()


def test_other_request_exception_raises_client_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.RequestException()
        with pytest.raises(FIRMSClientError):
            client.get_area_hotspots()


def test_unexpected_http_failure_raises_client_error():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(418)
        with pytest.raises(FIRMSClientError):
            client.get_area_hotspots()


# ---------------------------------------------------------------------------
# Security: the MAP key must never leak into exceptions or logs
# ---------------------------------------------------------------------------


def test_exception_messages_never_contain_the_map_key():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(500)
        with pytest.raises(FIRMSServiceUnavailableError) as exc_info:
            client.get_area_hotspots()

    assert FAKE_MAP_KEY not in str(exc_info.value)


def test_request_exception_chains_never_contain_the_map_key():
    client = make_client()

    with patch("src.external.firms.firms_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError(
            f"failed GET {BASE_URL}/area/csv/{FAKE_MAP_KEY}/{SOURCE}/34.0,29.4,35.9,33.4/1"
        )
        with pytest.raises(FIRMSServiceUnavailableError) as exc_info:
            client.get_area_hotspots()

    assert FAKE_MAP_KEY not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_logs_never_contain_the_map_key(caplog):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.firms.firms_client.requests.get") as mock_get:
            mock_get.return_value = FakeResponse(200)
            client.get_area_hotspots()

    for record in caplog.records:
        assert FAKE_MAP_KEY not in record.getMessage()
        assert "/area/csv/" not in record.getMessage()
