"""Unit tests for IMSClient.

All HTTP calls are mocked - no real network access is performed and no real
IMS API token is required to run these tests.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from src.external.ims.exceptions import (
    IMSAuthenticationError,
    IMSClientError,
    IMSConfigurationError,
    IMSInvalidResponseError,
    IMSServiceUnavailableError,
    IMSStationNotFoundError,
)
from src.external.ims.ims_client import IMSClient

FIXTURES_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "ims_sample_response.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))

BASE_URL = "https://api.ims.gov.il/v1/envista"
FAKE_TOKEN = "test-secret-token-123"


class FakeResponse:
    """Minimal stand-in for `requests.Response`, used to mock IMS HTTP replies."""

    def __init__(self, status_code: int, json_data=None, invalid_json: bool = False):
        self.status_code = status_code
        self.ok = status_code < 400
        self._json_data = json_data
        self._invalid_json = invalid_json

    def json(self):
        if self._invalid_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._json_data


def make_client(api_token: str = FAKE_TOKEN) -> IMSClient:
    return IMSClient(base_url=BASE_URL, api_token=api_token, timeout=10)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_request_without_api_token_raises_configuration_error():
    client = make_client(api_token="")

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        with pytest.raises(IMSConfigurationError, match="IMS API token is not configured."):
            client.get_stations()

    mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# get_stations
# ---------------------------------------------------------------------------


def test_get_stations_requests_correct_url():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["stations_list"])
        client.get_stations()

    called_url = mock_get.call_args.args[0]
    assert called_url == f"{BASE_URL}/stations"


def test_get_stations_sends_correct_authorization_header():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["stations_list"])
        client.get_stations()

    headers = mock_get.call_args.kwargs["headers"]
    assert headers["Authorization"] == f"ApiToken {FAKE_TOKEN}"
    assert headers["Accept"] == "application/json"


def test_get_stations_returns_parsed_json():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["stations_list"])
        result = client.get_stations()

    assert result == FIXTURES["stations_list"]
    # Raw IMS channel names must be preserved, not renamed.
    assert result[0]["stationId"] == 17


def test_get_stations_applies_configured_timeout():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["stations_list"])
        client.get_stations()

    assert mock_get.call_args.kwargs["timeout"] == 10


# ---------------------------------------------------------------------------
# get_station
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("station_id", [1, 2, 85])
def test_get_station_valid_id_requests_correct_url(station_id):
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["station"])
        client.get_station(station_id)

    called_url = mock_get.call_args.args[0]
    assert called_url == f"{BASE_URL}/stations/{station_id}"


@pytest.mark.parametrize("station_id", [None, 0, -1, "abc"])
def test_get_station_invalid_id_raises_without_http_request(station_id):
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        with pytest.raises(IMSClientError):
            client.get_station(station_id)

    mock_get.assert_not_called()


def test_get_station_not_found_raises_station_not_found_error():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(404)
        with pytest.raises(IMSStationNotFoundError):
            client.get_station(999)


def test_get_station_returns_parsed_json():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["station"])
        result = client.get_station(17)

    assert result == FIXTURES["station"]


# ---------------------------------------------------------------------------
# get_station_data
# ---------------------------------------------------------------------------


def test_get_station_data_returns_raw_channels_with_original_names():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["observation"])
        result = client.get_station_data(17)

    channel_names = {channel["name"] for channel in result["channels"]}
    assert channel_names == {"TD", "RH", "WS", "WD", "WSmax", "Rain"}


def test_get_station_data_handles_missing_or_invalid_measurement():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, FIXTURES["observation_missing_measurement"])
        result = client.get_station_data(17)

    rh_channel = next(c for c in result["channels"] if c["name"] == "RH")
    assert rh_channel["valid"] is False
    assert rh_channel["value"] is None


# ---------------------------------------------------------------------------
# Authentication failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_error_status_codes_raise_authentication_error(status_code):
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code)
        with pytest.raises(IMSAuthenticationError):
            client.get_stations()


# ---------------------------------------------------------------------------
# Service failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_server_error_status_codes_raise_service_unavailable_error(status_code):
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(status_code)
        with pytest.raises(IMSServiceUnavailableError):
            client.get_stations()


def test_request_timeout_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout()
        with pytest.raises(IMSServiceUnavailableError):
            client.get_stations()


def test_connection_error_raises_service_unavailable_error():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError()
        with pytest.raises(IMSServiceUnavailableError):
            client.get_stations()


# ---------------------------------------------------------------------------
# Invalid responses
# ---------------------------------------------------------------------------


def test_invalid_json_response_raises_invalid_response_error():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(200, invalid_json=True)
        with pytest.raises(IMSInvalidResponseError):
            client.get_stations()


# ---------------------------------------------------------------------------
# Security: the API token must never leak into exceptions or logs
# ---------------------------------------------------------------------------


def test_exception_messages_never_contain_the_api_token():
    client = make_client()

    with patch("src.external.ims.ims_client.requests.get") as mock_get:
        mock_get.return_value = FakeResponse(500)
        with pytest.raises(IMSServiceUnavailableError) as exc_info:
            client.get_stations()

    assert FAKE_TOKEN not in str(exc_info.value)


def test_logs_never_contain_the_api_token(caplog):
    client = make_client()

    with caplog.at_level("DEBUG"):
        with patch("src.external.ims.ims_client.requests.get") as mock_get:
            mock_get.return_value = FakeResponse(200, FIXTURES["stations_list"])
            client.get_stations()

    for record in caplog.records:
        assert FAKE_TOKEN not in record.getMessage()
        assert "Authorization" not in record.getMessage()
