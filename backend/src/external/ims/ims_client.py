"""HTTP client for the Israeli Meteorological Service (IMS) Observation Data API.

This module is responsible ONLY for communicating with the IMS API over HTTP and
returning the raw JSON payloads it responds with. It has no knowledge of the
database, repositories, wildfire risk logic, scheduling, or any other EcoGuard
domain concept — that all belongs to the WeatherAgent layer built on top of this
client.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from src.config.settings import settings
from src.external.ims.exceptions import (
    IMSAuthenticationError,
    IMSClientError,
    IMSConfigurationError,
    IMSInvalidResponseError,
    IMSServiceUnavailableError,
    IMSStationNotFoundError,
)

logger = logging.getLogger(__name__)

_AUTHENTICATION_ERROR_STATUS_CODES = (401, 403)
_NOT_FOUND_STATUS_CODE = 404
_SERVICE_UNAVAILABLE_STATUS_CODES = (500, 502, 503, 504)


class IMSClient:
    """Thin HTTP client for the IMS Observation Data API.

    Returns raw IMS JSON data (e.g. channel names such as "TD", "RH", "WS" are
    kept as-is). Mapping to EcoGuard's internal representation happens in a
    later layer (WeatherAgent), not here.
    """

    def __init__(
        self,
        base_url: str = settings.IMS_BASE_URL,
        api_token: str = settings.IMS_API_TOKEN,
        timeout: int = settings.IMS_REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout

    def get_stations(self) -> list[dict]:
        """Retrieve the raw list of IMS weather stations (GET /stations).

        Returns the parsed JSON as received from IMS. Not every station
        supports every measurement, and this method makes no assumption about
        that — it simply returns what IMS reports.
        """
        logger.info("Requesting IMS stations")
        return self._send_get_request("stations")

    def get_station(self, station_id: int) -> dict:
        """Retrieve raw information for a single IMS station (GET /stations/{id}).

        Validates `station_id` before sending any request; invalid values
        raise an IMSClientError without contacting IMS.
        """
        self._validate_station_id(station_id)
        logger.info("Requesting IMS station %s", station_id)
        return self._send_get_request(f"stations/{station_id}")

    def get_station_data(self, station_id: int) -> dict:
        """Retrieve the latest raw observation for a single IMS station
        (GET /stations/{id}/data/latest).

        Verified against the real IMS API: the response is wrapped as
        `{"stationId": ..., "data": [{"datetime": ..., "channels": [...]}]}`.
        (The bare `/stations/{id}/data` path returns 404.)
        """
        self._validate_station_id(station_id)
        logger.info("Requesting IMS station data for station %s", station_id)
        return self._send_get_request(f"stations/{station_id}/data/latest")

    def _build_headers(self) -> dict[str, str]:
        """Build the authorization headers required by the IMS API.

        Centralized so authorization-header construction is not duplicated
        across methods. Never logs the token.
        """
        return {
            "Authorization": f"ApiToken {self.api_token}",
            "Accept": "application/json",
        }

    def _build_url(self, endpoint: str) -> str:
        """Safely join the configured base URL with a request endpoint path."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _send_get_request(self, endpoint: str) -> Any:
        """Send a GET request to the IMS API and return the parsed JSON body.

        Centralizes token validation, URL/header construction, the HTTP call,
        status-code handling, and JSON parsing, converting any HTTP/network
        problem into an IMS-specific exception.
        """
        if not self.api_token:
            raise IMSConfigurationError("IMS API token is not configured.")

        url = self._build_url(endpoint)
        headers = self._build_headers()

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
        except requests.exceptions.Timeout as exc:
            logger.error("IMS request timed out: %s", endpoint)
            raise IMSServiceUnavailableError("IMS request timed out.") from exc
        except requests.exceptions.ConnectionError as exc:
            logger.error("IMS connection failed: %s", endpoint)
            raise IMSServiceUnavailableError("Could not connect to IMS service.") from exc
        except requests.exceptions.RequestException as exc:
            logger.error("IMS request failed: %s", endpoint)
            raise IMSClientError("IMS request failed.") from exc

        self._raise_for_status(response)

        try:
            return response.json()
        except ValueError as exc:
            logger.error("IMS returned an invalid (non-JSON) response: %s", endpoint)
            raise IMSInvalidResponseError("IMS returned an invalid JSON response.") from exc

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Translate an IMS HTTP error status code into an IMS-specific exception."""
        status_code = response.status_code

        if status_code in _AUTHENTICATION_ERROR_STATUS_CODES:
            logger.error("IMS authentication failed with HTTP %s", status_code)
            raise IMSAuthenticationError("IMS authentication failed.")
        if status_code == _NOT_FOUND_STATUS_CODE:
            logger.error("IMS resource not found (HTTP 404)")
            raise IMSStationNotFoundError("Requested IMS resource was not found.")
        if status_code in _SERVICE_UNAVAILABLE_STATUS_CODES:
            logger.error("IMS service returned HTTP %s", status_code)
            raise IMSServiceUnavailableError(f"IMS service returned HTTP {status_code}.")
        if not response.ok:
            logger.error("IMS request failed with unexpected HTTP %s", status_code)
            raise IMSClientError(f"IMS request failed with HTTP {status_code}.")

    @staticmethod
    def _validate_station_id(station_id: int) -> None:
        """Validate a station id before it is used to build a request URL."""
        if isinstance(station_id, bool) or not isinstance(station_id, int) or station_id <= 0:
            raise IMSClientError(f"Invalid station_id: {station_id!r}. Must be a positive integer.")
