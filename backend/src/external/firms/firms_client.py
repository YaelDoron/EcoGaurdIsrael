"""HTTP client for the NASA FIRMS Area CSV API.

This module is responsible ONLY for communicating with NASA FIRMS over HTTP and
returning raw CSV detections as dictionaries. It has no knowledge of databases,
repositories, wildfire confirmation, scheduling, or EcoGuard domain models.
"""
from __future__ import annotations

import csv
import io
import logging

import requests

from src.config.settings import settings
from src.external.firms.exceptions import (
    FIRMSAuthenticationError,
    FIRMSClientError,
    FIRMSConfigurationError,
    FIRMSInvalidResponseError,
    FIRMSServiceUnavailableError,
)

logger = logging.getLogger(__name__)

_AUTHENTICATION_ERROR_STATUS_CODES = (401, 403)
_SERVICE_UNAVAILABLE_STATUS_CODES = (500, 502, 503, 504)


class FIRMSClient:
    """Thin HTTP client for the NASA FIRMS Global Area API."""

    def __init__(
        self,
        base_url: str = settings.FIRMS_BASE_URL,
        map_key: str = settings.FIRMS_MAP_KEY,
        source: str = settings.FIRMS_SOURCE,
        day_range: int = settings.FIRMS_DAY_RANGE,
        timeout: int = settings.FIRMS_REQUEST_TIMEOUT,
        west: float = settings.FIRMS_WEST,
        south: float = settings.FIRMS_SOUTH,
        east: float = settings.FIRMS_EAST,
        north: float = settings.FIRMS_NORTH,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.map_key = map_key
        self.source = source
        self.day_range = day_range
        self.timeout = timeout
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def get_area_hotspots(self) -> list[dict[str, str]]:
        """Retrieve raw FIRMS thermal anomaly detections for the configured area."""
        self._validate_configuration()
        url = self._build_area_url()

        logger.info("Requesting FIRMS area hotspot data using source %s", self.source)

        try:
            response = requests.get(url, timeout=self.timeout)
        except requests.exceptions.Timeout:
            logger.error("FIRMS request timed out")
            raise FIRMSServiceUnavailableError("FIRMS request timed out.") from None
        except requests.exceptions.ConnectionError:
            logger.error("FIRMS connection failed")
            raise FIRMSServiceUnavailableError("Could not connect to FIRMS service.") from None
        except requests.exceptions.RequestException:
            logger.error("FIRMS request failed")
            raise FIRMSClientError("FIRMS request failed.") from None

        self._raise_for_status(response)
        detections = self._parse_csv_response(response.text)

        if not detections:
            logger.info("FIRMS returned no detections")
        else:
            logger.info("FIRMS returned %s detections", len(detections))

        return detections

    def _build_area_coordinates(self) -> str:
        """Build the west,south,east,north FIRMS area coordinate string."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def _build_area_url(self) -> str:
        """Build the FIRMS Area CSV URL. Do not log or expose this URL."""
        coordinates = self._build_area_coordinates()
        return f"{self.base_url}/area/csv/{self.map_key}/{self.source}/{coordinates}/{self.day_range}"

    def _validate_configuration(self) -> None:
        """Validate FIRMS request configuration before making any HTTP request."""
        if not self.map_key:
            raise FIRMSConfigurationError("FIRMS MAP key is not configured.")
        if not isinstance(self.source, str) or not self.source:
            raise FIRMSConfigurationError("FIRMS source must be a non-empty string.")
        if isinstance(self.day_range, bool) or not isinstance(self.day_range, int):
            raise FIRMSConfigurationError("FIRMS day range must be an integer.")
        if self.day_range < 1 or self.day_range > 5:
            raise FIRMSConfigurationError("FIRMS day range must be between 1 and 5.")
        if not self._is_valid_longitude(self.west) or not self._is_valid_longitude(self.east):
            raise FIRMSConfigurationError("FIRMS west/east coordinates must be valid longitudes.")
        if not self._is_valid_latitude(self.south) or not self._is_valid_latitude(self.north):
            raise FIRMSConfigurationError("FIRMS south/north coordinates must be valid latitudes.")
        if self.west >= self.east:
            raise FIRMSConfigurationError("FIRMS west coordinate must be less than east coordinate.")
        if self.south >= self.north:
            raise FIRMSConfigurationError("FIRMS south coordinate must be less than north coordinate.")

    @staticmethod
    def _is_valid_longitude(value: float) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and -180 <= value <= 180

    @staticmethod
    def _is_valid_latitude(value: float) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and -90 <= value <= 90

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        """Translate a FIRMS HTTP error status code into a FIRMS-specific exception."""
        status_code = response.status_code

        if status_code in _AUTHENTICATION_ERROR_STATUS_CODES:
            logger.error("FIRMS authentication failed with HTTP %s", status_code)
            raise FIRMSAuthenticationError("FIRMS authentication failed.")
        if status_code in _SERVICE_UNAVAILABLE_STATUS_CODES:
            logger.error("FIRMS service unavailable with HTTP %s", status_code)
            raise FIRMSServiceUnavailableError(f"FIRMS service returned HTTP {status_code}.")
        if not response.ok:
            logger.error("FIRMS request failed with unexpected HTTP %s", status_code)
            raise FIRMSClientError(f"FIRMS request failed with HTTP {status_code}.")

    @staticmethod
    def _parse_csv_response(response_text: str) -> list[dict[str, str]]:
        """Parse FIRMS CSV text into raw dictionaries while preserving columns."""
        if not response_text or response_text.lstrip().startswith("<"):
            raise FIRMSInvalidResponseError("FIRMS returned an invalid CSV response.")

        try:
            reader = csv.DictReader(io.StringIO(response_text))
            fieldnames = reader.fieldnames
            if not fieldnames or "latitude" not in fieldnames or "longitude" not in fieldnames:
                raise FIRMSInvalidResponseError("FIRMS response is missing detection coordinates.")

            rows = list(reader)
        except csv.Error as exc:
            raise FIRMSInvalidResponseError("FIRMS returned a malformed CSV response.") from exc

        if any(None in row for row in rows):
            raise FIRMSInvalidResponseError("FIRMS returned a malformed CSV response.")

        return rows
