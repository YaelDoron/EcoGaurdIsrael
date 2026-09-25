"""Converts raw IMS API JSON into EcoGuard's internal weather models.

WeatherMapper is the only layer that understands both the raw IMS field
names (e.g. "TD", "stationId") and EcoGuard's internal field names (e.g.
"temperature", "external_station_id"). It is stateless and performs no I/O:
no database, no network, no IMSClient, no repository, no scheduler, no
environment variables.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.mappers.exceptions import (
    InvalidWeatherTimestampError,
    MissingRequiredWeatherFieldError,
    WeatherMappingError,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

logger = logging.getLogger(__name__)

# Centralized IMS channel name -> EcoGuard field name mapping. Keep all raw
# IMS channel strings ("TD", "RH", ...) confined to this dict rather than
# scattered across methods.
IMS_CHANNEL_MAPPING: dict[str, str] = {
    "TD": "temperature",
    "RH": "relative_humidity",
    "WS": "wind_speed",
    "WD": "wind_direction",
    "WSmax": "wind_gust",
    "Rain": "rainfall",
}

# IMS reports WS/WSmax in m/s (IMS API documentation, Appendix C); EcoGuard's
# canonical WeatherObservation wind unit is km/h, so both are converted here,
# at the ingestion boundary, and nowhere downstream.
IMS_MS_CHANNELS: frozenset[str] = frozenset({"WS", "WSmax"})
MS_TO_KMH = 3.6


class WeatherMapper:
    """Stateless mapper from raw IMS JSON to EcoGuard's internal weather models."""

    @staticmethod
    def map_station(raw_station: dict[str, Any]) -> WeatherStation:
        """Convert a raw IMS station dict into a WeatherStation.

        `stationId`, `name`, `location.latitude` and `location.longitude`
        are mandatory. `regionId` and `active` are optional and default to
        `None` when absent - values are never invented.
        """
        station_id = raw_station.get("stationId")
        name = raw_station.get("name")
        location = raw_station.get("location") or {}
        latitude = WeatherMapper._to_float(location.get("latitude"))
        longitude = WeatherMapper._to_float(location.get("longitude"))

        if station_id is None:
            raise MissingRequiredWeatherFieldError("Raw IMS station is missing required field 'stationId'.")
        if not name:
            raise MissingRequiredWeatherFieldError("Raw IMS station is missing required field 'name'.")
        if latitude is None:
            raise MissingRequiredWeatherFieldError(
                "Raw IMS station is missing or has an invalid 'location.latitude'."
            )
        if longitude is None:
            raise MissingRequiredWeatherFieldError(
                "Raw IMS station is missing or has an invalid 'location.longitude'."
            )

        try:
            return WeatherStation(
                external_station_id=station_id,
                name=name,
                latitude=latitude,
                longitude=longitude,
                region_id=raw_station.get("regionId"),
                active=raw_station.get("active"),
            )
        except ValueError as exc:
            raise WeatherMappingError(f"Failed to map IMS station: {exc}") from exc

    @staticmethod
    def map_stations(raw_stations: list[dict[str, Any]]) -> list[WeatherStation]:
        """Convert a list of raw IMS station dicts into a list of WeatherStation."""
        return [WeatherMapper.map_station(raw_station) for raw_station in raw_stations]

    @staticmethod
    def map_observation(raw_observation: dict[str, Any]) -> WeatherObservation:
        """Convert a raw IMS observation dict into a WeatherObservation.

        `stationId` and `datetime` are mandatory. Each of the six
        wildfire-relevant channels (TD, RH, WS, WD, WSmax, Rain) is optional:
        a missing or invalid channel results in `None` for that field rather
        than rejecting the whole observation. Unknown channels are ignored.
        """
        station_id = raw_observation.get("stationId")
        raw_timestamp = raw_observation.get("datetime")

        if station_id is None:
            raise MissingRequiredWeatherFieldError("Raw IMS observation is missing required field 'stationId'.")
        if not raw_timestamp:
            raise MissingRequiredWeatherFieldError("Raw IMS observation is missing required field 'datetime'.")

        timestamp = WeatherMapper._parse_timestamp(raw_timestamp)
        measurements = WeatherMapper._extract_channel_values(raw_observation.get("channels") or [])

        try:
            return WeatherObservation(
                station_external_id=station_id,
                timestamp=timestamp,
                **measurements,
            )
        except ValueError as exc:
            raise WeatherMappingError(f"Failed to map IMS observation: {exc}") from exc

    @staticmethod
    def _parse_timestamp(raw_timestamp: Any) -> datetime:
        """Parse the raw IMS datetime string into a Python datetime.

        TODO: IMS timezone behavior has not been verified against a live
        response yet (see Task 1). This preserves the timestamp exactly as
        represented - naive or with whatever offset IMS provides - without
        inventing a timezone correction. Revisit once real IMS responses are
        available.
        """
        if not isinstance(raw_timestamp, str):
            raise InvalidWeatherTimestampError(
                f"IMS observation timestamp must be a string, got {raw_timestamp!r}."
            )
        try:
            return datetime.fromisoformat(raw_timestamp)
        except ValueError as exc:
            raise InvalidWeatherTimestampError(
                f"IMS observation timestamp could not be parsed: {raw_timestamp!r}."
            ) from exc

    @staticmethod
    def _extract_channel_values(channels: list[dict[str, Any]]) -> dict[str, float | None]:
        """Extract the wildfire-relevant channel values from raw IMS channels.

        Rules (see Task 2 spec):
        - Unknown channel names are ignored.
        - `valid: false` takes precedence over any present `value` -> `None`.
        - A missing `valid` key is treated as valid when a non-null `value`
          is provided.
        - Numeric strings are converted to float; values that cannot be
          converted are treated as unusable (-> `None`) and logged, without
          failing the rest of the observation.
        - If the same relevant channel appears more than once, the last
          valid, numeric occurrence wins (channels are processed in order).
        - WS/WSmax are converted from IMS m/s to canonical km/h.
        """
        measurements: dict[str, float | None] = {field: None for field in IMS_CHANNEL_MAPPING.values()}

        for channel in channels:
            channel_name = channel.get("name")
            if channel_name not in IMS_CHANNEL_MAPPING:
                continue

            raw_value = channel.get("value")
            is_valid = channel.get("valid", raw_value is not None)
            if not is_valid or raw_value is None:
                continue

            numeric_value = WeatherMapper._to_float(raw_value)
            if numeric_value is None:
                logger.warning("Ignoring unparseable value for IMS channel %s", channel_name)
                continue
            if channel_name in IMS_MS_CHANNELS:
                numeric_value *= MS_TO_KMH

            measurements[IMS_CHANNEL_MAPPING[channel_name]] = numeric_value

        return measurements

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Convert a raw value to float, returning None if that is not possible."""
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None
