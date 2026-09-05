"""Converts raw NASA FIRMS detections into EcoGuard satellite hotspot models.

SatelliteHotspotMapper understands raw FIRMS field names and produces the
internal SatelliteHotspot dataclass. It performs no I/O and does not confirm
wildfires, filter geography, deduplicate records, or know about persistence.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from src.mappers.exceptions import (
    InvalidSatelliteDetectionTimeError,
    MissingRequiredSatelliteFieldError,
    SatelliteHotspotMappingError,
)
from src.models.satellite_hotspot import SatelliteHotspot

logger = logging.getLogger(__name__)

_ACQ_TIME_PATTERN = re.compile(r"^\d{4}$")


class SatelliteHotspotMapper:
    """Stateless mapper from raw FIRMS CSV rows to SatelliteHotspot."""

    @staticmethod
    def map_detection(raw_detection: dict[str, Any]) -> SatelliteHotspot:
        """Convert one raw FIRMS detection dict into a SatelliteHotspot."""
        latitude = SatelliteHotspotMapper._required_float(raw_detection, "latitude")
        longitude = SatelliteHotspotMapper._required_float(raw_detection, "longitude")
        detected_at = SatelliteHotspotMapper._parse_detected_at(
            raw_detection.get("acq_date"),
            raw_detection.get("acq_time"),
        )

        try:
            return SatelliteHotspot(
                latitude=latitude,
                longitude=longitude,
                detected_at=detected_at,
                confidence=SatelliteHotspotMapper._optional_string(raw_detection.get("confidence")),
                frp=SatelliteHotspotMapper._optional_float(raw_detection.get("frp"), "frp"),
                brightness=SatelliteHotspotMapper._optional_float(
                    raw_detection.get("bright_ti4"), "bright_ti4"
                ),
                satellite=SatelliteHotspotMapper._optional_string(raw_detection.get("satellite")),
                instrument=SatelliteHotspotMapper._optional_string(raw_detection.get("instrument")),
                day_night=SatelliteHotspotMapper._optional_string(raw_detection.get("daynight")),
            )
        except ValueError as exc:
            raise SatelliteHotspotMappingError(f"Failed to map FIRMS detection: {exc}") from exc

    @staticmethod
    def map_detections(raw_detections: list[dict[str, Any]]) -> list[SatelliteHotspot]:
        """Convert raw FIRMS detections into hotspots, preserving input order."""
        return [SatelliteHotspotMapper.map_detection(raw_detection) for raw_detection in raw_detections]

    @staticmethod
    def _required_float(raw_detection: dict[str, Any], field_name: str) -> float:
        value = raw_detection.get(field_name)
        numeric_value = SatelliteHotspotMapper._to_float(value)
        if numeric_value is None:
            raise MissingRequiredSatelliteFieldError(
                f"Raw FIRMS detection is missing or has an invalid '{field_name}'."
            )
        return numeric_value

    @staticmethod
    def _optional_float(value: Any, field_name: str) -> float | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None

        numeric_value = SatelliteHotspotMapper._to_float(value)
        if numeric_value is None:
            logger.warning("Ignoring unparseable FIRMS value for %s", field_name)
        return numeric_value

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)

        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _parse_detected_at(raw_date: Any, raw_time: Any) -> datetime:
        """Parse FIRMS acquisition date and time without timezone conversion.

        TODO: Confirm FIRMS acquisition timezone semantics before introducing
        any normalization. This currently preserves the provided date/time as a
        naive datetime.
        """
        if not raw_date:
            raise MissingRequiredSatelliteFieldError("Raw FIRMS detection is missing required field 'acq_date'.")
        if not raw_time:
            raise MissingRequiredSatelliteFieldError("Raw FIRMS detection is missing required field 'acq_time'.")
        if not isinstance(raw_date, str) or not isinstance(raw_time, str):
            raise InvalidSatelliteDetectionTimeError("FIRMS acquisition date/time must be strings.")
        if not _ACQ_TIME_PATTERN.fullmatch(raw_time):
            raise InvalidSatelliteDetectionTimeError(
                f"FIRMS acquisition time could not be parsed: {raw_time!r}."
            )

        hour = int(raw_time[:2])
        minute = int(raw_time[2:])
        if hour > 23 or minute > 59:
            raise InvalidSatelliteDetectionTimeError(
                f"FIRMS acquisition time could not be parsed: {raw_time!r}."
            )

        try:
            parsed_date = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError as exc:
            raise InvalidSatelliteDetectionTimeError(
                f"FIRMS acquisition date could not be parsed: {raw_date!r}."
            ) from exc

        return parsed_date.replace(hour=hour, minute=minute)

    @staticmethod
    def _to_float(value: Any) -> float | None:
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
