"""Orchestrates one satellite hotspot collection cycle.

SatelliteHotspotAgent coordinates FIRMSClient, SatelliteHotspotMapper, and
SatelliteHotspotRepository. It contains no HTTP implementation, CSV parsing,
SQLAlchemy session logic, duplicate-key generation, scheduling, or wildfire
confirmation logic.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.collection.satellite_hotspot_collection_result import (
    SatelliteHotspotCollectionResult,
)
from src.config.settings import settings
from src.external.firms.exceptions import FIRMSClientError
from src.external.firms.firms_client import FIRMSClient
from src.mappers.exceptions import SatelliteHotspotMappingError
from src.mappers.satellite_hotspot_mapper import SatelliteHotspotMapper
from src.repositories.exceptions import SatelliteHotspotRepositoryError
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository

logger = logging.getLogger(__name__)


class SatelliteHotspotAgent:
    """Coordinates one FIRMS thermal-anomaly collection cycle."""

    def __init__(
        self,
        firms_client: FIRMSClient,
        hotspot_mapper: SatelliteHotspotMapper,
        hotspot_repository: SatelliteHotspotRepository,
        west: float = settings.FIRMS_WEST,
        south: float = settings.FIRMS_SOUTH,
        east: float = settings.FIRMS_EAST,
        north: float = settings.FIRMS_NORTH,
    ) -> None:
        self._firms_client = firms_client
        self._hotspot_mapper = hotspot_mapper
        self._hotspot_repository = hotspot_repository
        self._west = west
        self._south = south
        self._east = east
        self._north = north

    def collect(self) -> SatelliteHotspotCollectionResult:
        """Run one satellite hotspot collection cycle.

        Fetches raw FIRMS detections once, defensively filters valid
        coordinates against the configured bounding box, maps in-bounds rows,
        and persists mapped hotspots. Individual detection failures are
        isolated; FIRMS retrieval failure stops before any repository write.
        """
        logger.info("Starting satellite hotspot collection")

        try:
            raw_detections = self._firms_client.get_area_hotspots()
        except FIRMSClientError as exc:
            logger.error("Failed to retrieve FIRMS area hotspots: %s", exc)
            return SatelliteHotspotCollectionResult(success=False, error_message=str(exc))

        if raw_detections:
            logger.info("FIRMS returned %d detections", len(raw_detections))
        else:
            logger.info("FIRMS returned no detections")

        result = SatelliteHotspotCollectionResult(
            success=True,
            detections_received=len(raw_detections),
        )

        for raw_detection in raw_detections:
            self._process_detection(raw_detection, result)

        logger.info(
            "Satellite hotspot collection completed: %d received, %d saved, "
            "%d duplicates skipped, %d ignored outside area, %d failed",
            result.detections_received,
            result.hotspots_saved,
            result.duplicates_skipped,
            result.ignored_outside_area,
            result.detections_failed,
        )
        return result

    def _process_detection(
        self,
        raw_detection: dict[str, Any],
        result: SatelliteHotspotCollectionResult,
    ) -> None:
        coordinates = self._extract_coordinates(raw_detection)
        if coordinates is None:
            logger.error("Failed to read FIRMS detection coordinates")
            result.detections_failed += 1
            return

        latitude, longitude = coordinates
        if not self._is_inside_configured_area(latitude, longitude):
            logger.info("Ignoring detection outside configured area")
            result.ignored_outside_area += 1
            return

        try:
            hotspot = self._hotspot_mapper.map_detection(raw_detection)
        except SatelliteHotspotMappingError as exc:
            logger.error("Failed to map FIRMS detection: %s", exc)
            result.detections_failed += 1
            return

        try:
            save_result = self._hotspot_repository.save_hotspot(hotspot)
        except SatelliteHotspotRepositoryError as exc:
            logger.error("Failed to save satellite hotspot: %s", exc)
            result.detections_failed += 1
            return

        if save_result.is_duplicate:
            logger.info("Skipped duplicate satellite hotspot")
            result.duplicates_skipped += 1
        else:
            logger.info("Stored satellite hotspot")
            result.hotspots_saved += 1

    def _extract_coordinates(self, raw_detection: dict[str, Any]) -> tuple[float, float] | None:
        latitude = self._to_float(raw_detection.get("latitude"))
        longitude = self._to_float(raw_detection.get("longitude"))
        if latitude is None or longitude is None:
            return None
        return latitude, longitude

    def _is_inside_configured_area(self, latitude: float, longitude: float) -> bool:
        return self._west <= longitude <= self._east and self._south <= latitude <= self._north

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return None
        return None
