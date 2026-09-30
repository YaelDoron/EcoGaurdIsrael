"""Orchestrates one weather-data collection cycle.

WeatherAgent coordinates IMSClient, WeatherMapper and WeatherRepository to
run a single end-to-end collection pass: fetch stations from IMS, persist
them, fetch each station's current observation data, and persist it.

It performs orchestration ONLY. It contains no HTTP logic (that is
IMSClient's job), no SQL/SQLAlchemy logic (that is WeatherRepository's job),
no wildfire-risk logic, and no scheduling - one call to `collect()` is one
collection run.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.collection.weather_collection_result import WeatherCollectionResult
from src.external.ims.exceptions import IMSClientError
from src.external.ims.ims_client import IMSClient
from src.mappers.exceptions import WeatherMappingError
from src.mappers.weather_mapper import WeatherMapper
from src.repositories.exceptions import WeatherRepositoryError
from src.repositories.weather_repository import WeatherRepository

logger = logging.getLogger(__name__)


class WeatherAgent:
    """Coordinates a single weather-data collection cycle across IMS, the mapper, and the repository."""

    def __init__(
        self,
        ims_client: IMSClient,
        weather_mapper: WeatherMapper,
        weather_repository: WeatherRepository,
    ) -> None:
        self._ims_client = ims_client
        self._weather_mapper = weather_mapper
        self._weather_repository = weather_repository

    def collect(self) -> WeatherCollectionResult:
        """Run one complete weather-data collection cycle.

        Fetches the IMS station list, then processes each station
        independently: map -> persist -> fetch its observation -> map ->
        persist. A single station's failure (mapping, persistence, or
        observation retrieval/mapping/persistence) is isolated so later
        stations are still processed. The whole cycle only fails
        (`success=False`) if the station list itself could not be
        retrieved from IMS - in that case no repository write is attempted
        and previously stored data is left untouched.
        """
        logger.info("Starting weather collection")

        try:
            raw_stations = self._ims_client.get_stations()
        except IMSClientError as exc:
            logger.error("Failed to retrieve IMS station list: %s", exc)
            return WeatherCollectionResult(success=False, error_message=str(exc))

        logger.info("IMS returned %d stations", len(raw_stations))
        result = WeatherCollectionResult(success=True, stations_received=len(raw_stations))

        for raw_station in raw_stations:
            self._process_station(raw_station, result)

        logger.info(
            "Weather collection completed: %d/%d stations processed, %d failed, %d inactive skipped, "
            "%d observations saved, %d duplicates skipped, %d observation failures",
            result.stations_processed,
            result.stations_received,
            result.stations_failed,
            result.stations_skipped,
            result.observations_saved,
            result.duplicates_skipped,
            result.observations_failed,
        )
        return result

    def _process_station(self, raw_station: dict[str, Any], result: WeatherCollectionResult) -> None:
        """Map, persist, and collect weather data for a single raw IMS station.

        Every failure path is caught here so one station's problem never
        stops the rest of the run. Stations IMS reports as `active: false`
        are skipped entirely - their "latest" data can be decades old.
        """
        if raw_station.get("active") is False:
            logger.info("Skipping inactive IMS station %s", raw_station.get("stationId"))
            result.stations_skipped += 1
            return

        try:
            station = self._weather_mapper.map_station(raw_station)
        except WeatherMappingError as exc:
            logger.error("Failed to map station: %s", exc)
            result.stations_failed += 1
            return

        try:
            self._weather_repository.save_station(station)
        except WeatherRepositoryError as exc:
            logger.error("Failed to save station %s: %s", station.external_station_id, exc)
            result.stations_failed += 1
            return

        logger.info("Stored/updated station %s", station.external_station_id)
        result.stations_processed += 1

        self._collect_observation(station.external_station_id, result)

    def _collect_observation(self, external_station_id: int, result: WeatherCollectionResult) -> None:
        """Retrieve, map, and persist the current observation for an already-stored station."""
        try:
            raw_observation = self._ims_client.get_station_data(external_station_id)
        except IMSClientError as exc:
            logger.error("Failed to retrieve data for station %s: %s", external_station_id, exc)
            result.observations_failed += 1
            return

        try:
            observation = self._weather_mapper.map_observation(raw_observation)
        except WeatherMappingError as exc:
            logger.error("Failed to map observation for station %s: %s", external_station_id, exc)
            result.observations_failed += 1
            return

        try:
            save_result = self._weather_repository.save_observation(observation)
        except WeatherRepositoryError as exc:
            logger.error("Failed to save observation for station %s: %s", external_station_id, exc)
            result.observations_failed += 1
            return

        if save_result.is_duplicate:
            logger.info("Skipped duplicate observation for station %s", external_station_id)
            result.duplicates_skipped += 1
        else:
            logger.info("Stored observation for station %s", external_station_id)
            result.observations_saved += 1
