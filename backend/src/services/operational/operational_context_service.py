"""Resolve the fire stations and available resources for an active wildfire.

OperationalContextService only resolves *which* stored fire stations (and
their available firefighting resources) are geographically relevant to an
incident location. It does not allocate or dispatch resources, and it does
not persist anything.
"""
from __future__ import annotations

import math

from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.services.fire_danger import haversine_distance_km

DEFAULT_OPERATIONAL_RADIUS_KM = 5.0

# Progressive search radii tried, in order, when a narrower radius finds no
# stations - e.g. an incident far from any station. Only steps strictly
# wider than the caller's radius_km are actually tried (see
# _radius_search_sequence).
_EXPANDED_RADII_KM = (20.0, 50.0)


class OperationalContextService:
    """Builds the geographic operational context around an active wildfire."""

    def __init__(
        self,
        fire_station_repository: FireStationRepository | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
    ) -> None:
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )

    def get_stations_in_operational_area(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = DEFAULT_OPERATIONAL_RADIUS_KM,
    ) -> list[FireStationDB]:
        """Return stored fire stations within `radius_km` of (latitude, longitude).

        Distance to each station's stored coordinates is computed with the
        Haversine formula. If no station falls within `radius_km`, the search
        radius is progressively widened (see `_EXPANDED_RADII_KM`) so sparsely
        covered areas still produce a usable result. If no station is found
        even at the widest radius, the single closest station overall is
        returned instead of an empty list, so a routing algorithm (Epic 5)
        always has at least one station to work with.

        Results are always ordered nearest-first. Note this method only
        looks at station *presence* - a returned station may still have zero
        available resources. Use get_available_operational_context() when
        resource availability matters.
        """
        self._validate_area(latitude, longitude, radius_km)

        stations_with_distance = self._stations_with_distance(latitude, longitude)

        for candidate_radius_km in self._radius_search_sequence(radius_km):
            in_range = self._within_radius(stations_with_distance, candidate_radius_km)
            if in_range:
                return in_range

        return self._closest_station_fallback(stations_with_distance)

    def get_available_resources(self, stations: list[FireStationDB]) -> list[FirefightingResourceDB]:
        """Return AVAILABLE firefighting resources attached to the given stations.

        Returns an empty list immediately, without querying the database, if
        `stations` is empty.
        """
        if not stations:
            return []

        station_ids = [station.id for station in stations]
        return self._firefighting_resource_repository.get_available_resources(station_ids)

    def get_available_operational_context(
        self,
        latitude: float,
        longitude: float,
        min_resources: int = 1,
    ) -> tuple[list[FireStationDB], list[FirefightingResourceDB]]:
        """Return (stations, available_resources) with at least `min_resources` trucks.

        A station existing nearby is not enough on its own: if it has no
        available trucks, a routing algorithm would still starve. This
        method widens the search radius (5.0 -> 20.0 -> 50.0 km, the same
        progression as get_stations_in_operational_area) until the stations
        found within a given radius collectively have at least
        `min_resources` AVAILABLE resources.

        If even the widest radius does not yield enough available resources
        (e.g. a genuine regional shortage), it falls back to accumulating
        the absolute closest stations one at a time, regardless of distance,
        until `min_resources` is met or every stored station has been
        included - so a routing algorithm gets the best obtainable answer
        rather than an empty or insufficient one.

        Both returned lists are ordered nearest-station-first.
        """
        self._validate_area(latitude, longitude, DEFAULT_OPERATIONAL_RADIUS_KM)
        self._validate_min_resources(min_resources)

        stations_with_distance = self._stations_with_distance(latitude, longitude)

        for candidate_radius_km in self._radius_search_sequence(DEFAULT_OPERATIONAL_RADIUS_KM):
            stations = self._within_radius(stations_with_distance, candidate_radius_km)
            available_resources = self.get_available_resources(stations)
            if len(available_resources) >= min_resources:
                return stations, available_resources

        return self._closest_stations_until_enough_resources(stations_with_distance, min_resources)

    def _closest_stations_until_enough_resources(
        self,
        stations_with_distance: list[tuple[FireStationDB, float]],
        min_resources: int,
    ) -> tuple[list[FireStationDB], list[FirefightingResourceDB]]:
        ordered_stations = [
            station for station, _distance in sorted(stations_with_distance, key=lambda pair: pair[1])
        ]

        accumulated_stations: list[FireStationDB] = []
        available_resources: list[FirefightingResourceDB] = []
        for station in ordered_stations:
            accumulated_stations.append(station)
            available_resources = self.get_available_resources(accumulated_stations)
            if len(available_resources) >= min_resources:
                break

        return accumulated_stations, available_resources

    def _stations_with_distance(
        self, latitude: float, longitude: float
    ) -> list[tuple[FireStationDB, float]]:
        return [
            (station, haversine_distance_km(latitude, longitude, station.latitude, station.longitude))
            for station in self._fire_station_repository.get_all_stations()
        ]

    @staticmethod
    def _within_radius(
        stations_with_distance: list[tuple[FireStationDB, float]],
        radius_km: float,
    ) -> list[FireStationDB]:
        in_range = sorted(
            (pair for pair in stations_with_distance if pair[1] <= radius_km),
            key=lambda pair: pair[1],
        )
        return [station for station, _distance in in_range]

    @staticmethod
    def _closest_station_fallback(
        stations_with_distance: list[tuple[FireStationDB, float]],
    ) -> list[FireStationDB]:
        """Return the single nearest station, or [] if there are none stored at all."""
        if not stations_with_distance:
            return []
        closest_station, _distance = min(stations_with_distance, key=lambda pair: pair[1])
        return [closest_station]

    @staticmethod
    def _radius_search_sequence(radius_km: float) -> list[float]:
        """Return the ascending radii to try, starting from `radius_km`.

        Expansion steps are only included when strictly larger than the
        previous one, so calling with an already-wide radius_km does not
        redundantly re-check smaller expansion steps.
        """
        radii = [radius_km]
        for expanded_radius_km in _EXPANDED_RADII_KM:
            if expanded_radius_km > radii[-1]:
                radii.append(expanded_radius_km)
        return radii

    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        for field_name, value in (("latitude", latitude), ("longitude", longitude)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite number, got {value!r}.")
        if not -90 <= latitude <= 90:
            raise ValueError(f"latitude must be within [-90, 90], got {latitude!r}.")
        if not -180 <= longitude <= 180:
            raise ValueError(f"longitude must be within [-180, 180], got {longitude!r}.")

    @classmethod
    def _validate_area(cls, latitude: float, longitude: float, radius_km: float) -> None:
        cls._validate_coordinates(latitude, longitude)
        if isinstance(radius_km, bool) or not isinstance(radius_km, (int, float)) or not math.isfinite(radius_km):
            raise ValueError(f"radius_km must be a finite number, got {radius_km!r}.")
        if radius_km <= 0:
            raise ValueError(f"radius_km must be greater than 0, got {radius_km!r}.")

    @staticmethod
    def _validate_min_resources(min_resources: int) -> None:
        if isinstance(min_resources, bool) or not isinstance(min_resources, int) or min_resources <= 0:
            raise ValueError(f"min_resources must be a positive integer, got {min_resources!r}.")
