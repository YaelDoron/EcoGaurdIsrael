"""Resolve the fire stations and available resources for an active wildfire.

OperationalContextService only resolves *which* stored fire stations (and
their available firefighting resources) are geographically relevant to an
incident location. It does not allocate or dispatch resources, and it does
not persist anything.

Stage 0 of the Global Multi-Incident Optimizer refactor (see the
architecture audit) added an optional `excluded_fire_event_id` parameter to
the resource-selection methods below: when given, resources already used by
another active FireEvent's CURRENT ResponsePlan are excluded from the
candidate pool (via CrossEventReservedResourceResolver), on top of the
existing `status == AVAILABLE` filter - never instead of it. The excluded
FireEvent's OWN current plan is never filtered out, so a FireEvent
replanning itself keeps its own already-assigned resources eligible.
Passing `None` (the default) preserves the exact prior behavior with no
cross-event exclusion at all - every call site not yet updated to pass a
fire_event_id is unaffected. This is a READ-side conflict-prevention
mechanism only: no FirefightingResource.status write, no new database
state, no transaction spanning two FireEvents' planning cycles - see
CrossEventReservedResourceResolver's own docstring for the remaining race
condition this does not close.
"""
from __future__ import annotations

import logging
import math

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.models.graph_edge import GraphEdge
from src.models.graph_node import GraphNode
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.road_network_repository import RoadNetworkRepository
from src.services.fire_danger import haversine_distance_km
from src.services.operational.road_network_fetcher import RoadNetworkFetcher
from src.services.resource_reservation import CrossEventReservedResourceResolver

logger = logging.getLogger(__name__)

DEFAULT_OPERATIONAL_RADIUS_KM = 5.0

# Progressive search radii tried, in order, when a narrower radius finds no
# stations - e.g. an incident far from any station. Only steps strictly
# wider than the caller's radius_km are actually tried (see
# _radius_search_sequence).
_EXPANDED_RADII_KM = (20.0, 50.0)

# ~11 km at Israel's latitude; extends the road-network bbox beyond the
# fire and its stations so large highway detours outside the immediate
# station vicinity are still included, rather than clipping routes exactly
# at their coordinates.
BUFFER_DEGREES = 0.1


class OperationalContext(BaseModel):
    """Aggregate operational context for an active wildfire.

    Combines the fire stations relevant to the incident, their available
    firefighting resources, and the road network subgraph covering both -
    everything Epic 5's routing logic needs in one object.
    """

    # Required because `stations`/`available_resources` are SQLAlchemy ORM
    # types, not Pydantic models. Note: if this ever becomes a FastAPI
    # response_model, these two fields will not JSON-serialize as-is - they
    # would need converting to the FireStation/FirefightingResource domain
    # dataclasses first.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stations: list[FireStationDB]
    available_resources: list[FirefightingResourceDB]
    road_nodes: list[GraphNode]
    road_edges: list[GraphEdge]


class OperationalContextService:
    """Builds the geographic operational context around an active wildfire."""

    def __init__(
        self,
        fire_station_repository: FireStationRepository | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
        road_network_repository: RoadNetworkRepository | None = None,
        road_network_fetcher: RoadNetworkFetcher | None = None,
        cross_event_reserved_resource_resolver: CrossEventReservedResourceResolver | None = None,
    ) -> None:
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )
        self._road_network_repository = road_network_repository or RoadNetworkRepository()
        self._road_network_fetcher = road_network_fetcher or RoadNetworkFetcher()
        self._cross_event_reserved_resource_resolver = (
            cross_event_reserved_resource_resolver or CrossEventReservedResourceResolver()
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

    def get_available_resources(
        self,
        stations: list[FireStationDB],
        excluded_fire_event_id: int | None = None,
    ) -> list[FirefightingResourceDB]:
        """Return AVAILABLE firefighting resources attached to the given stations.

        Returns an empty list immediately, without querying the database, if
        `stations` is empty.

        `excluded_fire_event_id`: see the module docstring - when given,
        resources already used by another active FireEvent's current plan
        are removed from the result. `None` (the default) applies no such
        filter.
        """
        if not stations:
            return []

        station_ids = [station.id for station in stations]
        resources = self._firefighting_resource_repository.get_available_resources(station_ids)
        if excluded_fire_event_id is None:
            return resources

        reserved_resource_ids = (
            self._cross_event_reserved_resource_resolver.get_resource_ids_reserved_by_other_active_plans(
                excluded_fire_event_id=excluded_fire_event_id
            )
        )
        if not reserved_resource_ids:
            return resources
        return [resource for resource in resources if resource.id not in reserved_resource_ids]

    def get_available_operational_context(
        self,
        latitude: float,
        longitude: float,
        min_resources: int = 1,
        excluded_fire_event_id: int | None = None,
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

        `excluded_fire_event_id`: see the module docstring - forwarded to
        every `get_available_resources` call this method makes, including
        during the closest-stations fallback.
        """
        self._validate_area(latitude, longitude, DEFAULT_OPERATIONAL_RADIUS_KM)
        self._validate_min_resources(min_resources)

        stations_with_distance = self._stations_with_distance(latitude, longitude)

        for candidate_radius_km in self._radius_search_sequence(DEFAULT_OPERATIONAL_RADIUS_KM):
            stations = self._within_radius(stations_with_distance, candidate_radius_km)
            available_resources = self.get_available_resources(stations, excluded_fire_event_id)
            if len(available_resources) >= min_resources:
                return stations, available_resources

        return self._closest_stations_until_enough_resources(
            stations_with_distance, min_resources, excluded_fire_event_id
        )

    def build_context(
        self,
        db: Session,
        fire_latitude: float,
        fire_longitude: float,
        min_resources: int = 1,
        excluded_fire_event_id: int | None = None,
    ) -> OperationalContext:
        """Build the full operational context for an active wildfire.

        Combines the nearby fire stations and their available resources
        (get_available_operational_context) with the road network covering
        the fire and those stations (RoadNetworkRepository.get_network_in_bbox),
        so Epic 5's routing has everything it needs from one call.

        Road network data is lazily loaded: the database is checked first
        (RoadNetworkRepository.get_network_in_bbox), and only on a cache
        miss - no nodes stored for this bbox yet - is OSM queried live
        (RoadNetworkFetcher.fetch_network_in_bbox), with the result saved
        back to the database so the next call for an overlapping area is a
        cache hit.

        `db` is a caller-owned SQLAlchemy session (e.g. a FastAPI
        `Depends(get_db)` session) - RoadNetworkRepository takes it per call
        rather than owning its own session factory, unlike the other two
        repositories injected into this service.

        `excluded_fire_event_id`: see the module docstring.
        """
        stations, available_resources = self.get_available_operational_context(
            fire_latitude, fire_longitude, min_resources, excluded_fire_event_id
        )

        min_lat, max_lat, min_lon, max_lon = self._calculate_bounding_box(
            fire_latitude, fire_longitude, stations
        )
        road_nodes, road_edges = self._road_network_repository.get_network_in_bbox(
            db, min_lat, max_lat, min_lon, max_lon
        )

        if not road_nodes:
            road_nodes, road_edges = self._load_road_network_from_osm(
                db, min_lat, max_lat, min_lon, max_lon
            )

        return OperationalContext(
            stations=stations,
            available_resources=available_resources,
            road_nodes=road_nodes,
            road_edges=road_edges,
        )

    def _load_road_network_from_osm(
        self,
        db: Session,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Fetch a bbox's road network from OSM and cache it, on a database cache miss.

        Called only when RoadNetworkRepository.get_network_in_bbox found no
        stored nodes for the bbox. If OSM itself returns no network either
        (e.g. a genuinely road-less area, or an OSM/Overpass failure -
        RoadNetworkFetcher already degrades those to empty lists rather than
        raising), nothing is saved and empty lists are returned as-is.
        """
        logger.info(
            "Road network cache miss for bbox (min_lat=%.5f, max_lat=%.5f, min_lon=%.5f, max_lon=%.5f); "
            "fetching from OSM.",
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        )
        fetched_nodes, fetched_edges = self._road_network_fetcher.fetch_network_in_bbox(
            min_lat, max_lat, min_lon, max_lon
        )

        if not fetched_nodes:
            logger.warning("OSM fetch returned no road network for this bounding box either.")
            return fetched_nodes, fetched_edges

        self._road_network_repository.save_network(db, fetched_nodes, fetched_edges)
        logger.info(
            "Fetched and cached %d node(s) and %d edge(s) from OSM.",
            len(fetched_nodes),
            len(fetched_edges),
        )
        return fetched_nodes, fetched_edges

    def _calculate_bounding_box(
        self,
        fire_latitude: float,
        fire_longitude: float,
        stations: list[FireStationDB],
    ) -> tuple[float, float, float, float]:
        """Return (min_lat, max_lat, min_lon, max_lon) covering the fire and all given stations.

        Adds a BUFFER_DEGREES margin on every side so the returned road
        network extends beyond the outermost station/fire point. Bounds are
        clamped to valid lat/lon ranges so an incident near a coordinate
        extreme can't push the bbox out of range and trip
        RoadNetworkRepository's own validation.
        """
        latitudes = [fire_latitude] + [station.latitude for station in stations]
        longitudes = [fire_longitude] + [station.longitude for station in stations]

        min_lat = max(-90.0, min(latitudes) - BUFFER_DEGREES)
        max_lat = min(90.0, max(latitudes) + BUFFER_DEGREES)
        min_lon = max(-180.0, min(longitudes) - BUFFER_DEGREES)
        max_lon = min(180.0, max(longitudes) + BUFFER_DEGREES)

        return min_lat, max_lat, min_lon, max_lon

    def _closest_stations_until_enough_resources(
        self,
        stations_with_distance: list[tuple[FireStationDB, float]],
        min_resources: int,
        excluded_fire_event_id: int | None = None,
    ) -> tuple[list[FireStationDB], list[FirefightingResourceDB]]:
        ordered_stations = [
            station for station, _distance in sorted(stations_with_distance, key=lambda pair: pair[1])
        ]

        accumulated_stations: list[FireStationDB] = []
        available_resources: list[FirefightingResourceDB] = []
        for station in ordered_stations:
            accumulated_stations.append(station)
            available_resources = self.get_available_resources(accumulated_stations, excluded_fire_event_id)
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
