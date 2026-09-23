"""Shared spherical-earth geographic calculations.

Pure math only: no I/O, no repository access, no external geospatial
dependency. `EARTH_RADIUS_KM` matches the mean-Earth-radius value already used
by `fire_detection_calculator.py` and `fire_event_repository.py` prior to this
module's extraction, so existing distance behavior is preserved exactly.
"""
from __future__ import annotations

import math
from typing import Iterable, Protocol

EARTH_RADIUS_KM = 6371.0088

# First-Mile Heuristic Fallback threshold: the ordinary, harmless slack
# between a station's exact coordinate and the nearest real road node (a
# building's own driveway/forecourt, typically well under 100m). Shared by
# GlobalRouteMatrixBuilder (adds a time/distance correction to a route's
# totals past this gap) and ResponsePlanPresenter (bridges the same gap in
# `path_coordinates` with one honest straight-line segment from the
# station's real coordinate) - defined once here so both layers agree on
# exactly the same threshold, never two independently-tuned numbers that
# could drift apart and disagree about whether a given route needed either
# correction.
FIRST_MILE_PENALTY_THRESHOLD_KM = 0.1


class NamedCircularArea(Protocol):
    """Structural shape shared by any persisted circular area with a name.

    `FireDangerAreaSnapshot` already has exactly these fields; this Protocol
    lets `resolve_nearest_containing_area_name` stay a pure-math helper with
    no dependency on that (or any other) concrete model.
    """

    area_id: str
    area_name: str
    area_latitude: float
    area_longitude: float
    area_radius_km: float


def resolve_nearest_containing_area_name(
    latitude: float,
    longitude: float,
    areas: Iterable[NamedCircularArea],
) -> str | None:
    """Return the name of the nearest area whose circle actually contains the point.

    Only areas whose circle (center + radius_km) contains (latitude,
    longitude) are candidates; among those, the nearest center wins, with
    `area_id` as a deterministic tiebreaker for an exact distance tie.
    Returns `None` when no area's circle contains the point - being merely
    the geographically nearest area is not enough. Shared by every read-side
    "resolve a presentation-only area label for a coordinate" use (satellite
    hotspots, active FireEvents) so this logic is never duplicated.
    """
    best_area_name: str | None = None
    best_key: tuple[float, str] | None = None
    for area in areas:
        distance_km = haversine_distance_km(latitude, longitude, area.area_latitude, area.area_longitude)
        if distance_km > area.area_radius_km:
            continue
        key = (distance_km, area.area_id)
        if best_key is None or key < best_key:
            best_key = key
            best_area_name = area.area_name
    return best_area_name


def haversine_distance_km(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    """Return the great-circle distance in km between two coordinates."""
    first_latitude_rad = math.radians(first_latitude)
    second_latitude_rad = math.radians(second_latitude)
    latitude_delta = math.radians(second_latitude - first_latitude)
    longitude_delta = math.radians(second_longitude - first_longitude)
    a = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude_rad)
        * math.cos(second_latitude_rad)
        * math.sin(longitude_delta / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def initial_bearing_deg(
    first_latitude: float,
    first_longitude: float,
    second_latitude: float,
    second_longitude: float,
) -> float:
    """Return the initial compass bearing (deg clockwise from north, [0, 360))
    for traveling along the great circle from the first coordinate toward the
    second. Identical coordinates return 0.0."""
    first_latitude_rad = math.radians(first_latitude)
    second_latitude_rad = math.radians(second_latitude)
    longitude_delta_rad = math.radians(second_longitude - first_longitude)

    y = math.sin(longitude_delta_rad) * math.cos(second_latitude_rad)
    x = math.cos(first_latitude_rad) * math.sin(second_latitude_rad) - (
        math.sin(first_latitude_rad) * math.cos(second_latitude_rad) * math.cos(longitude_delta_rad)
    )
    return math.degrees(math.atan2(y, x)) % 360.0


def destination_point(
    latitude: float,
    longitude: float,
    bearing_deg: float,
    distance_km: float,
) -> tuple[float, float]:
    """Return the coordinate reached by traveling `distance_km` along the
    great circle at compass bearing `bearing_deg` (deg clockwise from north)
    starting from (`latitude`, `longitude`)."""
    angular_distance = distance_km / EARTH_RADIUS_KM
    bearing_rad = math.radians(bearing_deg)
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)

    destination_latitude_rad = math.asin(
        math.sin(latitude_rad) * math.cos(angular_distance)
        + math.cos(latitude_rad) * math.sin(angular_distance) * math.cos(bearing_rad)
    )
    destination_longitude_rad = longitude_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(angular_distance) * math.cos(latitude_rad),
        math.cos(angular_distance) - math.sin(latitude_rad) * math.sin(destination_latitude_rad),
    )
    destination_longitude_rad = (destination_longitude_rad + 3 * math.pi) % (2 * math.pi) - math.pi

    return math.degrees(destination_latitude_rad), math.degrees(destination_longitude_rad)
