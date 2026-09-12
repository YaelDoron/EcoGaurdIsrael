"""Predefined demo locations for simulation scenarios."""
from __future__ import annotations

from types import MappingProxyType

from src.simulation.simulation_location import SimulationLocation

CARMEL_LOCATION = SimulationLocation(
    name="Carmel Demo Area",
    latitude=32.7310,
    longitude=35.0460,
)
JERUSALEM_FOREST_LOCATION = SimulationLocation(
    name="Jerusalem Forest Demo Area",
    latitude=31.7740,
    longitude=35.1390,
)
GALILEE_LOCATION = SimulationLocation(
    name="Galilee Demo Area",
    latitude=32.9650,
    longitude=35.3810,
)
GOLAN_LOCATION = SimulationLocation(
    name="Golan Heights Demo Area",
    latitude=33.0850,
    longitude=35.7800,
)
JUDEAN_HILLS_LOCATION = SimulationLocation(
    name="Judean Hills Demo Area",
    latitude=31.6650,
    longitude=35.0450,
)

SIMULATION_LOCATIONS = MappingProxyType(
    {
        "carmel": CARMEL_LOCATION,
        "jerusalem_forest": JERUSALEM_FOREST_LOCATION,
        "galilee": GALILEE_LOCATION,
        "golan": GOLAN_LOCATION,
        "judean_hills": JUDEAN_HILLS_LOCATION,
    }
)

DEFAULT_CARMEL_LOCATION = CARMEL_LOCATION


def get_simulation_location(key: str) -> SimulationLocation:
    """Return a predefined simulation location by stable machine key."""
    try:
        return SIMULATION_LOCATIONS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(SIMULATION_LOCATIONS))
        raise ValueError(f"Unsupported simulation location key {key!r}. Supported keys: {supported}.") from exc


def get_simulation_location_key(location: SimulationLocation) -> str | None:
    """Return the stable predefined key for a location, if it is registered."""
    for key, registered_location in SIMULATION_LOCATIONS.items():
        if location == registered_location:
            return key
    return None
