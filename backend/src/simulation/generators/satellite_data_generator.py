"""Generate deterministic simulated satellite thermal detections."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import random

from src.models.satellite_hotspot import SatelliteHotspot
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import DEFAULT_CARMEL_LOCATION

SIMULATED_HOTSPOT_MIN_RADIUS_KM = 0.2
SIMULATED_HOTSPOT_MAX_RADIUS_KM = 2.0
SIMULATED_HOTSPOTS_PER_ACTIVE_FIRE_EVENT = 1
SIMULATED_FRP_RANGE = (10.0, 120.0)
SIMULATED_BRIGHTNESS_RANGE = (300.0, 380.0)
SIMULATED_CONFIDENCE_VALUES = ("n", "h")
SIMULATED_SATELLITE = "SIM-NOAA-20"
SIMULATED_INSTRUMENT = "VIIRS"
_KM_PER_LATITUDE_DEGREE = 111.32


@dataclass(frozen=True)
class GeneratedSatelliteData:
    """Generated satellite thermal detections."""

    hotspots: tuple[SatelliteHotspot, ...]


class SatelliteDataGenerator:
    """Generate SatelliteHotspot objects for demo scenarios."""

    def __init__(self, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    def generate(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    ) -> GeneratedSatelliteData:
        """Generate deterministic simulated satellite data for one simulation timestamp."""
        if not isinstance(scenario_type, ScenarioType):
            raise ValueError(f"scenario_type must be a ScenarioType, got {scenario_type!r}")
        if not isinstance(timestamp, datetime):
            raise ValueError(f"timestamp must be a datetime, got {timestamp!r}")
        if not isinstance(location, SimulationLocation):
            raise ValueError(f"location must be a SimulationLocation, got {location!r}")

        if scenario_type is not ScenarioType.ACTIVE_FIRE:
            return GeneratedSatelliteData(hotspots=())

        hotspots = tuple(
            self._generate_hotspot(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                hotspot_index=hotspot_index,
            )
            for hotspot_index in range(1, SIMULATED_HOTSPOTS_PER_ACTIVE_FIRE_EVENT + 1)
        )
        return GeneratedSatelliteData(hotspots=hotspots)

    def _generate_hotspot(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        hotspot_index: int,
    ) -> SatelliteHotspot:
        rng = random.Random(
            self._derive_hotspot_seed(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                hotspot_index=hotspot_index,
            )
        )
        latitude, longitude = self._generate_nearby_coordinates(rng, location)
        return SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=timestamp,
            confidence=rng.choice(SIMULATED_CONFIDENCE_VALUES),
            frp=self._round_measurement(rng.uniform(*SIMULATED_FRP_RANGE)),
            brightness=self._round_measurement(rng.uniform(*SIMULATED_BRIGHTNESS_RANGE)),
            satellite=SIMULATED_SATELLITE,
            instrument=SIMULATED_INSTRUMENT,
            day_night=self._day_night(timestamp),
        )

    def _derive_hotspot_seed(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        hotspot_index: int,
    ) -> int:
        seed_material = "|".join(
            [
                str(self._seed),
                scenario_type.value,
                timestamp.isoformat(),
                location.name,
                f"{location.latitude:.6f}",
                f"{location.longitude:.6f}",
                str(hotspot_index),
            ]
        )
        return self._stable_int(seed_material)

    @staticmethod
    def _generate_nearby_coordinates(
        rng: random.Random,
        location: SimulationLocation,
    ) -> tuple[float, float]:
        radius_km = rng.uniform(SIMULATED_HOTSPOT_MIN_RADIUS_KM, SIMULATED_HOTSPOT_MAX_RADIUS_KM)
        bearing_radians = rng.uniform(0.0, 2.0 * math.pi)
        north_km = radius_km * math.cos(bearing_radians)
        east_km = radius_km * math.sin(bearing_radians)

        latitude_delta = north_km / _KM_PER_LATITUDE_DEGREE
        longitude_scale = _KM_PER_LATITUDE_DEGREE * math.cos(math.radians(location.latitude))
        longitude_delta = east_km / longitude_scale
        return (
            round(location.latitude + latitude_delta, 6),
            round(location.longitude + longitude_delta, 6),
        )

    @staticmethod
    def _stable_int(seed_material: str) -> int:
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    @staticmethod
    def _round_measurement(value: float) -> float:
        return round(value, 1)

    @staticmethod
    def _day_night(timestamp: datetime) -> str:
        return "D" if 6 <= timestamp.hour < 18 else "N"
