"""Generate deterministic simulated weather domain objects."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import random
import re

from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import DEFAULT_CARMEL_LOCATION, get_simulation_location_key

SIMULATED_WEATHER_STATION_ID_BASE = 900_000
SIMULATED_WEATHER_STATION_ID_SPAN = 10_000
SIMULATED_WEATHER_STATIONS_PER_LOCATION = 3

# Small deterministic offsets around the event anchor. These are roughly
# 1.5-2.0 km from the anchor in Israel, enough to look like a local station
# network while remaining in the same demo area.
SIMULATED_WEATHER_STATION_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.0120, -0.0100),
    (0.0140, 0.0000),
    (-0.0020, 0.0160),
)


@dataclass(frozen=True)
class WeatherScenarioProfile:
    """Demo weather ranges for scenario generation, not fire-detection thresholds."""

    temperature_celsius: tuple[float, float]
    relative_humidity_percent: tuple[float, float]
    wind_speed_kmh: tuple[float, float]
    wind_gust_extra_kmh: tuple[float, float]
    rainfall_mm: tuple[float, float]


WEATHER_SCENARIO_PROFILES: dict[ScenarioType, WeatherScenarioProfile] = {
    ScenarioType.LOW_RISK_NO_FIRE: WeatherScenarioProfile(
        temperature_celsius=(18.0, 28.0),
        relative_humidity_percent=(45.0, 75.0),
        wind_speed_kmh=(0.0, 15.0),
        wind_gust_extra_kmh=(0.0, 10.0),
        rainfall_mm=(0.0, 5.0),
    ),
    ScenarioType.HIGH_RISK_NO_FIRE: WeatherScenarioProfile(
        temperature_celsius=(34.0, 41.0),
        relative_humidity_percent=(10.0, 25.0),
        wind_speed_kmh=(20.0, 40.0),
        wind_gust_extra_kmh=(2.0, 18.0),
        rainfall_mm=(0.0, 0.0),
    ),
    ScenarioType.ACTIVE_FIRE: WeatherScenarioProfile(
        temperature_celsius=(33.0, 41.0),
        relative_humidity_percent=(10.0, 28.0),
        wind_speed_kmh=(15.0, 40.0),
        wind_gust_extra_kmh=(2.0, 18.0),
        rainfall_mm=(0.0, 0.0),
    ),
}


@dataclass(frozen=True)
class GeneratedStationWeather:
    """Generated weather for one simulated station."""

    station: WeatherStation
    observation: WeatherObservation


@dataclass(frozen=True)
class GeneratedWeatherData:
    """Generated weather measurements for a local simulated station network."""

    measurements: tuple[GeneratedStationWeather, ...]

    @property
    def stations(self) -> tuple[WeatherStation, ...]:
        return tuple(measurement.station for measurement in self.measurements)

    @property
    def observations(self) -> tuple[WeatherObservation, ...]:
        return tuple(measurement.observation for measurement in self.measurements)


class WeatherDataGenerator:
    """Generate WeatherStation and WeatherObservation objects for demo scenarios."""

    def __init__(self, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    def generate(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    ) -> GeneratedWeatherData:
        """Generate deterministic simulated weather data for one simulation timestamp."""
        if not isinstance(scenario_type, ScenarioType):
            raise ValueError(f"scenario_type must be a ScenarioType, got {scenario_type!r}")
        if not isinstance(timestamp, datetime):
            raise ValueError(f"timestamp must be a datetime, got {timestamp!r}")
        if not isinstance(location, SimulationLocation):
            raise ValueError(f"location must be a SimulationLocation, got {location!r}")

        profile = WEATHER_SCENARIO_PROFILES[scenario_type]
        measurements = tuple(
            self._generate_station_weather(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                station_index=station_index,
                profile=profile,
            )
            for station_index in range(1, SIMULATED_WEATHER_STATIONS_PER_LOCATION + 1)
        )
        return GeneratedWeatherData(measurements=measurements)

    def _generate_station_weather(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        station_index: int,
        profile: WeatherScenarioProfile,
    ) -> GeneratedStationWeather:
        station = self._build_station(location=location, station_index=station_index)
        rng = random.Random(
            self._derive_observation_seed(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                station_external_id=station.external_station_id,
            )
        )
        wind_speed = self._round_measurement(self._uniform(rng, profile.wind_speed_kmh))
        wind_gust = self._round_measurement(wind_speed + self._uniform(rng, profile.wind_gust_extra_kmh))
        observation = WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=timestamp,
            temperature=self._round_measurement(self._uniform(rng, profile.temperature_celsius)),
            relative_humidity=self._round_measurement(
                self._uniform(rng, profile.relative_humidity_percent)
            ),
            wind_speed=wind_speed,
            wind_direction=self._round_measurement(rng.uniform(0.0, 360.0)),
            wind_gust=wind_gust,
            rainfall=self._generate_rainfall(rng, profile),
        )
        return GeneratedStationWeather(station=station, observation=observation)

    def _build_station(self, location: SimulationLocation, station_index: int) -> WeatherStation:
        station_key = self._station_key(location)
        station_id = self._station_id(location=location, station_index=station_index)
        latitude_offset, longitude_offset = SIMULATED_WEATHER_STATION_OFFSETS[station_index - 1]
        return WeatherStation(
            external_station_id=station_id,
            name=f"SIM-{station_key}-{station_index:02d}",
            latitude=round(location.latitude + latitude_offset, 6),
            longitude=round(location.longitude + longitude_offset, 6),
            region_id=None,
            active=True,
        )

    def _derive_observation_seed(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        station_external_id: int,
    ) -> int:
        seed_material = "|".join(
            [
                str(self._seed),
                scenario_type.value,
                timestamp.isoformat(),
                location.name,
                f"{location.latitude:.6f}",
                f"{location.longitude:.6f}",
                str(station_external_id),
            ]
        )
        return self._stable_int(seed_material)

    @staticmethod
    def _station_id(location: SimulationLocation, station_index: int) -> int:
        seed_material = "|".join(
            [
                "sim-weather-station",
                location.name,
                f"{location.latitude:.6f}",
                f"{location.longitude:.6f}",
                str(station_index),
            ]
        )
        suffix = WeatherDataGenerator._stable_int(seed_material) % SIMULATED_WEATHER_STATION_ID_SPAN
        return SIMULATED_WEATHER_STATION_ID_BASE + suffix

    @staticmethod
    def _stable_int(seed_material: str) -> int:
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    @staticmethod
    def _uniform(rng: random.Random, value_range: tuple[float, float]) -> float:
        low, high = value_range
        if low == high:
            return low
        return rng.uniform(low, high)

    @staticmethod
    def _generate_rainfall(rng: random.Random, profile: WeatherScenarioProfile) -> float:
        low, high = profile.rainfall_mm
        if low == high:
            return low
        if rng.random() < 0.55:
            return 0.0
        return WeatherDataGenerator._round_measurement(rng.uniform(max(0.1, low), high))

    @staticmethod
    def _round_measurement(value: float) -> float:
        return round(value, 1)

    @staticmethod
    def _slug_location_name(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().upper()).strip("-")
        return slug or "LOCATION"

    @staticmethod
    def _station_key(location: SimulationLocation) -> str:
        predefined_key = get_simulation_location_key(location)
        if predefined_key is not None:
            return predefined_key.replace("_", "-").upper()
        return WeatherDataGenerator._slug_location_name(location.name)
