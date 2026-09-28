"""Generate deterministic simulated wildfire news reports."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import random
import re

from src.models.fire_report import WildfireReport
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import DEFAULT_CARMEL_LOCATION, get_simulation_location_key

SIMULATED_NEWS_SOURCE_FEED = "EcoGuard Simulation News"
SIMULATED_NEWS_BASE_URL = "https://simulation.ecoguard.local"
SIMULATED_NEWS_MAX_RADIUS_KM = 1.0
_KM_PER_LATITUDE_DEGREE = 111.32

_LOCATION_REPORT_NAMES = {
    "carmel": "Carmel",
    "jerusalem_forest": "Jerusalem Forest",
    "galilee": "Galilee",
    "golan": "Golan Heights",
    "judean_hills": "Judean Hills",
}

_INITIAL_TITLE_TEMPLATES = (
    "Initial report of smoke and flames in the {location} area",
    "Initial reports of fire hotspots in the {location} area",
    "Heavy smoke seen in the {location} area",
)
_INITIAL_SUMMARY_TEMPLATES = (
    "Several reports from the area describe smoke and flames observed near the {location} area.",
    "Residents and hikers in the area report smoke visible from several points around {location}.",
    "Initial reports have been received of a developing fire in the {location} area.",
)
_FOLLOW_UP_TITLE_TEMPLATES = (
    "Further reports of fire spreading in the {location} area",
    "Update: fire hotspots visible in the {location} area",
    "Fire continues to be seen at multiple hotspots in the {location} area",
)
_FOLLOW_UP_SUMMARY_TEMPLATES = (
    "Additional reports from the area indicate the fire is continuing to spread and more hotspots are being observed.",
    "Further information from the area describes dense smoke and additional fire hotspots around {location}.",
    "Ongoing reports indicate the event is developing in the {location} area.",
)


@dataclass(frozen=True)
class GeneratedNewsData:
    """Generated simulated wildfire news reports."""

    reports: tuple[WildfireReport, ...]


class NewsDataGenerator:
    """Generate WildfireReport objects for demo scenarios."""

    def __init__(self, seed: int = 42) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")
        self._seed = seed

    def generate(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
        report_index: int = 0,
        seed_key: str | None = None,
    ) -> GeneratedNewsData:
        """Generate deterministic simulated news data for one simulation timestamp.

        `seed_key` (the event's schedule identity, passed by the executor) replaces
        the absolute timestamp in the RNG seed; `timestamp` still sets
        published_at/fetched_at and the report's unique source URL.
        """
        if not isinstance(scenario_type, ScenarioType):
            raise ValueError(f"scenario_type must be a ScenarioType, got {scenario_type!r}")
        if not isinstance(timestamp, datetime):
            raise ValueError(f"timestamp must be a datetime, got {timestamp!r}")
        if not isinstance(location, SimulationLocation):
            raise ValueError(f"location must be a SimulationLocation, got {location!r}")
        if isinstance(report_index, bool) or not isinstance(report_index, int) or report_index < 0:
            raise ValueError(f"report_index must be a non-negative integer, got {report_index!r}")

        if scenario_type is not ScenarioType.ACTIVE_FIRE:
            return GeneratedNewsData(reports=())

        rng = random.Random(
            self._derive_report_seed(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                report_index=report_index,
                seed_key=seed_key,
            )
        )
        report_location_name = self._report_location_name(location)
        title, summary = self._generate_text(
            rng=rng,
            report_location_name=report_location_name,
            report_index=report_index,
        )
        latitude, longitude = self._generate_report_coordinates(rng, location)
        report = WildfireReport(
            source_url=self._source_url(
                scenario_type=scenario_type,
                timestamp=timestamp,
                location=location,
                report_index=report_index,
            ),
            source_feed=SIMULATED_NEWS_SOURCE_FEED,
            title=title,
            summary=summary,
            location_name=report_location_name,
            latitude=latitude,
            longitude=longitude,
            published_at=timestamp,
            fetched_at=timestamp,
        )
        return GeneratedNewsData(reports=(report,))

    def _derive_report_seed(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        report_index: int,
        seed_key: str | None = None,
    ) -> int:
        seed_material = "|".join(
            [
                str(self._seed),
                scenario_type.value,
                seed_key if seed_key is not None else timestamp.isoformat(),
                location.name,
                f"{location.latitude:.6f}",
                f"{location.longitude:.6f}",
                str(report_index),
            ]
        )
        return self._stable_int(seed_material)

    @staticmethod
    def _generate_text(
        rng: random.Random,
        report_location_name: str,
        report_index: int,
    ) -> tuple[str, str]:
        if report_index == 0:
            title_template = rng.choice(_INITIAL_TITLE_TEMPLATES)
            summary_template = rng.choice(_INITIAL_SUMMARY_TEMPLATES)
        else:
            title_template = rng.choice(_FOLLOW_UP_TITLE_TEMPLATES)
            summary_template = rng.choice(_FOLLOW_UP_SUMMARY_TEMPLATES)

        return (
            title_template.format(location=report_location_name),
            summary_template.format(location=report_location_name),
        )

    @staticmethod
    def _generate_report_coordinates(
        rng: random.Random,
        location: SimulationLocation,
    ) -> tuple[float, float]:
        radius_km = rng.uniform(0.0, SIMULATED_NEWS_MAX_RADIUS_KM)
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

    def _source_url(
        self,
        scenario_type: ScenarioType,
        timestamp: datetime,
        location: SimulationLocation,
        report_index: int,
    ) -> str:
        location_key = self._location_key(location)
        timestamp_slug = re.sub(r"[^0-9A-Za-z]+", "-", timestamp.isoformat()).strip("-")
        return (
            f"{SIMULATED_NEWS_BASE_URL}/{scenario_type.value}/"
            f"{location_key}/report-{report_index + 1}/{timestamp_slug}"
        )

    @staticmethod
    def _location_key(location: SimulationLocation) -> str:
        predefined_key = get_simulation_location_key(location)
        if predefined_key is not None:
            return predefined_key
        return re.sub(r"[^a-z0-9]+", "-", location.name.strip().lower()).strip("-") or "location"

    @staticmethod
    def _report_location_name(location: SimulationLocation) -> str:
        location_key = get_simulation_location_key(location)
        if location_key is not None:
            return _LOCATION_REPORT_NAMES[location_key]
        return location.name

    @staticmethod
    def _stable_int(seed_material: str) -> int:
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)
