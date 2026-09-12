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
    "carmel": "הכרמל",
    "jerusalem_forest": "יער ירושלים",
    "galilee": "הגליל",
    "golan": "רמת הגולן",
    "judean_hills": "הרי יהודה",
}

_INITIAL_TITLE_TEMPLATES = (
    "דיווח ראשוני על עשן ולהבות באזור {location}",
    "דיווחים ראשוניים על מוקדי אש באזור {location}",
    "עשן כבד נראה באזור {location}",
)
_INITIAL_SUMMARY_TEMPLATES = (
    "מספר דיווחים מהאזור מתארים עשן ולהבות שנצפו בסמוך לאזור {location}.",
    "תושבים ומטיילים באזור מדווחים על עשן הנראה ממספר נקודות סביב {location}.",
    "התקבלו דיווחים ראשוניים על שריפה מתפתחת באזור {location}.",
)
_FOLLOW_UP_TITLE_TEMPLATES = (
    "דיווחים נוספים על התפשטות האש באזור {location}",
    "עדכון נוסף: מוקדי אש נראים באזור {location}",
    "האש ממשיכה להיראות במספר מוקדים באזור {location}",
)
_FOLLOW_UP_SUMMARY_TEMPLATES = (
    "דיווחים נוספים מהאזור מצביעים על כך שהאש ממשיכה להתפשט ונצפים מוקדים נוספים.",
    "מידע נוסף שמגיע מהאזור מתאר עשן סמיך ומוקדי אש נוספים סביב {location}.",
    "דיווחים מתמשכים מצביעים על התפתחות האירוע באזור {location}.",
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
    ) -> GeneratedNewsData:
        """Generate deterministic simulated news data for one simulation timestamp."""
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
    ) -> int:
        seed_material = "|".join(
            [
                str(self._seed),
                scenario_type.value,
                timestamp.isoformat(),
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
