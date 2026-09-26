"""Scenario/location data validation for the canonical demo locations.

No live external calls: Copernicus results are pinned as a recorded fixture
(production path after the Task 13 sampling fix - 20 x 20 = 400 valid pixels
per 2 km box - `scripts/audit_simulation_locations.py`, 2026-09-25) so the
mapping logic is exercised deterministically. Re-run the audit script to
refresh the fixture if the location data ever changes.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.external.copernicus import CopernicusCoverFraction, CopernicusLandCoverStatistics
from src.services.fire_severity.fire_severity_input_config import VEGETATION_RADIUS_KM
from src.services.fire_spread.fire_spread_input_service import _map_dominant_land_cover
from src.models.fire_spread_fuel_class import FireSpreadFuelClass as FC
from src.simulation import SIMULATION_LOCATIONS, ScenarioType
from src.simulation.generators import news_data_generator as ndg
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import (
    SIMULATED_HOTSPOT_MAX_RADIUS_KM,
    SIMULATED_HOTSPOT_MIN_RADIUS_KM,
    SatelliteDataGenerator,
)
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.simulation_locations import GALILEE_LOCATION
from src.simulation.simulation_scenario import build_operations_demo_scenario
from tests.services.fire_severity.test_fire_severity_input_service import AS_OF, build_service, make_event

REPO_ROOT = Path(__file__).resolve().parents[3]
TIMESTAMP = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
OLD_BUILT_UP_GALILEE = (32.9650, 35.3810)  # Beit Jann village - replaced

# Corrected production-path dominant class at each anchor (Copernicus Global Land Cover
# 100 m, 2019, +/-1 km box sampled as 400 pixels). Pre-fix single-pixel values were wrong.
RECORDED_ANCHOR_DOMINANT_CLASS = {
    "carmel": "Shrub cover",  # Shrub 28 / Tree 24 / Built-up 22 / Grass 16 / Crop 8 / Bare 2
    "jerusalem_forest": "Shrub cover",  # Shrub 32 / Grass 28 / Tree 27 / Built-up 6 / Crop 4 / Bare 3
    "galilee": "Tree cover",  # Har Kamon: Tree 35 / Shrub 30 / Grass 18 / Built-up 8 / Crop 7 / Bare 2
    "golan": "Crop cover",  # Crop 67 / Grass 14 / Shrub 9 / Tree 5 / Built-up 3 / Bare 2
    "judean_hills": "Grass cover",  # Grass 40 / Shrub 19 / Crop 18 / Tree 15 / Bare 5 / Built-up 2
}
# "Tree cover" maps to the EcoGuard-derived GENERIC_TREE spread fuel (Task 14, §4.3.2).
TREE_DOMINANT_ANCHORS = {"galilee"}
BURNABLE = {FC.GRASSLAND, FC.SHRUBS, FC.AGRO_FORESTRY, FC.CONIFERS_FIRE_PRONE, FC.BROADLEAVES_FIRE_PRONE}
ENGLISH_AREA_NAMES = {
    "carmel": "Carmel",
    "jerusalem_forest": "Jerusalem Forest",
    "galilee": "Galilee",
    "golan": "Golan Heights",
    "judean_hills": "Judean Hills",
}


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


# --- Location definitions ---------------------------------------------------


def test_canonical_ids_names_and_coordinates_are_unique_and_inside_israel():
    locations = dict(SIMULATION_LOCATIONS)
    assert set(locations) == {"carmel", "jerusalem_forest", "galilee", "golan", "judean_hills"}
    assert len({loc.name for loc in locations.values()}) == len(locations)
    assert len({(loc.latitude, loc.longitude) for loc in locations.values()}) == len(locations)
    for location in locations.values():
        assert 29.4 <= location.latitude <= 33.4
        assert 34.2 <= location.longitude <= 35.9


def test_canonical_locations_are_defined_only_in_the_registry_module():
    src = REPO_ROOT / "backend" / "src"
    definers = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if re.search(r'SimulationLocation\(\s*name="[^"]+ Demo Area"', path.read_text(encoding="utf-8"))
    ]
    assert definers == ["simulation/simulation_locations.py"]


def test_galilee_is_no_longer_the_built_up_beit_jann_point_and_stays_in_the_galilee():
    assert (GALILEE_LOCATION.latitude, GALILEE_LOCATION.longitude) != OLD_BUILT_UP_GALILEE
    assert (GALILEE_LOCATION.latitude, GALILEE_LOCATION.longitude) == (32.9150, 35.3450)
    assert 32.6 <= GALILEE_LOCATION.latitude <= 33.3 and 35.0 <= GALILEE_LOCATION.longitude <= 35.7


# --- Vegetation sanity (recorded fixture, no live call) -----------------------


@pytest.mark.parametrize("key", sorted(RECORDED_ANCHOR_DOMINANT_CLASS))
def test_every_wildfire_demo_anchor_is_real_vegetation_not_built_up_or_water(key):
    assert RECORDED_ANCHOR_DOMINANT_CLASS[key] in {"Tree cover", "Shrub cover", "Grass cover", "Crop cover"}


@pytest.mark.parametrize("key", sorted(RECORDED_ANCHOR_DOMINANT_CLASS))
def test_every_anchor_maps_to_a_supported_burnable_spread_fuel(key):
    assert _map_dominant_land_cover(RECORDED_ANCHOR_DOMINANT_CLASS[key]) in BURNABLE | {FC.GENERIC_TREE}


def test_tree_dominant_anchor_uses_the_generic_tree_fuel():
    for key in TREE_DOMINANT_ANCHORS:
        assert _map_dominant_land_cover(RECORDED_ANCHOR_DOMINANT_CLASS[key]) is FC.GENERIC_TREE


def test_old_galilee_point_mapped_to_non_burnable_bare_soil():
    assert _map_dominant_land_cover("Built-up cover") is FC.BARE_SOIL


def test_new_galilee_coordinate_reaches_vegetation_retrieval_with_corrected_composition():
    stored = make_event()
    stored = replace(
        stored,
        event=replace(stored.event, latitude=GALILEE_LOCATION.latitude, longitude=GALILEE_LOCATION.longitude),
    )
    har_kamon = CopernicusLandCoverStatistics(  # corrected 400-pixel result
        cover_fractions=(
            CopernicusCoverFraction("tree", 0.35),
            CopernicusCoverFraction("shrub", 0.30),
            CopernicusCoverFraction("grass", 0.18),
            CopernicusCoverFraction("built_up", 0.08),
            CopernicusCoverFraction("crops", 0.07),
            CopernicusCoverFraction("bare", 0.02),
        ),
        source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        dataset_year=2019,
        radius_km=VEGETATION_RADIUS_KM,
        sample_count=400,
        no_data_count=0,
    )
    service, _, _, _, provider = build_service(stored_event=stored, statistics=har_kamon)

    result = service.prepare_input(stored.id, AS_OF)

    assert provider.calls == [
        {"latitude": GALILEE_LOCATION.latitude, "longitude": GALILEE_LOCATION.longitude, "radius_km": VEGETATION_RADIUS_KM}
    ]
    assert result.vegetation_data.dominant_land_cover == "Tree cover"
    # Area-weighted fuel score of the fixture: .35*.9 + .30*.8 + .18*.7 + .08*.1 + .07*.5 + .02*.1
    assert result.vegetation_data.fuel_score == pytest.approx(0.726, abs=1e-9)  # severity still uses it
    assert _map_dominant_land_cover(result.vegetation_data.dominant_land_cover) is FC.GENERIC_TREE  # Task 14


# --- Scenario generation --------------------------------------------------------


def test_same_seed_reproduces_the_same_locations_and_schedule():
    for seed in (7, 42, 177):
        first, second = build_operations_demo_scenario(seed=seed), build_operations_demo_scenario(seed=seed)
        assert [(i.incident_id, i.location, i.scenario_type) for i in first.incidents] == [
            (i.incident_id, i.location, i.scenario_type) for i in second.incidents
        ]
        assert first.events == second.events


def test_galilee_incidents_use_the_current_canonical_coordinate():
    galilee_incidents = [
        incident
        for seed in range(1, 60)
        for incident in build_operations_demo_scenario(seed=seed).incidents
        if incident.incident_id.startswith("incident-galilee")
    ]
    assert galilee_incidents
    assert all(incident.location is GALILEE_LOCATION for incident in galilee_incidents)


@pytest.mark.parametrize("key", sorted(SIMULATION_LOCATIONS))
def test_news_location_name_matches_the_scenario_area(key):
    report = NewsDataGenerator(seed=3).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, SIMULATION_LOCATIONS[key]).reports[0]
    assert report.location_name == ndg._LOCATION_REPORT_NAMES[key]
    assert SIMULATION_LOCATIONS[key].name.startswith(ENGLISH_AREA_NAMES[key])


def test_frontend_english_place_names_cover_every_simulator_news_location():
    source = (REPO_ROOT / "frontend" / "src" / "components" / "map" / "stationTranslations.ts").read_text(encoding="utf-8")
    block = source.split("export const SIMULATION_PLACE_NAMES", 1)[1].split("};", 1)[0]
    mapping = dict(re.findall(r'"?([֐-׿][֐-׿ ]*)"?:\s*"([^"]+)"', block))
    assert mapping == {ndg._LOCATION_REPORT_NAMES[key]: ENGLISH_AREA_NAMES[key] for key in SIMULATION_LOCATIONS}


@pytest.mark.parametrize("key", sorted(SIMULATION_LOCATIONS))
def test_satellite_hotspots_surround_the_current_anchor(key):
    location = SIMULATION_LOCATIONS[key]
    hotspots = SatelliteDataGenerator(seed=11).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location).hotspots
    assert hotspots
    for hotspot in hotspots:
        distance = _km(location.latitude, location.longitude, hotspot.latitude, hotspot.longitude)
        assert SIMULATED_HOTSPOT_MIN_RADIUS_KM - 0.01 <= distance <= SIMULATED_HOTSPOT_MAX_RADIUS_KM + 0.01


@pytest.mark.parametrize("key", sorted(SIMULATION_LOCATIONS))
def test_simulated_weather_stations_sit_near_the_current_anchor(key):
    location = SIMULATION_LOCATIONS[key]
    stations = WeatherDataGenerator(seed=11).generate(ScenarioType.ACTIVE_FIRE, TIMESTAMP, location).stations
    assert len(stations) == 3
    for station in stations:
        assert _km(location.latitude, location.longitude, station.latitude, station.longitude) < 2.5
