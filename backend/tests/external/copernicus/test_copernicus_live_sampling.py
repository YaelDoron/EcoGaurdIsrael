"""LIVE Copernicus verification of the corrected sampling (opt-in only).

Skipped unless ECOGUARD_LIVE_COPERNICUS=1 (and credentials are configured), so
ordinary test runs never call the external API. Run with:
    ECOGUARD_LIVE_COPERNICUS=1 python -m pytest -m integration tests/external/copernicus/test_copernicus_live_sampling.py
"""
from __future__ import annotations

import os
import time

import pytest

from src.external.copernicus import CopernicusLandCoverClient
from src.mappers.vegetation_mapper import VegetationMapper
from src.services.fire_severity.fire_severity_input_config import VEGETATION_RADIUS_KM
from src.simulation import SIMULATION_LOCATIONS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("ECOGUARD_LIVE_COPERNICUS") != "1", reason="live Copernicus test (opt-in)"),
]


@pytest.fixture(scope="module")
def client() -> CopernicusLandCoverClient:
    return CopernicusLandCoverClient()


@pytest.mark.parametrize("key", sorted(SIMULATION_LOCATIONS))
def test_live_box_is_a_400_pixel_grid_with_complete_class_fractions(client, key):
    time.sleep(1.0)  # stay under the Statistical API's burst rate limit
    location = SIMULATION_LOCATIONS[key]
    stats = client.get_land_cover_statistics(location.latitude, location.longitude, VEGETATION_RADIUS_KM)

    assert stats is not None
    assert stats.sample_count == 400  # formerly 1: resx/resy=100 meant 100 degrees in EPSG:4326
    assert stats.valid_pixel_count > 0
    # Per-pixel fractional covers sum to ~100 %, so the valid-pixel band means sum to ~1
    # (formerly 0.60-0.88 for a single resampled pixel, hidden by renormalisation).
    assert sum(f.fraction for f in stats.cover_fractions) == pytest.approx(1.0, abs=0.03)
    assert VegetationMapper().map_statistics(stats) is not None


def test_live_nearby_boxes_are_spatially_distinguishable(client):
    time.sleep(1.0)
    carmel = SIMULATION_LOCATIONS["carmel"]
    base = client.get_land_cover_statistics(carmel.latitude, carmel.longitude, VEGETATION_RADIUS_KM)
    time.sleep(1.0)
    shifted = client.get_land_cover_statistics(carmel.latitude + 500 / 111_320, carmel.longitude, VEGETATION_RADIUS_KM)
    assert {f.category: round(f.fraction, 3) for f in base.cover_fractions} != {
        f.category: round(f.fraction, 3) for f in shifted.cover_fractions
    }
