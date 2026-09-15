"""Optional live integration test for Copernicus land-cover statistics.

Skipped unless COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are configured.
The test does not print credentials or access tokens.
"""
from __future__ import annotations

import pytest

from src.config.settings import settings
from src.external.copernicus import CopernicusLandCoverClient
from src.mappers.vegetation_mapper import VegetationMapper
from src.services.fire_severity.fire_severity_input_config import VEGETATION_RADIUS_KM
from src.simulation.simulation_locations import DEFAULT_CARMEL_LOCATION


pytestmark = pytest.mark.integration


def test_copernicus_land_cover_statistics_for_carmel_demo_location():
    if not settings.COPERNICUS_CLIENT_ID or not settings.COPERNICUS_CLIENT_SECRET:
        pytest.skip("Copernicus credentials are not configured; skipping live land-cover integration test.")

    statistics = CopernicusLandCoverClient().get_land_cover_statistics(
        latitude=DEFAULT_CARMEL_LOCATION.latitude,
        longitude=DEFAULT_CARMEL_LOCATION.longitude,
        radius_km=VEGETATION_RADIUS_KM,
    )
    vegetation_data = VegetationMapper().map_statistics(statistics)

    assert vegetation_data is not None
    assert 0 <= vegetation_data.fuel_score <= 1
    assert vegetation_data.dataset_year == 2019
