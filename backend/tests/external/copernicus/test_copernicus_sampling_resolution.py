"""Sampling-resolution regressions for the Copernicus land-cover statistics request.

Root cause (Task 13): the request sent `resx`/`resy = 100` with bounds in
EPSG:4326. The Sentinel Hub Statistical API defines resx/resy in the units of
`input.bounds.properties.crs` (official OpenAPI, StatisticalRequestAggregation),
i.e. 100 DEGREES per pixel here, so the whole 2 km x 2 km box was ONE resampled
pixel (live response: sampleCount = 1). The request now asks for an explicit
pixel grid (width/height) at the product's native ~100 m resolution.
"""
from __future__ import annotations

import math

import pytest
import requests

from src.external.copernicus import CopernicusLandCoverClient
from src.external.copernicus.copernicus_land_cover_client import (
    COPERNICUS_SAMPLING_VERSION,
    TARGET_SAMPLE_RESOLUTION_M,
    _parse_statistics_payload,
)
from src.mappers.vegetation_mapper import VegetationMapper

CATEGORIES = [
    "tree", "shrub", "grass", "crops", "bare", "built_up",
    "permanent_water", "seasonal_water", "moss_lichen", "snow",
]


def _client() -> CopernicusLandCoverClient:
    return CopernicusLandCoverClient(
        client_id="id", client_secret="secret", token_url="https://token.test",
        statistics_url="https://stats.test", collection_id="collection", timeout=5,
    )


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_008.8 * math.asin(math.sqrt(a))


def _box(request):
    ring = request["input"]["bounds"]["geometry"]["coordinates"][0]
    west, south = ring[0]
    east, north = ring[2]
    return west, south, east, north


def payload(sample_count=400, no_data_count=0, **means):
    bands = {
        f"B{i}": {"stats": {"mean": means.get(c, 0.0), "sampleCount": sample_count, "noDataCount": no_data_count}}
        for i, c in enumerate(CATEGORIES)
    }
    return {"data": [{"outputs": {"vegetation": {"bands": bands}}}]}


# --- Request correctness -------------------------------------------------------


def test_request_uses_an_explicit_pixel_grid_not_crs_unit_resolution():
    aggregation = _client()._build_statistics_request(31.774, 35.139, 1.0)["aggregation"]
    assert "resx" not in aggregation and "resy" not in aggregation
    assert (aggregation["width"], aggregation["height"]) == (20, 20)


def test_bounds_stay_in_epsg_4326():
    request = _client()._build_statistics_request(31.774, 35.139, 1.0)
    assert request["input"]["bounds"]["properties"]["crs"].endswith("/EPSG/0/4326")


@pytest.mark.parametrize("latitude", [29.55, 31.774, 33.085])
def test_one_km_radius_box_is_two_km_square_sampled_at_about_100_m(latitude):
    request = _client()._build_statistics_request(latitude, 35.1, 1.0)
    west, south, east, north = _box(request)
    width_m = _haversine_m(latitude, west, latitude, east)
    height_m = _haversine_m(south, 35.1, north, 35.1)
    assert width_m == pytest.approx(2000.0, rel=0.01)  # longitude delta scaled by cos(latitude)
    assert height_m == pytest.approx(2000.0, rel=0.01)
    aggregation = request["aggregation"]
    assert width_m / aggregation["width"] == pytest.approx(TARGET_SAMPLE_RESOLUTION_M, rel=0.05)
    assert height_m / aggregation["height"] == pytest.approx(TARGET_SAMPLE_RESOLUTION_M, rel=0.05)


def test_pixel_grid_scales_with_the_requested_radius():
    aggregation = _client()._build_statistics_request(31.774, 35.139, 0.5)["aggregation"]
    assert (aggregation["width"], aggregation["height"]) == (10, 10)


def test_neighbouring_requests_are_not_collapsed_by_request_construction():
    client = _client()
    base = _box(client._build_statistics_request(31.774, 35.139, 1.0))
    shifted = _box(client._build_statistics_request(31.774 + 300 / 111_320, 35.139, 1.0))
    # A 300 m shift moves the box by ~3 pixel rows at 100 m sampling (formerly: same single pixel).
    assert (shifted[3] - base[3]) * 111_320 / TARGET_SAMPLE_RESOLUTION_M == pytest.approx(3.0, rel=0.05)


# --- Aggregation ---------------------------------------------------------------------


def test_pixel_counts_are_parsed_and_the_denominator_is_valid_pixels():
    stats = _parse_statistics_payload(
        payload(sample_count=400, no_data_count=40, shrub=0.5, grass=0.3, tree=0.2), radius_km=1.0
    )
    assert (stats.sample_count, stats.no_data_count, stats.valid_pixel_count) == (400, 40, 360)
    assert sum(f.fraction for f in stats.cover_fractions) == pytest.approx(1.0)


def test_all_no_data_box_is_no_usable_data():
    assert _parse_statistics_payload(payload(sample_count=400, no_data_count=400, shrub=0.4), radius_km=1.0) is None


def test_counts_are_optional_for_older_payloads():
    stats = _parse_statistics_payload({"data": [{"outputs": {"vegetation": {"bands": {"B0": {"stats": {"mean": 0.5}}}}}}]}, 1.0)
    assert stats.sample_count is None and stats.valid_pixel_count is None


def test_percentages_sum_to_one_and_dominant_class_is_the_largest_share():
    stats = _parse_statistics_payload(
        payload(tree=0.27, shrub=0.32, grass=0.28, crops=0.04, bare=0.03, built_up=0.06), radius_km=1.0
    )
    vegetation = VegetationMapper().map_statistics(stats)
    assert sum(share for _, share in vegetation.land_cover_distribution) == pytest.approx(1.0)
    assert vegetation.dominant_land_cover == "Shrub cover"


# --- Cache ------------------------------------------------------------------------


def test_cache_key_carries_the_sampling_version(monkeypatch):
    client = _client()
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if url == "https://token.test":
            class T:
                status_code = 200
                def json(self):
                    return {"access_token": "t", "expires_in": 3600}
            return T()

        class S:
            status_code = 200
            def json(self):
                return payload(shrub=1.0)
        return S()

    monkeypatch.setattr(requests, "post", fake_post)
    client.get_land_cover_statistics(31.774, 35.139, 1.0)
    assert any(COPERNICUS_SAMPLING_VERSION in key for key in client._statistics_cache)
