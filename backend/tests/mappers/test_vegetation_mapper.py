"""Tests for Copernicus fractional-cover to EcoGuard vegetation mapping."""
from __future__ import annotations

import pytest

from src.external.copernicus import CopernicusCoverFraction, CopernicusLandCoverStatistics
from src.mappers.vegetation_mapper import (
    BARE_FUEL_SCORE,
    BUILT_UP_FUEL_SCORE,
    FOREST_FUEL_SCORE,
    GRASSLAND_FUEL_SCORE,
    CROPLAND_FUEL_SCORE,
    SHRUBLAND_FUEL_SCORE,
    VEGETATION_DATASET_YEAR,
    VEGETATION_SOURCE,
    WATER_FUEL_SCORE,
    VegetationMapper,
)


def stats(*fractions: tuple[str, float]) -> CopernicusLandCoverStatistics:
    return CopernicusLandCoverStatistics(
        cover_fractions=tuple(
            CopernicusCoverFraction(category=category, fraction=fraction)
            for category, fraction in fractions
        ),
        source=VEGETATION_SOURCE,
        dataset_year=VEGETATION_DATASET_YEAR,
        radius_km=1.0,
    )


@pytest.mark.parametrize(
    ("category", "expected_label", "expected_fuel"),
    [
        ("tree", "Tree cover", FOREST_FUEL_SCORE),
        ("shrub", "Shrub cover", SHRUBLAND_FUEL_SCORE),
        ("grass", "Grass cover", GRASSLAND_FUEL_SCORE),
        ("crops", "Crop cover", CROPLAND_FUEL_SCORE),
        ("bare", "Bare cover", BARE_FUEL_SCORE),
        ("built_up", "Built-up cover", BUILT_UP_FUEL_SCORE),
        ("permanent_water", "Permanent water cover", WATER_FUEL_SCORE),
        ("seasonal_water", "Seasonal water cover", WATER_FUEL_SCORE),
    ],
)
def test_supported_copernicus_categories_map_to_expected_fuel_scores(
    category,
    expected_label,
    expected_fuel,
):
    data = VegetationMapper().map_statistics(stats((category, 1.0)))

    assert data.fuel_score == pytest.approx(expected_fuel)
    assert data.dominant_land_cover == expected_label
    assert data.source == VEGETATION_SOURCE
    assert data.dataset_year == VEGETATION_DATASET_YEAR


def test_mixed_classes_are_area_weighted():
    data = VegetationMapper().map_statistics(stats(("tree", 0.60), ("grass", 0.25), ("built_up", 0.15)))

    expected = 0.60 * FOREST_FUEL_SCORE + 0.25 * GRASSLAND_FUEL_SCORE + 0.15 * BUILT_UP_FUEL_SCORE
    assert data.fuel_score == pytest.approx(expected)
    assert data.dominant_land_cover == "Tree cover"


def test_unknown_classes_are_ignored_and_known_classes_are_renormalized():
    data = VegetationMapper().map_statistics(stats(("tree", 0.25), ("unknown", 0.75)))

    assert data.fuel_score == pytest.approx(FOREST_FUEL_SCORE)
    assert data.land_cover_distribution == (("Tree cover", 1.0),)


def test_all_unknown_classes_return_unavailable():
    assert VegetationMapper().map_statistics(stats(("unknown", 1.0))) is None


def test_dominant_class_tie_breaks_deterministically_by_label_then_code():
    data = VegetationMapper().map_statistics(stats(("grass", 0.50), ("shrub", 0.50)))

    assert data.dominant_land_cover == "Grass cover"


def test_missing_statistics_return_unavailable():
    assert VegetationMapper().map_statistics(None) is None
