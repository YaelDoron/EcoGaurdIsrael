"""Map Copernicus Global Land Cover fractions to EcoGuard vegetation context.

The fractional-cover categories are from the Copernicus Global Land Cover
100 m yearly V3 product. EcoGuard fuel scores are application-level context
weights, not official Copernicus fire-risk values.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.external.copernicus import CopernicusLandCoverStatistics
from src.models.vegetation_data import VegetationData

VEGETATION_SOURCE = "COPERNICUS_GLOBAL_LAND_COVER_100M_API"
VEGETATION_DATASET_YEAR = 2019


@dataclass(frozen=True)
class VegetationCategoryMapping:
    """EcoGuard mapping metadata for one Copernicus fractional-cover category."""

    label: str
    fuel_score: float


FOREST_FUEL_SCORE = 0.90
SHRUBLAND_FUEL_SCORE = 0.80
GRASSLAND_FUEL_SCORE = 0.70
CROPLAND_FUEL_SCORE = 0.50
BARE_FUEL_SCORE = 0.10
BUILT_UP_FUEL_SCORE = 0.10
MOSS_LICHEN_FUEL_SCORE = 0.10
SNOW_FUEL_SCORE = 0.00
WATER_FUEL_SCORE = 0.00

COPERNICUS_FUEL_MAPPINGS: dict[str, VegetationCategoryMapping] = {
    "tree": VegetationCategoryMapping("Tree cover", FOREST_FUEL_SCORE),
    "shrub": VegetationCategoryMapping("Shrub cover", SHRUBLAND_FUEL_SCORE),
    "grass": VegetationCategoryMapping("Grass cover", GRASSLAND_FUEL_SCORE),
    "crops": VegetationCategoryMapping("Crop cover", CROPLAND_FUEL_SCORE),
    "bare": VegetationCategoryMapping("Bare cover", BARE_FUEL_SCORE),
    "built_up": VegetationCategoryMapping("Built-up cover", BUILT_UP_FUEL_SCORE),
    "permanent_water": VegetationCategoryMapping("Permanent water cover", WATER_FUEL_SCORE),
    "seasonal_water": VegetationCategoryMapping("Seasonal water cover", WATER_FUEL_SCORE),
    "moss_lichen": VegetationCategoryMapping("Moss and lichen cover", MOSS_LICHEN_FUEL_SCORE),
    "snow": VegetationCategoryMapping("Snow cover", SNOW_FUEL_SCORE),
}


class VegetationMapper:
    """Convert land-cover class fractions into EcoGuard vegetation data."""

    def map_statistics(
        self,
        statistics: CopernicusLandCoverStatistics | None,
    ) -> VegetationData | None:
        """Return area-weighted vegetation data, or None if no mapped classes remain."""
        if statistics is None:
            return None
        known_fractions = []
        known_total_fraction = 0.0
        for item in statistics.cover_fractions:
            mapping = COPERNICUS_FUEL_MAPPINGS.get(item.category)
            if mapping is None:
                continue
            known_fractions.append((item, mapping))
            known_total_fraction += item.fraction

        if known_total_fraction <= 0:
            return None

        normalized_distribution = tuple(
            (
                mapping.label,
                item.fraction / known_total_fraction,
            )
            for item, mapping in sorted(
                known_fractions,
                key=lambda pair: (pair[1].label, pair[0].category),
            )
        )
        fuel_score = sum(
            (item.fraction / known_total_fraction) * mapping.fuel_score
            for item, mapping in known_fractions
        )
        dominant = min(
            (
                (-(item.fraction / known_total_fraction), mapping.label, item.category)
                for item, mapping in known_fractions
            ),
            key=lambda value: value,
        )[1]

        return VegetationData(
            fuel_score=fuel_score,
            dominant_land_cover=dominant,
            source=statistics.source,
            dataset_year=statistics.dataset_year,
            radius_km=statistics.radius_km,
            land_cover_distribution=normalized_distribution,
        )
