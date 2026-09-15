"""The verified PROPAGATOR nominal fuel classes used by wildfire-spread prediction.

These seven classes are transcribed from the official CIMAFoundation
`propagator_sim` reference implementation's default vegetation tables
(`prob_table.txt`, `p_vegetation.txt`), verified in Task 4A and recorded in
`backend/docs/fire_spread_prediction.md` (Scientific Verification Details,
"Verified p_n table"). They are the PROPAGATOR classes themselves, not
EcoGuard's Copernicus `dominant_land_cover` labels -- mapping one to the
other is a Task 5 concern, not this module's.
"""
from __future__ import annotations

from enum import Enum


class FireSpreadFuelClass(Enum):
    """Verified official PROPAGATOR fuel/vegetation class (7-class table)."""

    BROADLEAVES_FIRE_PRONE = "broadleaves_fire_prone"
    SHRUBS = "shrubs"
    BARE_SOIL = "bare_soil"
    GRASSLAND = "grassland"
    CONIFERS_FIRE_PRONE = "conifers_fire_prone"
    AGRO_FORESTRY = "agro_forestry"
    BROADLEAVES_NON_FIRE_PRONE = "broadleaves_non_fire_prone"
