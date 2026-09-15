"""Vegetation/fuel context derived from local land-cover data."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class VegetationData:
    """Compact EcoGuard vegetation context for severity input preparation."""

    fuel_score: float
    dominant_land_cover: str | None
    source: str
    dataset_year: int
    radius_km: float
    land_cover_distribution: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        self._validate_finite_number("fuel_score", self.fuel_score)
        if not 0 <= self.fuel_score <= 1:
            raise ValueError(f"fuel_score must be within [0, 1], got {self.fuel_score!r}")
        if self.dominant_land_cover is not None and (
            not isinstance(self.dominant_land_cover, str) or not self.dominant_land_cover.strip()
        ):
            raise ValueError(
                "dominant_land_cover must be a non-empty string or None, "
                f"got {self.dominant_land_cover!r}"
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"source must be non-empty, got {self.source!r}")
        if isinstance(self.dataset_year, bool) or not isinstance(self.dataset_year, int) or self.dataset_year <= 0:
            raise ValueError(f"dataset_year must be a positive integer, got {self.dataset_year!r}")
        self._validate_finite_number("radius_km", self.radius_km)
        if self.radius_km <= 0:
            raise ValueError(f"radius_km must be greater than 0, got {self.radius_km!r}")

        distribution = tuple(self.land_cover_distribution)
        for item in distribution:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"land_cover_distribution items must be (label, fraction), got {item!r}")
            label, fraction = item
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"land-cover label must be non-empty, got {label!r}")
            self._validate_finite_number("land-cover fraction", fraction)
            if not 0 <= fraction <= 1:
                raise ValueError(f"land-cover fraction must be within [0, 1], got {fraction!r}")
        object.__setattr__(self, "land_cover_distribution", distribution)

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
