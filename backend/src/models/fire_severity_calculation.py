"""Result of a pure active wildfire severity calculation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from src.models.fire_severity_level import FireSeverityLevel


@dataclass(frozen=True)
class FireSeverityCalculation:
    """EcoGuard operational severity score and classification.

    The score is deterministic for identical inputs. It is an application-level
    operational score, not a scientifically calibrated probability or official
    emergency-authority threshold.
    """

    score: float
    level: FireSeverityLevel
    frp_factor: float
    wind_factor: float
    dryness_factor: float
    vegetation_factor: float | None
    available_weight: float
    methodology: str
    methodology_version: str

    def __post_init__(self) -> None:
        self._validate_finite_number("score", self.score)
        if not 0 <= self.score <= 100:
            raise ValueError(f"score must be within [0, 100], got {self.score!r}")
        if not isinstance(self.level, FireSeverityLevel):
            raise ValueError(f"level must be a FireSeverityLevel, got {self.level!r}")

        for field_name in ("frp_factor", "wind_factor", "dryness_factor"):
            value = getattr(self, field_name)
            self._validate_finite_number(field_name, value)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be within [0, 1], got {value!r}")

        if self.vegetation_factor is not None:
            self._validate_finite_number("vegetation_factor", self.vegetation_factor)
            if not 0 <= self.vegetation_factor <= 1:
                raise ValueError(
                    "vegetation_factor must be within [0, 1], "
                    f"got {self.vegetation_factor!r}"
                )

        self._validate_finite_number("available_weight", self.available_weight)
        if self.available_weight <= 0:
            raise ValueError(f"available_weight must be positive, got {self.available_weight!r}")

        if not isinstance(self.methodology, str) or not self.methodology.strip():
            raise ValueError(f"methodology must be non-empty, got {self.methodology!r}")
        if not isinstance(self.methodology_version, str) or not self.methodology_version.strip():
            raise ValueError(
                "methodology_version must be non-empty, "
                f"got {self.methodology_version!r}"
            )

    @staticmethod
    def _validate_finite_number(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name} must be a finite number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
