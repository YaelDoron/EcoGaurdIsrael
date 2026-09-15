"""Result of a pure wildfire-spread cellular-automata calculation.

This is the calculator's own pure output: a set of predicted cells (possibly
empty -- a valid "no predicted spread" result) plus the horizon and
methodology identity. It intentionally does not know about FireEvent,
FireSeverityAssessment, persistence status, or any database id -- combining
this with those belongs to a later agent task, which builds the persisted
FireSpreadPrediction domain object from this result.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_spread_prediction import (
    CA_TIME_STEP_MINUTES,
    SUPPORTED_HORIZON_MINUTES,
    FireSpreadPredictionCell,
)


@dataclass(frozen=True)
class FireSpreadCalculation:
    """Deterministic spread-prediction cells and methodology identity.

    `cells` may legitimately be empty: a scientifically valid calculation can
    conclude that no neighboring cell crosses the deterministic propagation
    threshold within the selected horizon.
    """

    cells: tuple[FireSpreadPredictionCell, ...]
    horizon_minutes: int
    methodology: str
    methodology_version: str

    def __post_init__(self) -> None:
        if isinstance(self.horizon_minutes, bool) or not isinstance(self.horizon_minutes, int):
            raise ValueError(f"horizon_minutes must be an integer, got {self.horizon_minutes!r}")
        if self.horizon_minutes not in SUPPORTED_HORIZON_MINUTES:
            raise ValueError(
                f"horizon_minutes must be one of {SUPPORTED_HORIZON_MINUTES}, got {self.horizon_minutes!r}"
            )

        self._validate_non_empty_string("methodology", self.methodology)
        self._validate_non_empty_string("methodology_version", self.methodology_version)

        if not isinstance(self.cells, tuple) or not all(
            isinstance(cell, FireSpreadPredictionCell) for cell in self.cells
        ):
            raise ValueError(f"cells must be a tuple of FireSpreadPredictionCell, got {self.cells!r}")

        max_step = self.horizon_minutes // CA_TIME_STEP_MINUTES
        for cell in self.cells:
            if cell.reached_step > max_step:
                raise ValueError(
                    f"cell reached_step {cell.reached_step!r} exceeds the horizon's "
                    f"{max_step!r} CA steps for horizon_minutes={self.horizon_minutes!r}"
                )

        ordering_keys = [(cell.reached_step, cell.latitude, cell.longitude) for cell in self.cells]
        if ordering_keys != sorted(ordering_keys):
            raise ValueError("cells must be ordered by (reached_step, latitude, longitude)")

    @staticmethod
    def _validate_non_empty_string(field_name: str, value: object) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
