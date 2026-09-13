"""Result of preparing weather inputs for fire-danger calculation."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_danger_input import FireDangerInput
from src.models.fire_danger_input_status import FireDangerInputStatus


@dataclass(frozen=True)
class FireDangerInputResult:
    """Prepared FFWI input plus traceability metadata for source observations."""

    status: FireDangerInputStatus
    input_data: FireDangerInput | None
    observation_ids: tuple[int, ...]
    station_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, FireDangerInputStatus):
            raise ValueError(f"status must be a FireDangerInputStatus, got {self.status!r}")
        if self.input_data is not None and not isinstance(self.input_data, FireDangerInput):
            raise ValueError(f"input_data must be a FireDangerInput or None, got {self.input_data!r}")

        object.__setattr__(self, "observation_ids", tuple(self.observation_ids))
        object.__setattr__(self, "station_ids", tuple(self.station_ids))

        if self.status is FireDangerInputStatus.READY:
            if self.input_data is None:
                raise ValueError("READY fire-danger input results must include input_data.")
            if not self.observation_ids:
                raise ValueError("READY fire-danger input results must include observation_ids.")
            if not self.station_ids:
                raise ValueError("READY fire-danger input results must include station_ids.")
        elif self.status is FireDangerInputStatus.INSUFFICIENT_DATA and self.input_data is not None:
            raise ValueError("INSUFFICIENT_DATA fire-danger input results must not include input_data.")
