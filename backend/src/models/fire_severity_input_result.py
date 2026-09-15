"""Prepared input and traceability for active wildfire severity."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_severity_input import FireSeverityInput
from src.models.fire_severity_input_status import FireSeverityInputStatus
from src.models.vegetation_data import VegetationData


@dataclass(frozen=True)
class FireSeverityInputResult:
    """Severity input preparation result with deterministic traceability."""

    status: FireSeverityInputStatus
    input_data: FireSeverityInput | None
    fire_event_id: int
    weather_observation_ids: tuple[int, ...] = ()
    satellite_hotspot_ids: tuple[int, ...] = ()
    selected_frp_hotspot_id: int | None = None
    vegetation_data: VegetationData | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, FireSeverityInputStatus):
            raise ValueError(f"status must be a FireSeverityInputStatus, got {self.status!r}")
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")

        weather_ids = self._normalize_positive_ids("weather_observation_ids", self.weather_observation_ids)
        hotspot_ids = self._normalize_positive_ids("satellite_hotspot_ids", self.satellite_hotspot_ids)
        object.__setattr__(self, "weather_observation_ids", weather_ids)
        object.__setattr__(self, "satellite_hotspot_ids", hotspot_ids)

        if self.selected_frp_hotspot_id is not None and (
            isinstance(self.selected_frp_hotspot_id, bool)
            or not isinstance(self.selected_frp_hotspot_id, int)
            or self.selected_frp_hotspot_id <= 0
        ):
            raise ValueError(
                "selected_frp_hotspot_id must be a positive integer or None, "
                f"got {self.selected_frp_hotspot_id!r}"
            )
        if self.vegetation_data is not None and not isinstance(self.vegetation_data, VegetationData):
            raise ValueError(f"vegetation_data must be VegetationData or None, got {self.vegetation_data!r}")

        if self.status is FireSeverityInputStatus.READY:
            if not isinstance(self.input_data, FireSeverityInput):
                raise ValueError("READY severity input result requires input_data.")
            if not weather_ids:
                raise ValueError("READY severity input result requires weather observation traceability.")
            if not hotspot_ids:
                raise ValueError("READY severity input result requires satellite hotspot traceability.")
            if self.selected_frp_hotspot_id is None:
                raise ValueError("READY severity input result requires selected_frp_hotspot_id.")
            if self.selected_frp_hotspot_id not in hotspot_ids:
                raise ValueError("selected_frp_hotspot_id must be included in satellite_hotspot_ids.")
        elif self.input_data is not None:
            raise ValueError(f"{self.status.value} severity input result must not include input_data.")

    @staticmethod
    def _normalize_positive_ids(field_name: str, values: tuple[int, ...]) -> tuple[int, ...]:
        try:
            ids = tuple(values)
        except TypeError as exc:
            raise ValueError(f"{field_name} must be iterable.") from exc
        for value in ids:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must contain positive integers, got {value!r}")
        return tuple(sorted(ids))
