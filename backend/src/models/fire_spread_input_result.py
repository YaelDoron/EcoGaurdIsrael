"""Prepared input and traceability for wildfire-spread prediction."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_spread_input import FireSpreadInput
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_insufficient_data_reason import FireSpreadInsufficientDataReason


@dataclass(frozen=True)
class FireSpreadInputResult:
    """Spread input preparation result with deterministic traceability.

    A READY result carries the calculator-ready `FireSpreadInput` plus the
    ids needed to trace it back to the upstream persisted records it was
    built from. INSUFFICIENT_DATA/INACTIVE_EVENT results never carry
    `input_data`, but may still carry whichever trace ids were already
    resolved before the failure point (e.g. a severity assessment id even
    though weather selection subsequently failed), mirroring
    FireSeverityInputResult's own traceability pattern.

    `insufficient_data_reason` is required for INSUFFICIENT_DATA and must be
    None for READY/INACTIVE_EVENT.
    """

    status: FireSpreadInputStatus
    input_data: FireSpreadInput | None
    fire_event_id: int
    severity_assessment_id: int | None = None
    weather_observation_id: int | None = None
    insufficient_data_reason: FireSpreadInsufficientDataReason | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fire_event_id, bool) or not isinstance(self.fire_event_id, int) or self.fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {self.fire_event_id!r}")
        if not isinstance(self.status, FireSpreadInputStatus):
            raise ValueError(f"status must be a FireSpreadInputStatus, got {self.status!r}")

        self._validate_optional_positive_int("severity_assessment_id", self.severity_assessment_id)
        self._validate_optional_positive_int("weather_observation_id", self.weather_observation_id)

        if self.status is FireSpreadInputStatus.READY:
            if not isinstance(self.input_data, FireSpreadInput):
                raise ValueError("READY spread input result requires input_data.")
            if self.severity_assessment_id is None:
                raise ValueError("READY spread input result requires severity_assessment_id.")
            if self.weather_observation_id is None:
                raise ValueError("READY spread input result requires weather_observation_id.")
        elif self.input_data is not None:
            raise ValueError(f"{self.status.value} spread input result must not include input_data.")

        if self.status is FireSpreadInputStatus.INSUFFICIENT_DATA:
            if not isinstance(self.insufficient_data_reason, FireSpreadInsufficientDataReason):
                raise ValueError(
                    "insufficient_data spread input result requires a FireSpreadInsufficientDataReason, "
                    f"got {self.insufficient_data_reason!r}"
                )
        elif self.insufficient_data_reason is not None:
            raise ValueError(f"{self.status.value} spread input result must not include insufficient_data_reason.")

    @staticmethod
    def _validate_optional_positive_int(field_name: str, value: object) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{field_name} must be a positive integer or None, got {value!r}")
