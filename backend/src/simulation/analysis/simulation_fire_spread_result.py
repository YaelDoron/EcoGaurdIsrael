"""Result for simulation-triggered wildfire-spread prediction coordination."""
from __future__ import annotations

from dataclasses import dataclass

from src.repositories.fire_spread_prediction_repository import StoredFireSpreadPrediction


@dataclass(frozen=True)
class SimulationFireSpreadResult:
    """Whether a simulation event triggered wildfire-spread predictions."""

    triggered: bool
    prediction_results: tuple[StoredFireSpreadPrediction, ...] = ()
    failed_fire_event_ids: tuple[int, ...] = ()
    error_messages: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}")

        prediction_results = tuple(self.prediction_results)
        for prediction_result in prediction_results:
            if not isinstance(prediction_result, StoredFireSpreadPrediction):
                raise ValueError(
                    "prediction_results must contain StoredFireSpreadPrediction items, "
                    f"got {prediction_result!r}"
                )
        object.__setattr__(self, "prediction_results", prediction_results)

        failed_ids = tuple(sorted(set(self.failed_fire_event_ids)))
        for fire_event_id in failed_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(
                    "failed_fire_event_ids must contain positive integer ids, "
                    f"got {fire_event_id!r}"
                )
        object.__setattr__(self, "failed_fire_event_ids", failed_ids)

        error_messages = tuple(self.error_messages)
        for error_message in error_messages:
            if not isinstance(error_message, str) or not error_message.strip():
                raise ValueError(f"error_messages must contain non-empty strings, got {error_message!r}")
        object.__setattr__(self, "error_messages", error_messages)

        if self.triggered:
            if self.reason is not None:
                raise ValueError("triggered simulation fire-spread results must not include reason.")
            if not prediction_results and not failed_ids:
                raise ValueError("triggered simulation fire-spread results must include a success or failure.")
            if failed_ids and not error_messages:
                raise ValueError("failed fire-spread triggers must include error_messages.")
        else:
            if prediction_results:
                raise ValueError("non-triggered simulation fire-spread results must not include predictions.")
            if failed_ids:
                raise ValueError("non-triggered simulation fire-spread results must not include failed ids.")
            if error_messages:
                raise ValueError("non-triggered simulation fire-spread results must not include errors.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation fire-spread results must include reason.")
