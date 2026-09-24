"""Result for simulation-triggered active fire detection coordination."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.analysis.fire_detection_result import FireDetectionResult


@dataclass(frozen=True)
class SimulationFireDetectionResult:
    """Whether a simulation event triggered active wildfire detection."""

    triggered: bool
    detection_result: FireDetectionResult | None
    reason: str | None = None
    # Task 9A: which of the detection cycle's FireEvents are RESPONSE-ELIGIBLE (CONFIRMED) right now. None = not
    # determined (no repository was wired, e.g. in unit tests); an empty tuple = none of them are.
    response_eligible_event_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}")
        if self.detection_result is not None and not isinstance(self.detection_result, FireDetectionResult):
            raise ValueError(
                "detection_result must be a FireDetectionResult or None, "
                f"got {self.detection_result!r}"
            )

        if self.triggered:
            if self.detection_result is None:
                raise ValueError("triggered simulation fire-detection results must include detection_result.")
        else:
            if self.detection_result is not None:
                raise ValueError("non-triggered simulation fire-detection results must not include detection_result.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation fire-detection results must include reason.")
