"""Result for simulation-triggered fire-danger assessment coordination."""
from __future__ import annotations

from dataclasses import dataclass

from src.agents.analysis.fire_danger_assessment_result import FireDangerAssessmentResult


@dataclass(frozen=True)
class SimulationFireDangerResult:
    """Whether a simulation event triggered a fire-danger assessment."""

    triggered: bool
    assessment_result: FireDangerAssessmentResult | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}")
        if self.assessment_result is not None and not isinstance(
            self.assessment_result,
            FireDangerAssessmentResult,
        ):
            raise ValueError(
                "assessment_result must be a FireDangerAssessmentResult or None, "
                f"got {self.assessment_result!r}"
            )

        if self.triggered:
            if self.assessment_result is None:
                raise ValueError("triggered simulation fire-danger results must include assessment_result.")
        else:
            if self.assessment_result is not None:
                raise ValueError("non-triggered simulation fire-danger results must not include assessment_result.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation fire-danger results must include reason.")
