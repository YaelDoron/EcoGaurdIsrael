"""Result object for a FireDangerAssessmentAgent run."""
from __future__ import annotations

from dataclasses import dataclass

from src.models.fire_danger_assessment import FireDangerAssessment


@dataclass(frozen=True)
class FireDangerAssessmentResult:
    """Outcome of one fire-danger assessment orchestration run."""

    assessment: FireDangerAssessment | None
    stored_assessment_id: int | None
    success: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.assessment is not None and not isinstance(self.assessment, FireDangerAssessment):
            raise ValueError(f"assessment must be a FireDangerAssessment or None, got {self.assessment!r}")
        if not isinstance(self.success, bool):
            raise ValueError(f"success must be a bool, got {self.success!r}")

        if self.success:
            if self.assessment is None:
                raise ValueError("successful fire-danger assessment results must include assessment.")
            if (
                isinstance(self.stored_assessment_id, bool)
                or not isinstance(self.stored_assessment_id, int)
                or self.stored_assessment_id <= 0
            ):
                raise ValueError("successful fire-danger assessment results must include stored_assessment_id.")
            if self.error_message is not None:
                raise ValueError("successful fire-danger assessment results must not include error_message.")
        else:
            if self.stored_assessment_id is not None:
                raise ValueError("failed fire-danger assessment results must not include stored_assessment_id.")
            if not isinstance(self.error_message, str) or not self.error_message.strip():
                raise ValueError("failed fire-danger assessment results must include error_message.")
