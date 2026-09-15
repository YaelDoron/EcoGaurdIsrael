"""Result for simulation-triggered active fire severity assessment coordination."""
from __future__ import annotations

from dataclasses import dataclass

from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment


@dataclass(frozen=True)
class SimulationFireSeverityResult:
    """Whether a simulation event triggered fire-severity assessments."""

    triggered: bool
    assessment_results: tuple[StoredFireSeverityAssessment, ...] = ()
    failed_fire_event_ids: tuple[int, ...] = ()
    error_messages: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool):
            raise ValueError(f"triggered must be a bool, got {self.triggered!r}")

        assessment_results = tuple(self.assessment_results)
        for assessment_result in assessment_results:
            if not isinstance(assessment_result, StoredFireSeverityAssessment):
                raise ValueError(
                    "assessment_results must contain StoredFireSeverityAssessment items, "
                    f"got {assessment_result!r}"
                )
        object.__setattr__(self, "assessment_results", assessment_results)

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
                raise ValueError("triggered simulation fire-severity results must not include reason.")
            if not assessment_results and not failed_ids:
                raise ValueError("triggered simulation fire-severity results must include a success or failure.")
            if failed_ids and not error_messages:
                raise ValueError("failed fire-severity triggers must include error_messages.")
        else:
            if assessment_results:
                raise ValueError("non-triggered simulation fire-severity results must not include assessments.")
            if failed_ids:
                raise ValueError("non-triggered simulation fire-severity results must not include failed ids.")
            if error_messages:
                raise ValueError("non-triggered simulation fire-severity results must not include errors.")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("non-triggered simulation fire-severity results must include reason.")
