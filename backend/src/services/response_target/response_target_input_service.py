"""Prepare normalized inputs for pure response-target generation.

This service retrieves the active FireEvent plus the latest current severity
and spread-prediction states at a caller-supplied instant. It does not invoke
ResponseTargetCalculator, calculate priority, apply target thresholds,
deduplicate geography, persist targets, or perform routing.
"""
from __future__ import annotations

from datetime import datetime

from src.models import (
    FireEventStatus,
    FireSeverityAssessmentStatus,
    FireSpreadPredictionStatus,
    PredictedRiskTargetCandidate,
    ResponseTargetInput,
    ResponseTargetInputResult,
    ResponseTargetInputStatus,
)
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import (
    FireSpreadPredictionRepository,
    StoredFireSpreadPredictionWithCells,
)

_ACTIVE_EVENT_STATUSES = {FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED}
_INACTIVE_EVENT_STATUSES = {FireEventStatus.RESOLVED, FireEventStatus.DISMISSED}


class ResponseTargetInputService:
    """Gather persisted data needed to call ResponseTargetCalculator later."""

    def __init__(
        self,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        fire_spread_prediction_repository: FireSpreadPredictionRepository | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._fire_spread_prediction_repository = fire_spread_prediction_repository or FireSpreadPredictionRepository()

    def prepare_input(self, fire_event_id: int, as_of: datetime) -> ResponseTargetInputResult:
        """Prepare ResponseTargetInput for one FireEvent at a timezone-aware instant."""
        _validate_fire_event_id(fire_event_id)
        _validate_aware_datetime("as_of", as_of)

        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            raise ValueError(f"FireEvent {fire_event_id!r} was not found.")

        if stored_event.event.status in _INACTIVE_EVENT_STATUSES:
            return ResponseTargetInputResult(
                status=ResponseTargetInputStatus.INACTIVE_EVENT,
                input_data=None,
                fire_event_id=fire_event_id,
            )
        if stored_event.event.status not in _ACTIVE_EVENT_STATUSES:
            raise ValueError(f"Unsupported FireEvent status for response targets: {stored_event.event.status!r}")

        severity_score = self._select_severity_score(fire_event_id, as_of)
        predicted_candidates = self._select_predicted_candidates(fire_event_id, as_of)

        return ResponseTargetInputResult(
            status=ResponseTargetInputStatus.READY,
            input_data=ResponseTargetInput(
                fire_event_id=fire_event_id,
                fire_latitude=stored_event.event.latitude,
                fire_longitude=stored_event.event.longitude,
                severity_score=severity_score,
                predicted_candidates=predicted_candidates,
            ),
            fire_event_id=fire_event_id,
        )

    def _select_severity_score(self, fire_event_id: int, as_of: datetime) -> float | None:
        latest_assessment = self._fire_severity_assessment_repository.get_latest_for_event_as_of(
            fire_event_id,
            as_of,
        )
        if latest_assessment is None:
            return None
        if latest_assessment.assessment.fire_event_id != fire_event_id:
            raise ValueError(
                "severity assessment fire_event_id must match requested fire_event_id, got "
                f"{latest_assessment.assessment.fire_event_id!r} for request {fire_event_id!r}"
            )
        if latest_assessment.assessment.status is FireSeverityAssessmentStatus.VALID:
            return latest_assessment.assessment.score
        return None

    def _select_predicted_candidates(
        self,
        fire_event_id: int,
        as_of: datetime,
    ) -> tuple[PredictedRiskTargetCandidate, ...]:
        candidates: list[PredictedRiskTargetCandidate] = []
        for horizon_minutes in SUPPORTED_HORIZON_MINUTES:
            stored_prediction = self._fire_spread_prediction_repository.get_latest_for_event_and_horizon_as_of(
                fire_event_id=fire_event_id,
                horizon_minutes=horizon_minutes,
                as_of=as_of,
            )
            if stored_prediction is None:
                continue
            if stored_prediction.prediction.fire_event_id != fire_event_id:
                raise ValueError(
                    "spread prediction fire_event_id must match requested fire_event_id, got "
                    f"{stored_prediction.prediction.fire_event_id!r} for request {fire_event_id!r}"
                )
            if stored_prediction.prediction.status is not FireSpreadPredictionStatus.VALID:
                continue
            candidates.extend(_prediction_cells_to_candidates(stored_prediction))

        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.prediction_horizon_minutes,
                    candidate.spread_prediction_id,
                    candidate.spread_prediction_cell_id,
                    candidate.latitude,
                    candidate.longitude,
                ),
            )
        )


def _prediction_cells_to_candidates(
    stored_prediction: StoredFireSpreadPredictionWithCells,
) -> tuple[PredictedRiskTargetCandidate, ...]:
    prediction = stored_prediction.prediction
    return tuple(
        PredictedRiskTargetCandidate(
            fire_event_id=prediction.fire_event_id,
            latitude=stored_cell.cell.latitude,
            longitude=stored_cell.cell.longitude,
            risk_score=stored_cell.cell.spread_risk_score,
            prediction_horizon_minutes=prediction.horizon_minutes,
            spread_prediction_id=stored_prediction.id,
            spread_prediction_cell_id=stored_cell.cell_id,
        )
        for stored_cell in stored_prediction.cells
    )


def _validate_fire_event_id(fire_event_id: object) -> None:
    if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
        raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")


def _validate_aware_datetime(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
