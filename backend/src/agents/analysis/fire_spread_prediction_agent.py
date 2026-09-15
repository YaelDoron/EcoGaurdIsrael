"""Orchestrates wildfire-spread prediction from prepared input to persistence.

Thin orchestration only: input retrieval, vegetation mapping, geographic grid
generation, and the PROPAGATOR scientific formulas all already live in
FireSpreadInputService and FireSpreadCalculator (Tasks 4B/5). This agent does
not contain scientific logic, weather queries, or direct SQLAlchemy code.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.calculators.fire_spread.fire_spread_calculator import FireSpreadCalculator
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.models.fire_spread_effective_state import FireSpreadEffectiveState, validate_effective_state_fingerprint
from src.models.fire_spread_input_result import FireSpreadInputResult
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_prediction import FireSpreadPrediction
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.repositories.fire_spread_prediction_repository import (
    FireSpreadPredictionRepository,
    StoredFireSpreadPrediction,
)
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService

logger = logging.getLogger(__name__)


class FireSpreadPredictionAgent:
    """Coordinates spread input preparation, CA calculation, and prediction persistence."""

    def __init__(
        self,
        input_service: FireSpreadInputService,
        calculator: FireSpreadCalculator,
        repository: FireSpreadPredictionRepository,
    ) -> None:
        self._input_service = input_service
        self._calculator = calculator
        self._repository = repository

    def predict(
        self,
        fire_event_id: int,
        as_of: datetime,
        horizon_minutes: int,
    ) -> StoredFireSpreadPrediction:
        """Predict wildfire spread for one FireEvent at a timezone-aware instant.

        `predicted_at` is set to `as_of` exactly -- this agent never calls
        `datetime.now()`, preserving deterministic traceability.
        """
        self._validate_request(fire_event_id, as_of)

        input_result = self._input_service.prepare_input(
            fire_event_id=fire_event_id,
            as_of=as_of,
            horizon_minutes=horizon_minutes,
        )
        return self.predict_from_input_result(
            input_result=input_result,
            as_of=as_of,
            horizon_minutes=horizon_minutes,
        )

    def predict_from_input_result(
        self,
        *,
        input_result: FireSpreadInputResult,
        as_of: datetime,
        horizon_minutes: int,
        effective_state_fingerprint: str | None = None,
    ) -> StoredFireSpreadPrediction:
        """Persist one prediction from an already prepared spread input result.

        Refresh orchestration uses this path to avoid preparing the same input
        twice. Existing direct callers keep using predict(...), which now
        delegates here after one input-service call.
        """
        self._validate_prepared_request(input_result, as_of, horizon_minutes, effective_state_fingerprint)
        logger.info(
            "Prepared fire-spread input for FireEvent %s with status %s",
            input_result.fire_event_id,
            input_result.status.value,
        )

        if input_result.status is FireSpreadInputStatus.READY:
            self._validate_ready_input_result(input_result)
            if effective_state_fingerprint is None:
                effective_state_fingerprint = FireSpreadEffectiveState.from_input(
                    fire_event_id=input_result.fire_event_id,
                    spread_input=input_result.input_data,
                ).fingerprint
            prediction = self._build_valid_prediction(
                input_result,
                as_of,
                horizon_minutes,
                effective_state_fingerprint,
            )
        elif input_result.status is FireSpreadInputStatus.INSUFFICIENT_DATA:
            prediction = self._build_not_calculated_prediction(
                input_result=input_result,
                as_of=as_of,
                horizon_minutes=horizon_minutes,
                status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            )
        elif input_result.status is FireSpreadInputStatus.INACTIVE_EVENT:
            prediction = self._build_not_calculated_prediction(
                input_result=input_result,
                as_of=as_of,
                horizon_minutes=horizon_minutes,
                status=FireSpreadPredictionStatus.INACTIVE_EVENT,
            )
        else:
            raise ValueError(f"Unsupported fire-spread input status: {input_result.status!r}")

        stored = self._repository.save_prediction(
            prediction=prediction,
            weather_observation_id=input_result.weather_observation_id,
        )
        logger.info(
            "Stored fire-spread prediction %s for FireEvent %s with status %s",
            stored.id,
            input_result.fire_event_id,
            stored.prediction.status.value,
        )
        return stored

    def _build_valid_prediction(
        self,
        input_result: FireSpreadInputResult,
        as_of: datetime,
        horizon_minutes: int,
        effective_state_fingerprint: str,
    ) -> FireSpreadPrediction:
        self._validate_ready_input_result(input_result)

        # The pure CA calculation happens here, and only here -- the agent
        # does not touch p_n/e_m/alpha_wh or grid generation itself.
        calculation = self._calculator.calculate(input_result.input_data)

        return FireSpreadPrediction(
            fire_event_id=input_result.fire_event_id,
            severity_assessment_id=input_result.severity_assessment_id,
            predicted_at=as_of,
            horizon_minutes=horizon_minutes,
            status=FireSpreadPredictionStatus.VALID,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=calculation.cells,
            effective_state_fingerprint=effective_state_fingerprint,
        )

    @staticmethod
    def _build_not_calculated_prediction(
        input_result: FireSpreadInputResult,
        as_of: datetime,
        horizon_minutes: int,
        status: FireSpreadPredictionStatus,
    ) -> FireSpreadPrediction:
        return FireSpreadPrediction(
            fire_event_id=input_result.fire_event_id,
            severity_assessment_id=input_result.severity_assessment_id,
            predicted_at=as_of,
            horizon_minutes=horizon_minutes,
            status=status,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=(),
        )

    @staticmethod
    def _validate_request(fire_event_id: int, as_of: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")

    @staticmethod
    def _validate_prepared_request(
        input_result: FireSpreadInputResult,
        as_of: datetime,
        horizon_minutes: int,
        effective_state_fingerprint: str | None,
    ) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
        if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int):
            raise ValueError(f"horizon_minutes must be an integer, got {horizon_minutes!r}")
        validate_effective_state_fingerprint(effective_state_fingerprint)

    @staticmethod
    def _validate_ready_input_result(input_result: FireSpreadInputResult) -> None:
        if input_result.input_data is None:
            raise ValueError("READY fire-spread input result requires input_data.")
        if input_result.severity_assessment_id is None:
            raise ValueError("READY fire-spread input result requires severity_assessment_id.")
