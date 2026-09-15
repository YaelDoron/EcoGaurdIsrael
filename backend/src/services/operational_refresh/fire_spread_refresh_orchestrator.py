"""Operational refresh orchestration for fire-spread predictions."""
from __future__ import annotations

from datetime import datetime
import logging
from collections.abc import Iterable

from src.agents.analysis.fire_spread_prediction_agent import FireSpreadPredictionAgent
from src.models.fire_spread_effective_state import FireSpreadEffectiveState
from src.models.fire_spread_input_status import FireSpreadInputStatus
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES
from src.models.fire_spread_prediction_status import FireSpreadPredictionStatus
from src.models.operational_refresh_trigger_type import OperationalRefreshTriggerType
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService
from src.services.operational_refresh.fire_spread_refresh_result import (
    FireSpreadRefreshHorizonResult,
    FireSpreadRefreshHorizonStatus,
    FireSpreadRefreshResult,
)
from src.services.operational_refresh.operational_refresh_policy import requires_spread_reevaluation

logger = logging.getLogger(__name__)


class FireSpreadRefreshOrchestrator:
    """Reevaluate current spread inputs and avoid duplicate prediction rows."""

    def __init__(
        self,
        *,
        input_service: FireSpreadInputService,
        prediction_agent: FireSpreadPredictionAgent,
        prediction_repository: FireSpreadPredictionRepository,
    ) -> None:
        self._input_service = input_service
        self._prediction_agent = prediction_agent
        self._prediction_repository = prediction_repository

    def refresh(
        self,
        *,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
        horizons: Iterable[int] = SUPPORTED_HORIZON_MINUTES,
    ) -> FireSpreadRefreshResult:
        """Refresh spread predictions for all requested horizons."""
        self._validate_request(fire_event_id, trigger_type, as_of)
        horizon_values = self._normalize_horizons(horizons)

        if not requires_spread_reevaluation(trigger_type):
            return FireSpreadRefreshResult(
                fire_event_id=fire_event_id,
                trigger_type=trigger_type,
                as_of=as_of,
                reevaluation_required=False,
                horizon_results=(),
            )

        horizon_results = tuple(
            self._refresh_horizon(
                fire_event_id=fire_event_id,
                as_of=as_of,
                horizon_minutes=horizon_minutes,
            )
            for horizon_minutes in horizon_values
        )
        return FireSpreadRefreshResult(
            fire_event_id=fire_event_id,
            trigger_type=trigger_type,
            as_of=as_of,
            reevaluation_required=True,
            horizon_results=horizon_results,
        )

    def _refresh_horizon(
        self,
        *,
        fire_event_id: int,
        as_of: datetime,
        horizon_minutes: int,
    ) -> FireSpreadRefreshHorizonResult:
        try:
            input_result = self._input_service.prepare_input(
                fire_event_id=fire_event_id,
                as_of=as_of,
                horizon_minutes=horizon_minutes,
            )
            latest = self._prediction_repository.get_latest_for_event_and_horizon_as_of(
                fire_event_id,
                horizon_minutes,
                as_of,
            )

            if input_result.status is FireSpreadInputStatus.READY:
                effective_state = FireSpreadEffectiveState.from_input(
                    fire_event_id=fire_event_id,
                    spread_input=input_result.input_data,
                )
                fingerprint = effective_state.fingerprint
                if latest is not None and latest.prediction.effective_state_fingerprint == fingerprint:
                    return FireSpreadRefreshHorizonResult(
                        fire_event_id=fire_event_id,
                        horizon_minutes=horizon_minutes,
                        status=FireSpreadRefreshHorizonStatus.NO_OP,
                        previous_prediction_id=latest.id,
                        effective_state_fingerprint=fingerprint,
                    )
                stored = self._prediction_agent.predict_from_input_result(
                    input_result=input_result,
                    as_of=as_of,
                    horizon_minutes=horizon_minutes,
                    effective_state_fingerprint=fingerprint,
                )
                return FireSpreadRefreshHorizonResult(
                    fire_event_id=fire_event_id,
                    horizon_minutes=horizon_minutes,
                    status=FireSpreadRefreshHorizonStatus.REFRESHED,
                    prediction_id=stored.id,
                    previous_prediction_id=latest.id if latest is not None else None,
                    effective_state_fingerprint=fingerprint,
                )

            if input_result.status is FireSpreadInputStatus.INSUFFICIENT_DATA:
                return self._handle_non_ready(
                    input_result=input_result,
                    as_of=as_of,
                    horizon_minutes=horizon_minutes,
                    latest=latest,
                    refresh_status=FireSpreadRefreshHorizonStatus.INSUFFICIENT_DATA,
                    prediction_status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
                )
            if input_result.status is FireSpreadInputStatus.INACTIVE_EVENT:
                return self._handle_non_ready(
                    input_result=input_result,
                    as_of=as_of,
                    horizon_minutes=horizon_minutes,
                    latest=latest,
                    refresh_status=FireSpreadRefreshHorizonStatus.INACTIVE_EVENT,
                    prediction_status=FireSpreadPredictionStatus.INACTIVE_EVENT,
                )
            raise ValueError(f"Unsupported fire-spread input status: {input_result.status!r}")
        except Exception as exc:  # noqa: BLE001 - per-horizon failure isolation.
            logger.exception("Fire-spread refresh failed for FireEvent %s horizon %s", fire_event_id, horizon_minutes)
            return FireSpreadRefreshHorizonResult(
                fire_event_id=fire_event_id,
                horizon_minutes=horizon_minutes,
                status=FireSpreadRefreshHorizonStatus.FAILED,
                error_message=str(exc) or "Fire-spread refresh failed.",
            )

    def _handle_non_ready(
        self,
        *,
        input_result,
        as_of: datetime,
        horizon_minutes: int,
        latest,
        refresh_status: FireSpreadRefreshHorizonStatus,
        prediction_status: FireSpreadPredictionStatus,
    ) -> FireSpreadRefreshHorizonResult:
        if latest is not None and latest.prediction.status is prediction_status:
            return FireSpreadRefreshHorizonResult(
                fire_event_id=input_result.fire_event_id,
                horizon_minutes=horizon_minutes,
                status=FireSpreadRefreshHorizonStatus.NO_OP,
                previous_prediction_id=latest.id,
            )
        stored = self._prediction_agent.predict_from_input_result(
            input_result=input_result,
            as_of=as_of,
            horizon_minutes=horizon_minutes,
        )
        return FireSpreadRefreshHorizonResult(
            fire_event_id=input_result.fire_event_id,
            horizon_minutes=horizon_minutes,
            status=refresh_status,
            prediction_id=stored.id,
            previous_prediction_id=latest.id if latest is not None else None,
        )

    @staticmethod
    def _normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
        try:
            values = tuple(sorted(set(horizons)))
        except TypeError as exc:
            raise ValueError("horizons must be iterable.") from exc
        for horizon_minutes in values:
            if horizon_minutes not in SUPPORTED_HORIZON_MINUTES:
                raise ValueError(f"horizon_minutes must be one of {SUPPORTED_HORIZON_MINUTES}, got {horizon_minutes!r}")
        return values

    @staticmethod
    def _validate_request(
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
    ) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(trigger_type, OperationalRefreshTriggerType):
            raise ValueError(f"trigger_type must be an OperationalRefreshTriggerType, got {trigger_type!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
