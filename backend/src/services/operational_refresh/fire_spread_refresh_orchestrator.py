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
        """Refresh spread predictions for all requested horizons. Fetches
        the FireEvent (and, unless a fresher in-cycle severity result is
        supplied elsewhere, the latest severity) from persistence -
        unchanged behavior/signature for existing callers. See
        refresh_for_event() for the performance-pass overload used by
        OperationalRefreshOrchestrator."""
        self._validate_request(fire_event_id, trigger_type, as_of)
        horizon_values = self._normalize_horizons(horizons)
        return self._run(
            fire_event_id=fire_event_id,
            trigger_type=trigger_type,
            as_of=as_of,
            horizon_values=horizon_values,
            load_shared_context=lambda: self._input_service.prepare_shared_context(fire_event_id, as_of),
        )

    def refresh_for_event(
        self,
        *,
        stored_event,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
        resolved_severity=None,
        horizons: Iterable[int] = SUPPORTED_HORIZON_MINUTES,
    ) -> FireSpreadRefreshResult:
        """Same refresh as refresh(), but for an ALREADY-LOADED StoredFireEvent
        and, optionally, an ALREADY-RESOLVED severity assessment (performance
        pass: avoids a redundant FireEvent fetch and, when `resolved_severity`
        is supplied, a redundant "latest severity" query within one
        OperationalRefreshOrchestrator cycle). `resolved_severity=None` falls
        back to a repository read, exactly like refresh()."""
        fire_event_id = stored_event.id
        self._validate_request(fire_event_id, trigger_type, as_of)
        horizon_values = self._normalize_horizons(horizons)
        return self._run(
            fire_event_id=fire_event_id,
            trigger_type=trigger_type,
            as_of=as_of,
            horizon_values=horizon_values,
            load_shared_context=lambda: self._input_service.prepare_shared_context_for_event(
                stored_event, as_of, resolved_severity=resolved_severity
            ),
        )

    def _run(
        self,
        *,
        fire_event_id: int,
        trigger_type: OperationalRefreshTriggerType,
        as_of: datetime,
        horizon_values: tuple[int, ...],
        load_shared_context,
    ) -> FireSpreadRefreshResult:
        if not requires_spread_reevaluation(trigger_type):
            return FireSpreadRefreshResult(
                fire_event_id=fire_event_id,
                trigger_type=trigger_type,
                as_of=as_of,
                reevaluation_required=False,
                horizon_results=(),
            )

        # Performance pass: load every horizon-independent input (FireEvent,
        # latest severity assessment, selected weather) ONCE here, instead of
        # once per horizon inside _refresh_horizon - profiling showed this
        # was a pure N+1 (identical DB reads repeated for 30m and 60m).
        # A failure here affects every horizon identically (they all depend
        # on the exact same shared data), so it is reported as FAILED for
        # every horizon rather than silently aborting the whole refresh -
        # preserving this module's per-horizon-result contract even though
        # the load itself is no longer literally per-horizon.
        try:
            shared_context = load_shared_context()
        except Exception as exc:  # noqa: BLE001 - shared-load failure isolation, see comment above.
            logger.exception("Fire-spread shared input load failed for FireEvent %s", fire_event_id)
            horizon_results = tuple(
                FireSpreadRefreshHorizonResult(
                    fire_event_id=fire_event_id,
                    horizon_minutes=horizon_minutes,
                    status=FireSpreadRefreshHorizonStatus.FAILED,
                    error_message=str(exc) or "Fire-spread refresh failed.",
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

        # Performance pass: the "latest prediction per horizon" read used to
        # happen once per horizon inside _refresh_horizon (2 separate
        # multi-round-trip selectinload queries for 30m/60m). Batched here
        # via the same window-function query Response Targets already uses,
        # cutting it to ~3 round trips total for both horizons combined.
        latest_by_horizon = self._prediction_repository.get_latest_for_event_and_horizons_as_of(
            fire_event_id, horizon_values, as_of
        )
        horizon_results = tuple(
            self._refresh_horizon(
                fire_event_id=fire_event_id,
                as_of=as_of,
                horizon_minutes=horizon_minutes,
                shared_context=shared_context,
                latest=latest_by_horizon.get(horizon_minutes),
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
        shared_context,
        latest,
    ) -> FireSpreadRefreshHorizonResult:
        try:
            input_result = self._input_service.build_input_for_horizon(shared_context, horizon_minutes)

            if input_result.status is FireSpreadInputStatus.READY:
                effective_state = FireSpreadEffectiveState.from_input(
                    fire_event_id=fire_event_id,
                    spread_input=input_result.input_data,
                )
                fingerprint = effective_state.fingerprint
                if latest is not None and latest.prediction.effective_state_fingerprint == fingerprint:
                    # NO_OP: `latest` is already the full WithCells object -
                    # handoff to Response Targets is free, zero extra reads.
                    return FireSpreadRefreshHorizonResult(
                        fire_event_id=fire_event_id,
                        horizon_minutes=horizon_minutes,
                        status=FireSpreadRefreshHorizonStatus.NO_OP,
                        previous_prediction_id=latest.id,
                        effective_state_fingerprint=fingerprint,
                        resolved_prediction=latest,
                    )
                stored = self._prediction_agent.predict_from_input_result(
                    input_result=input_result,
                    as_of=as_of,
                    horizon_minutes=horizon_minutes,
                    effective_state_fingerprint=fingerprint,
                )
                # REFRESHED: save_prediction() does not return cells, so one
                # follow-up read is needed for the WithCells handoff object -
                # this only happens on the minority "genuinely recomputed"
                # path (confirmed ~8/34 horizon-cycles in the last measured
                # run), never on the dominant NO_OP path above.
                resolved = self._prediction_repository.get_latest_for_event_and_horizon_as_of(
                    fire_event_id, horizon_minutes, as_of
                )
                return FireSpreadRefreshHorizonResult(
                    fire_event_id=fire_event_id,
                    horizon_minutes=horizon_minutes,
                    status=FireSpreadRefreshHorizonStatus.REFRESHED,
                    prediction_id=stored.id,
                    previous_prediction_id=latest.id if latest is not None else None,
                    effective_state_fingerprint=fingerprint,
                    resolved_prediction=resolved,
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
                resolved_prediction=latest,
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
