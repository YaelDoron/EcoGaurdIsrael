"""Orchestrates response-target generation from prepared input to persistence."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.response_target_generation_result import (
    ResponseTargetGenerationResult,
    ResponseTargetGenerationStatus,
)
from src.calculators.response_target.response_target_calculator import ResponseTargetCalculator
from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.models.response_target_input_status import ResponseTargetInputStatus
from src.models.response_target_set import ResponseTargetSet
from src.repositories.response_target_repository import ResponseTargetRepository
from src.services.response_target.response_target_input_service import ResponseTargetInputService

logger = logging.getLogger(__name__)


def _target_content_signature(target) -> tuple:
    """Everything that makes two targets semantically the SAME target - never
    a raw DB id (targets have none until persisted) or generated_at/set id.
    Order matters: build_targets()/the stored set both order targets
    deterministically (ACTIVE_FIRE first, then PREDICTED_RISK by descending
    priority - see ResponseTargetCalculator), so comparing as ordered
    sequences is correct, not just as sets."""
    return (
        target.target_type.value,
        target.latitude,
        target.longitude,
        target.priority_score,
        target.prediction_horizon_minutes,
        target.spread_prediction_id,
        target.spread_prediction_cell_id,
    )


def _same_target_content(targets, other_targets) -> bool:
    if len(targets) != len(other_targets):
        return False
    return all(
        _target_content_signature(a) == _target_content_signature(b)
        for a, b in zip(targets, other_targets)
    )


class ResponseTargetGenerationAgent:
    """Coordinates response-target input preparation, calculation, and snapshot persistence."""

    def __init__(
        self,
        input_service: ResponseTargetInputService,
        calculator: ResponseTargetCalculator,
        repository: ResponseTargetRepository,
    ) -> None:
        self._input_service = input_service
        self._calculator = calculator
        self._repository = repository

    def generate(self, *, fire_event_id: int, as_of: datetime) -> ResponseTargetGenerationResult:
        """Generate and persist response targets for one FireEvent at a
        timezone-aware instant. Fetches the FireEvent (and latest severity/
        spread) by id via the input service - unchanged behavior/signature
        for existing callers."""
        self._validate_request(fire_event_id, as_of)
        try:
            input_result = self._input_service.prepare_input(fire_event_id=fire_event_id, as_of=as_of)
            return self._generate_from_input_result(input_result, as_of)
        except Exception:
            logger.exception("Response-target generation failed for FireEvent %s", fire_event_id)
            return ResponseTargetGenerationResult(
                success=False,
                fire_event_id=fire_event_id,
                status=ResponseTargetGenerationStatus.FAILED,
                target_set_id=None,
                targets=(),
                target_count=0,
                error_message="Response-target generation failed.",
            )

    def generate_for_event(
        self,
        *,
        stored_event,
        as_of: datetime,
        resolved_severity=None,
        resolved_spread_by_horizon=None,
    ) -> ResponseTargetGenerationResult:
        """Same generation as generate(), but for an ALREADY-LOADED
        StoredFireEvent, and optionally ALREADY-RESOLVED severity/spread
        results (performance pass: avoids a redundant FireEvent fetch, and
        - when supplied - redundant "latest severity"/"latest spread"
        repository reads, since OperationalRefreshOrchestrator has just
        established these authoritative results earlier in the same cycle).
        Falls back to repository reads for anything not supplied, exactly
        like generate()."""
        fire_event_id = stored_event.id
        self._validate_request(fire_event_id, as_of)
        try:
            input_result = self._input_service.prepare_input_for_event(
                stored_event,
                as_of,
                resolved_severity=resolved_severity,
                resolved_spread_by_horizon=resolved_spread_by_horizon,
            )
            return self._generate_from_input_result(input_result, as_of)
        except Exception:
            logger.exception("Response-target generation failed for FireEvent %s", fire_event_id)
            return ResponseTargetGenerationResult(
                success=False,
                fire_event_id=fire_event_id,
                status=ResponseTargetGenerationStatus.FAILED,
                target_set_id=None,
                targets=(),
                target_count=0,
                error_message="Response-target generation failed.",
            )

    def _generate_from_input_result(self, input_result, as_of: datetime) -> ResponseTargetGenerationResult:
        """Shared body for generate()/generate_for_event() once an input
        result is in hand. Deliberately raises on any unexpected condition -
        both public entry points wrap this in their own try/except and
        convert any exception into a FAILED result."""
        fire_event_id = input_result.fire_event_id
        logger.info(
            "Prepared response-target input for FireEvent %s with status %s",
            fire_event_id,
            input_result.status.value,
        )

        if input_result.status is ResponseTargetInputStatus.INACTIVE_EVENT:
            return ResponseTargetGenerationResult(
                success=True,
                fire_event_id=fire_event_id,
                status=ResponseTargetGenerationStatus.INACTIVE_EVENT,
                target_set_id=None,
                targets=(),
                target_count=0,
                error_message=None,
            )
        if input_result.status is not ResponseTargetInputStatus.READY:
            raise ValueError(f"Unsupported response-target input status: {input_result.status!r}")
        if input_result.input_data is None:
            raise ValueError("READY response-target input result requires input_data.")

        target_input = input_result.input_data
        targets = self._calculator.build_targets(
            fire_event_id=target_input.fire_event_id,
            fire_latitude=target_input.fire_latitude,
            fire_longitude=target_input.fire_longitude,
            severity_score=target_input.severity_score,
            predicted_candidates=target_input.predicted_candidates,
        )
        if not targets:
            raise ValueError("READY response-target generation produced no targets.")

        # Performance pass: response_target_sets is append-only by design
        # (ResponseTargetSetDB's own docstring) - every environmental
        # refresh used to mint a brand-new set even when target CONTENT
        # was byte-identical to the current one (GlobalPlanningInputBuilder's
        # fingerprint already compensates for this by hashing target
        # CONTENT, not raw ids - see its docstring). Comparing the freshly
        # computed targets against the latest persisted set's content
        # lets us reuse the existing row instead of inserting a duplicate,
        # without touching the append-only invariant (nothing is ever
        # mutated or deleted) or adding any new schema/metadata.
        latest = self._repository.get_latest_for_event_as_of(target_input.fire_event_id, as_of)
        if latest is not None and _same_target_content(targets, latest.target_set.targets):
            logger.info(
                "Reused response-target set %s for FireEvent %s - content unchanged.",
                latest.id,
                target_input.fire_event_id,
            )
            return ResponseTargetGenerationResult(
                success=True,
                fire_event_id=target_input.fire_event_id,
                status=ResponseTargetGenerationStatus.GENERATED,
                target_set_id=latest.id,
                targets=latest.target_set.targets,
                target_count=len(latest.target_set.targets),
                error_message=None,
            )

        target_set = ResponseTargetSet(
            fire_event_id=target_input.fire_event_id,
            generated_at=as_of,
            methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
            methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
            targets=targets,
        )
        stored = self._repository.save_target_set(target_set)
        logger.info(
            "Stored response-target set %s for FireEvent %s with %s targets",
            stored.id,
            target_input.fire_event_id,
            len(stored.target_set.targets),
        )
        return ResponseTargetGenerationResult(
            success=True,
            fire_event_id=target_input.fire_event_id,
            status=ResponseTargetGenerationStatus.GENERATED,
            target_set_id=stored.id,
            targets=stored.target_set.targets,
            target_count=len(stored.target_set.targets),
            error_message=None,
        )

    @staticmethod
    def _validate_request(fire_event_id: int, as_of: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
