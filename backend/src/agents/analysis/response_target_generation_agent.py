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
        """Generate and persist response targets for one FireEvent at a timezone-aware instant."""
        self._validate_request(fire_event_id, as_of)

        try:
            input_result = self._input_service.prepare_input(fire_event_id=fire_event_id, as_of=as_of)
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

    @staticmethod
    def _validate_request(fire_event_id: int, as_of: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
