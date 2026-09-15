"""Coordinate response-target generation after simulation analysis touches FireEvents."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import logging

from src.agents.analysis.response_target_generation_agent import ResponseTargetGenerationAgent
from src.agents.analysis.response_target_generation_result import (
    ResponseTargetGenerationResult,
    ResponseTargetGenerationStatus,
)
from src.simulation.analysis.simulation_response_target_result import SimulationResponseTargetResult

logger = logging.getLogger(__name__)

NO_AFFECTED_FIRE_EVENTS_REASON = "no_affected_fire_events"


class SimulationResponseTargetCoordinator:
    """Trigger response-target generation for FireEvents affected by one simulation tick."""

    def __init__(self, generation_agent: ResponseTargetGenerationAgent) -> None:
        self._generation_agent = generation_agent

    def generate_for_fire_events(
        self,
        fire_event_ids: Iterable[int],
        as_of: datetime,
    ) -> SimulationResponseTargetResult:
        """Generate one response-target set per unique affected FireEvent.

        The production ResponseTargetGenerationAgent owns input gathering,
        target prioritization, and persistence. This coordinator only provides
        deterministic per-tick orchestration and fault isolation.
        """
        self._validate_timestamp(as_of)
        normalized_fire_event_ids = self._normalize_fire_event_ids(fire_event_ids)
        if not normalized_fire_event_ids:
            return SimulationResponseTargetResult(triggered=False, reason=NO_AFFECTED_FIRE_EVENTS_REASON)

        generation_results = []
        for fire_event_id in normalized_fire_event_ids:
            try:
                generation_results.append(
                    self._generation_agent.generate(
                        fire_event_id=fire_event_id,
                        as_of=as_of,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - isolate one FireEvent's analysis failure.
                logger.exception("Simulation response-target generation failed for FireEvent %s", fire_event_id)
                generation_results.append(
                    ResponseTargetGenerationResult(
                        success=False,
                        fire_event_id=fire_event_id,
                        status=ResponseTargetGenerationStatus.FAILED,
                        target_set_id=None,
                        targets=(),
                        target_count=0,
                        error_message=str(exc) or "Response-target generation failed.",
                    )
                )

        return SimulationResponseTargetResult(
            triggered=True,
            generation_results=tuple(generation_results),
            fire_event_ids=normalized_fire_event_ids,
        )

    @staticmethod
    def _normalize_fire_event_ids(fire_event_ids: Iterable[int]) -> tuple[int, ...]:
        normalized_ids = tuple(sorted(set(fire_event_ids)))
        for fire_event_id in normalized_ids:
            if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
                raise ValueError(f"fire_event_ids must contain positive integer ids, got {fire_event_id!r}")
        return normalized_ids

    @staticmethod
    def _validate_timestamp(as_of: datetime) -> None:
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
