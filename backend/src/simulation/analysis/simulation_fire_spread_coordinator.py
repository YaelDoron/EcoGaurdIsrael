"""Coordinate wildfire-spread prediction after simulated severity assessment succeeds.

Thin trigger policy only: the real FireSpreadPredictionAgent (and, inside it,
the real FireSpreadInputService/FireSpreadCalculator/FireSpreadPredictionRepository
from Tasks 4B-7) does all input selection, PROPAGATOR calculation, and
persistence. This coordinator does not duplicate weather selection,
vegetation mapping, EMC, grid/CA logic, or repository access -- it only
decides *which* FireEvents to predict spread for and *when*, mirroring
SimulationFireSeverityCoordinator's existing trigger-policy pattern.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_spread_prediction_agent import FireSpreadPredictionAgent
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES
from src.simulation.analysis.simulation_fire_severity_result import SimulationFireSeverityResult
from src.simulation.analysis.simulation_fire_spread_result import SimulationFireSpreadResult

logger = logging.getLogger(__name__)

SEVERITY_NOT_TRIGGERED_REASON = "severity_not_triggered"
NO_SEVERITY_ASSESSMENTS_REASON = "no_severity_assessments"


class SimulationFireSpreadCoordinator:
    """Trigger 30- and 60-minute wildfire-spread predictions after simulated severity assessment."""

    def __init__(self, spread_agent: FireSpreadPredictionAgent) -> None:
        self._spread_agent = spread_agent

    def handle_severity_result(
        self,
        severity_result: SimulationFireSeverityResult,
        event_timestamp: datetime,
    ) -> SimulationFireSpreadResult:
        """Predict spread, at every supported horizon, for every FireEvent this
        tick's severity assessment covered.

        Only FireEvents with a *successful* severity assessment this tick are
        considered -- this mirrors how severity itself only reacts to a
        successful detection result, rather than inventing a new trigger rule.
        The production agent (not this coordinator) decides VALID vs
        INSUFFICIENT_DATA vs INACTIVE_EVENT for each prediction; this
        coordinator never inspects or overrides that decision.
        """
        if not severity_result.triggered:
            return SimulationFireSpreadResult(triggered=False, reason=SEVERITY_NOT_TRIGGERED_REASON)

        fire_event_ids = tuple(
            sorted({stored.assessment.fire_event_id for stored in severity_result.assessment_results})
        )
        if not fire_event_ids:
            return SimulationFireSpreadResult(triggered=False, reason=NO_SEVERITY_ASSESSMENTS_REASON)

        prediction_results = []
        failed_fire_event_ids = []
        error_messages = []
        for fire_event_id in fire_event_ids:
            for horizon_minutes in SUPPORTED_HORIZON_MINUTES:
                try:
                    prediction_results.append(
                        self._spread_agent.predict(
                            fire_event_id=fire_event_id,
                            as_of=event_timestamp,
                            horizon_minutes=horizon_minutes,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - record per-event/horizon analysis failure.
                    logger.exception(
                        "Simulation fire-spread prediction failed for FireEvent %s horizon %s",
                        fire_event_id,
                        horizon_minutes,
                    )
                    failed_fire_event_ids.append(fire_event_id)
                    error_messages.append(str(exc) or "Fire-spread prediction failed.")

        return SimulationFireSpreadResult(
            triggered=True,
            prediction_results=tuple(prediction_results),
            failed_fire_event_ids=tuple(failed_fire_event_ids),
            error_messages=tuple(error_messages),
        )
