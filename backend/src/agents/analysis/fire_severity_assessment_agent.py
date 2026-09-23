"""Orchestrates active wildfire severity assessment from prepared input to persistence."""
from __future__ import annotations

from datetime import datetime
import logging

from src.calculators.fire_severity.fire_severity_calculator import FireSeverityCalculator
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_input_result import FireSeverityInputResult
from src.models.fire_severity_input_status import FireSeverityInputStatus
from src.models.vegetation_data import VegetationData
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.services.fire_severity.fire_severity_input_service import FireSeverityInputService

logger = logging.getLogger(__name__)


class FireSeverityAssessmentAgent:
    """Coordinates severity input preparation, calculation, and assessment persistence."""

    def __init__(
        self,
        input_service: FireSeverityInputService,
        calculator: FireSeverityCalculator,
        repository: FireSeverityAssessmentRepository,
    ) -> None:
        self._input_service = input_service
        self._calculator = calculator
        self._repository = repository

    def assess(self, fire_event_id: int, assessed_at: datetime) -> StoredFireSeverityAssessment:
        """Assess active wildfire severity for one FireEvent at a timezone-aware instant.

        Fetches the FireEvent by id, then delegates to the shared assessment
        body - unchanged behavior/signature for existing callers.
        """
        self._validate_request(fire_event_id, assessed_at)
        input_result = self._input_service.prepare_input(fire_event_id=fire_event_id, as_of=assessed_at)
        return self._assess_from_input_result(input_result, assessed_at)

    def assess_for_event(self, stored_event, assessed_at: datetime) -> StoredFireSeverityAssessment:
        """Same assessment as assess(), but for an ALREADY-LOADED StoredFireEvent
        (performance pass: avoids a redundant FireEvent fetch when the caller,
        FireSeverityRefreshOrchestrator, already has one for this refresh cycle
        and has determined a fresh compute is actually needed)."""
        self._validate_request(stored_event.id, assessed_at)
        input_result = self._input_service.prepare_input_for_event(stored_event, assessed_at)
        return self._assess_from_input_result(input_result, assessed_at)

    def assess_from_input_result(
        self, input_result: FireSeverityInputResult, assessed_at: datetime
    ) -> StoredFireSeverityAssessment:
        """Same assessment as assess()/assess_for_event(), but for an
        ALREADY-PREPARED FireSeverityInputResult (performance pass: avoids
        re-running input preparation - weather/satellite lookups and a real
        Copernicus vegetation call - when the caller, FireSeverityRefreshOrchestrator,
        already prepared this exact input_result to make its reuse decision
        and is now falling through to a fresh compute)."""
        self._validate_request(input_result.fire_event_id, assessed_at)
        return self._assess_from_input_result(input_result, assessed_at)

    def _assess_from_input_result(
        self, input_result: FireSeverityInputResult, assessed_at: datetime
    ) -> StoredFireSeverityAssessment:
        fire_event_id = input_result.fire_event_id
        logger.info(
            "Prepared fire-severity input for FireEvent %s with status %s",
            fire_event_id,
            input_result.status.value,
        )

        if input_result.status is FireSeverityInputStatus.READY:
            assessment = self._build_valid_assessment(input_result, assessed_at)
        elif input_result.status is FireSeverityInputStatus.INSUFFICIENT_DATA:
            assessment = self._build_not_calculated_assessment(
                input_result=input_result,
                assessed_at=assessed_at,
                status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
            )
        elif input_result.status is FireSeverityInputStatus.INACTIVE_EVENT:
            assessment = self._build_not_calculated_assessment(
                input_result=input_result,
                assessed_at=assessed_at,
                status=FireSeverityAssessmentStatus.INACTIVE_EVENT,
            )
        else:
            raise ValueError(f"Unsupported fire-severity input status: {input_result.status!r}")

        stored = self._repository.save_assessment(
            assessment=assessment,
            weather_observation_ids=input_result.weather_observation_ids,
            satellite_hotspot_ids=input_result.satellite_hotspot_ids,
            selected_frp_hotspot_id=input_result.selected_frp_hotspot_id,
        )
        logger.info(
            "Stored fire-severity assessment %s for FireEvent %s with status %s",
            stored.assessment_id,
            fire_event_id,
            stored.assessment.status.value,
        )
        return stored

    def _build_valid_assessment(
        self,
        input_result: FireSeverityInputResult,
        assessed_at: datetime,
    ) -> FireSeverityAssessment:
        if input_result.input_data is None:
            raise ValueError("READY fire-severity input result requires input_data.")

        calculation = self._calculator.calculate(input_result.input_data)
        return FireSeverityAssessment(
            fire_event_id=input_result.fire_event_id,
            assessed_at=assessed_at,
            status=FireSeverityAssessmentStatus.VALID,
            score=calculation.score,
            level=calculation.level,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            **self._vegetation_snapshot(input_result.vegetation_data),
        )

    @staticmethod
    def _build_not_calculated_assessment(
        input_result: FireSeverityInputResult,
        assessed_at: datetime,
        status: FireSeverityAssessmentStatus,
    ) -> FireSeverityAssessment:
        return FireSeverityAssessment(
            fire_event_id=input_result.fire_event_id,
            assessed_at=assessed_at,
            status=status,
            score=None,
            level=None,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            **FireSeverityAssessmentAgent._vegetation_snapshot(input_result.vegetation_data),
        )

    @staticmethod
    def _vegetation_snapshot(vegetation_data: VegetationData | None) -> dict[str, object | None]:
        if vegetation_data is None:
            return {
                "vegetation_source": None,
                "vegetation_dataset_year": None,
                "vegetation_radius_km": None,
                "vegetation_dominant_land_cover": None,
                "vegetation_fuel_score": None,
            }
        return {
            "vegetation_source": vegetation_data.source,
            "vegetation_dataset_year": vegetation_data.dataset_year,
            "vegetation_radius_km": vegetation_data.radius_km,
            "vegetation_dominant_land_cover": vegetation_data.dominant_land_cover,
            "vegetation_fuel_score": vegetation_data.fuel_score,
        }

    @staticmethod
    def _validate_request(fire_event_id: int, assessed_at: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(assessed_at, datetime) or assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {assessed_at!r}")
