"""Orchestrates fire-danger assessment from weather input through persistence."""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_danger_assessment_result import FireDangerAssessmentResult
from src.calculators.fire_danger.ffwi_calculator import FFWICalculator
from src.calculators.fire_danger.ffwi_config import (
    FFWI_METHODOLOGY_NAME,
    FFWI_METHODOLOGY_VERSION,
)
from src.models.assessment_area import AssessmentArea
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_input_status import FireDangerInputStatus
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.services.fire_danger.fire_danger_input_service import FireDangerInputService

logger = logging.getLogger(__name__)


class FireDangerAssessmentAgent:
    """Coordinates input preparation, FFWI calculation, and assessment persistence."""

    def __init__(
        self,
        input_service: FireDangerInputService,
        calculator: FFWICalculator,
        repository: FireDangerAssessmentRepository,
    ) -> None:
        self._input_service = input_service
        self._calculator = calculator
        self._repository = repository

    def assess(self, area: AssessmentArea, as_of: datetime) -> FireDangerAssessmentResult:
        """Assess fire-weather danger for an area at a timezone-aware instant."""
        self._validate_request(area, as_of)

        try:
            input_result = self._input_service.build_input(area=area, as_of=as_of)
        except Exception:
            logger.exception("Fire-danger input preparation failed for area %s", area.id)
            return FireDangerAssessmentResult(
                assessment=None,
                stored_assessment_id=None,
                success=False,
                error_message="Fire-danger input preparation failed.",
            )

        if input_result.status is FireDangerInputStatus.READY:
            return self._assess_ready_input(area, as_of, input_result)
        if input_result.status is FireDangerInputStatus.INSUFFICIENT_DATA:
            return self._persist_insufficient_data_assessment(area, as_of)

        return FireDangerAssessmentResult(
            assessment=None,
            stored_assessment_id=None,
            success=False,
            error_message="Fire-danger input preparation returned an unsupported status.",
        )

    def _assess_ready_input(self, area, as_of, input_result) -> FireDangerAssessmentResult:  # noqa: ANN001
        if input_result.input_data is None:
            return FireDangerAssessmentResult(
                assessment=None,
                stored_assessment_id=None,
                success=False,
                error_message="Fire-danger input preparation returned no input data.",
            )

        try:
            calculation = self._calculator.calculate(input_result.input_data)
        except Exception:
            logger.exception("Fire-danger calculation failed for area %s", area.id)
            return FireDangerAssessmentResult(
                assessment=None,
                stored_assessment_id=None,
                success=False,
                error_message="Fire-danger calculation failed.",
            )

        assessment = self._build_valid_assessment(area, as_of, calculation)
        return self._save_assessment(
            assessment=assessment,
            observation_ids=input_result.observation_ids,
            station_ids=input_result.station_ids,
        )

    def _persist_insufficient_data_assessment(
        self,
        area: AssessmentArea,
        as_of: datetime,
    ) -> FireDangerAssessmentResult:
        assessment = self._build_insufficient_data_assessment(area, as_of)
        return self._save_assessment(assessment=assessment, observation_ids=(), station_ids=())

    def _save_assessment(
        self,
        assessment: FireDangerAssessment,
        observation_ids: tuple[int, ...],
        station_ids: tuple[int, ...],
    ) -> FireDangerAssessmentResult:
        try:
            saved = self._repository.save_assessment(
                assessment,
                observation_ids=observation_ids,
                station_ids=station_ids,
            )
        except Exception:
            logger.exception("Fire-danger assessment persistence failed for area %s", assessment.area_id)
            return FireDangerAssessmentResult(
                assessment=assessment,
                stored_assessment_id=None,
                success=False,
                error_message="Fire-danger assessment persistence failed.",
            )

        return FireDangerAssessmentResult(
            assessment=saved.assessment,
            stored_assessment_id=saved.assessment_id,
            success=True,
            error_message=None,
        )

    @staticmethod
    def _build_valid_assessment(area, as_of, calculation) -> FireDangerAssessment:  # noqa: ANN001
        return FireDangerAssessment(
            area_id=area.id,
            area_name=area.name,
            area_latitude=area.latitude,
            area_longitude=area.longitude,
            area_radius_km=area.radius_km,
            assessed_at=as_of,
            status=FireDangerAssessmentStatus.VALID,
            score=calculation.score,
            level=calculation.level,
            methodology=FFWI_METHODOLOGY_NAME,
            methodology_version=FFWI_METHODOLOGY_VERSION,
        )

    @staticmethod
    def _build_insufficient_data_assessment(area: AssessmentArea, as_of: datetime) -> FireDangerAssessment:
        return FireDangerAssessment(
            area_id=area.id,
            area_name=area.name,
            area_latitude=area.latitude,
            area_longitude=area.longitude,
            area_radius_km=area.radius_km,
            assessed_at=as_of,
            status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
            score=None,
            level=None,
            methodology=FFWI_METHODOLOGY_NAME,
            methodology_version=FFWI_METHODOLOGY_VERSION,
        )

    @staticmethod
    def _validate_request(area: AssessmentArea, as_of: datetime) -> None:
        if not isinstance(area, AssessmentArea):
            raise ValueError(f"area must be an AssessmentArea, got {area!r}")
        if not isinstance(as_of, datetime) or as_of.tzinfo is None:
            raise ValueError(f"as_of must be a timezone-aware datetime, got {as_of!r}")
