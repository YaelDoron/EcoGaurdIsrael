"""Operational refresh orchestration for fire-severity assessments.

Mirrors FireSpreadRefreshOrchestrator's reuse-first architecture (performance
pass: profiling showed FireSeverityAssessmentAgent.assess() unconditionally
computed and persisted a brand-new FireSeverityAssessment row on every call,
even when nothing about the underlying evidence had changed).

Unlike Spread, this does NOT need a new fingerprint column: prepare_input()
already resolves and returns the exact `weather_observation_ids`/
`satellite_hotspot_ids`/`selected_frp_hotspot_id` it would persist, and
save_assessment() already persists those same ids via the existing
FireSeverityAssessmentWeatherInputDB/FireSeverityAssessmentSatelliteInputDB
trace tables (get_latest_for_event_as_of already joins and returns them).
WeatherObservationDB/SatelliteHotspotDB rows are append-only (never updated
after creation) - so "the same evidence-id set as last time" is provably
equivalent to "the same calculator inputs as last time" (same averaged
wind/humidity, same selected FRP value), without needing a separately
computed/stored fingerprint. This directly follows the reuse-existing-
preserved-input-reference-tables guidance over adding redundant metadata.

FireSeverityAssessmentAgent itself is intentionally left unchanged (used
elsewhere - API routers, the legacy per-event simulation coordinator - which
should keep their existing always-compute behavior); this orchestrator is
wired only into OperationalRefreshOrchestrator, the demo/production refresh
path this performance pass targets.
"""
from __future__ import annotations

from datetime import datetime
import logging

from src.agents.analysis.fire_severity_assessment_agent import FireSeverityAssessmentAgent
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_input_result import FireSeverityInputResult
from src.models.fire_severity_input_status import FireSeverityInputStatus
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.services.fire_severity.fire_severity_input_service import FireSeverityInputService

logger = logging.getLogger(__name__)


class FireSeverityRefreshOrchestrator:
    """Reevaluate current severity inputs and avoid duplicate assessment rows."""

    def __init__(
        self,
        *,
        input_service: FireSeverityInputService,
        assessment_agent: FireSeverityAssessmentAgent,
        assessment_repository: FireSeverityAssessmentRepository,
    ) -> None:
        self._input_service = input_service
        self._assessment_agent = assessment_agent
        self._assessment_repository = assessment_repository

    def refresh(self, fire_event_id: int, assessed_at: datetime) -> StoredFireSeverityAssessment:
        """Reuse the latest VALID assessment when its inputs are provably
        unchanged; otherwise delegate to FireSeverityAssessmentAgent.assess()
        exactly as before (full compute + persist). Fetches the FireEvent by
        id - unchanged behavior/signature for existing callers."""
        self._validate_request(fire_event_id, assessed_at)
        input_result = self._input_service.prepare_input(fire_event_id=fire_event_id, as_of=assessed_at)
        return self._refresh_from_input_result(
            fire_event_id=fire_event_id, assessed_at=assessed_at, input_result=input_result
        )

    def refresh_for_event(self, stored_event, assessed_at: datetime) -> StoredFireSeverityAssessment:
        """Same reuse decision as refresh(), but for an ALREADY-LOADED
        StoredFireEvent (performance pass: avoids a redundant FireEvent
        fetch within one OperationalRefreshOrchestrator cycle)."""
        self._validate_request(stored_event.id, assessed_at)
        input_result = self._input_service.prepare_input_for_event(stored_event, assessed_at)
        return self._refresh_from_input_result(
            fire_event_id=stored_event.id, assessed_at=assessed_at, input_result=input_result
        )

    def _refresh_from_input_result(
        self, *, fire_event_id: int, assessed_at: datetime, input_result: FireSeverityInputResult
    ) -> StoredFireSeverityAssessment:
        if input_result.status is FireSeverityInputStatus.READY:
            latest = self._assessment_repository.get_latest_for_event_as_of(fire_event_id, assessed_at)
            if latest is not None and self._is_reusable(latest, input_result):
                logger.info(
                    "Reused fire-severity assessment %s for FireEvent %s - evidence and vegetation unchanged.",
                    latest.assessment_id,
                    fire_event_id,
                )
                return latest

        # Performance pass: input_result was already prepared above (weather/
        # satellite lookups + a real Copernicus vegetation call) just to make
        # the reuse decision - assess_from_input_result() persists directly
        # from it instead of assess()/assess_for_event() re-preparing the
        # identical input from scratch.
        return self._assessment_agent.assess_from_input_result(input_result, assessed_at)

    @staticmethod
    def _is_reusable(latest: StoredFireSeverityAssessment, input_result: FireSeverityInputResult) -> bool:
        """True only when every input that could affect the calculator's
        output is provably identical to what produced `latest`. Fails open
        (returns False) on any uncertainty - a mismatch, a non-VALID latest
        assessment, or a methodology/version bump always forces a fresh
        compute+persist, never a reuse."""
        if latest.assessment.status is not FireSeverityAssessmentStatus.VALID:
            return False
        if latest.assessment.methodology != FIRE_SEVERITY_METHODOLOGY_NAME:
            return False
        if latest.assessment.methodology_version != FIRE_SEVERITY_METHODOLOGY_VERSION:
            return False
        if set(latest.weather_observation_ids) != set(input_result.weather_observation_ids):
            return False
        if set(latest.satellite_hotspot_ids) != set(input_result.satellite_hotspot_ids):
            return False
        if latest.selected_frp_hotspot_id != input_result.selected_frp_hotspot_id:
            return False

        vegetation = input_result.vegetation_data
        if vegetation is None:
            return latest.assessment.vegetation_source is None
        return (
            latest.assessment.vegetation_source == vegetation.source
            and latest.assessment.vegetation_dataset_year == vegetation.dataset_year
            and latest.assessment.vegetation_radius_km == vegetation.radius_km
            and latest.assessment.vegetation_dominant_land_cover == vegetation.dominant_land_cover
            and latest.assessment.vegetation_fuel_score == vegetation.fuel_score
        )

    @staticmethod
    def _validate_request(fire_event_id: int, assessed_at: datetime) -> None:
        if isinstance(fire_event_id, bool) or not isinstance(fire_event_id, int) or fire_event_id <= 0:
            raise ValueError(f"fire_event_id must be a positive integer, got {fire_event_id!r}")
        if not isinstance(assessed_at, datetime) or assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {assessed_at!r}")
