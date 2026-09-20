"""Read-only Operations Activity detail service (Task A5).

OperationsActivityQueryService answers exactly one question: "I have an
activity_type and an entity_id; give me its persisted details." It is a thin
adapter/facade over already-existing, already-tested repository/query-
service read methods - it introduces NO new SQL and NO new repository
capability; every method it calls below already existed before this task:

    FIRE_DANGER          -> FireDangerQueryService.get_assessment_detail()      (Task A4)
    SATELLITE_HOTSPOT    -> SatelliteHotspotRepository.get_by_id()
    NEWS_REPORT          -> NewsRepository.get_by_id()
    FIRE_EVENT           -> FireEventRepository.get_by_id()
                            + FireSeverityAssessmentRepository.get_latest_for_event()
    FIRE_SEVERITY        -> FireSeverityAssessmentRepository.get_by_id()
    GLOBAL_PLANNING_RUN  -> GlobalPlanningRunRepository.get_by_id() + get_members()
    WEATHER_CONDITIONS   -> FireDangerAssessmentRepository.get_by_id()
                            + WeatherRepository.get_observations_by_ids() -
                            entity_id IS the FireDangerAssessmentDB id (no
                            separate WeatherAlert entity exists); no FFWI
                            recalculation, just re-materializing the already-
                            traced observation rows

It never runs FireDangerAssessmentAgent/FFWICalculator, FireDetectionAgent/
FireDetectionCalculator, FireSeverityAssessmentAgent, a spread predictor,
SimulationRefreshCoordinator, routing, or the Global Optimizer/GA - only
plain repository reads, composed and reshaped into the Task A5 envelope
(src/models/operations_activity.py). It never writes anything.

occurred_at mapping (Task A5, Part 11) - the most semantically appropriate
persisted timestamp per type, never "now":
    FIRE_DANGER          -> assessment.assessed_at
    SATELLITE_HOTSPOT    -> hotspot.detected_at (persisted as-is; FIRMS
                             acquisition timestamps are stored naive - see
                             SatelliteHotspotMapper's own TODO - this method
                             does not fabricate a timezone for it)
    NEWS_REPORT          -> NewsRepository's own observed_at (published_at,
                             falling back to fetched_at - already computed
                             and made timezone-aware by NewsRepository)
    FIRE_EVENT           -> event.detected_at
    FIRE_SEVERITY        -> assessment.assessed_at
    GLOBAL_PLANNING_RUN  -> run.completed_at if the run has finished,
                             otherwise run.started_at (a RUNNING/FAILED-
                             before-completion run has no completed_at)

location mapping - never fabricated for an entity without reliable
coordinates (Task A5, Part 4):
    FIRE_DANGER          -> the assessment area's center
    SATELLITE_HOTSPOT    -> the hotspot's own lat/lon (always present)
    NEWS_REPORT          -> the report's lat/lon, only when both are
                             persisted (geocoding is not performed here)
    FIRE_EVENT           -> the event's own lat/lon (always present)
    FIRE_SEVERITY        -> None - FireSeverityAssessment persists no
                             coordinates of its own (only fire_event_id);
                             a consumer wanting a map pin should follow
                             fire_event_id to the FIRE_EVENT activity
                             instead of this method inventing one
    GLOBAL_PLANNING_RUN  -> None - a run spans multiple FireEvents/areas,
                             so no single coordinate is meaningful

title mapping - concise, neutral, never inventing geography or a
confirmed-fire claim not backed by persisted status (Task A5, Part 12):
    FIRE_DANGER          -> "Fire Danger Assessment - {area_name}" (area_name
                             is real, persisted area identity - see A4)
    SATELLITE_HOTSPOT    -> "Satellite Hotspot" (evidence, not a confirmed fire)
    NEWS_REPORT          -> the report's own persisted title/headline
    FIRE_EVENT           -> "Fire Event #{id}" (no area name - FireEvent
                             persists none, see src/models/fire_event.py)
    FIRE_SEVERITY        -> "Fire Severity Assessment"
    GLOBAL_PLANNING_RUN  -> "Global Response Plan"
"""
from __future__ import annotations

from datetime import datetime

from src.models.fire_evidence_type import FireEvidenceType
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireEventActivityDetails,
    FireEventEvidenceRefs,
    FireEventSeverityReference,
    FireSeverityActivityDetail,
    FireSeverityActivityDetails,
    GlobalPlanningRunActivityDetail,
    GlobalPlanningRunActivityDetails,
    GlobalPlanningRunMemberSummary,
    NewsReportActivityDetail,
    OperationsActivityDetail,
    OperationsActivityLocation,
    OperationsActivityType,
    SatelliteHotspotActivityDetail,
    WeatherConditionsActivityDetail,
    WeatherConditionsActivityDetails,
    WeatherConditionsStationReading,
)
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService


class OperationsActivityQueryService:
    """Adapter/facade dispatching one activity_type+entity_id request to the
    right already-existing read method and reshaping its result into the
    Task A5 envelope. Owns no persistence of its own."""

    def __init__(
        self,
        *,
        fire_danger_query_service: FireDangerQueryService | None = None,
        satellite_hotspot_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        fire_danger_assessment_repository: FireDangerAssessmentRepository | None = None,
        weather_repository: WeatherRepository | None = None,
    ) -> None:
        self._fire_danger_query_service = fire_danger_query_service or FireDangerQueryService()
        self._satellite_hotspot_repository = satellite_hotspot_repository or SatelliteHotspotRepository()
        self._news_repository = news_repository or NewsRepository()
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._fire_danger_assessment_repository = (
            fire_danger_assessment_repository or FireDangerAssessmentRepository()
        )
        self._weather_repository = weather_repository or WeatherRepository()

    def get_activity_detail(
        self,
        activity_type: OperationsActivityType,
        entity_id: int,
    ) -> OperationsActivityDetail | None:
        """Return the persisted detail for one activity, or None if entity_id does not exist.

        `activity_type` is trusted to already be a validated
        OperationsActivityType (the API layer enforces this via path
        enum parsing) - this dispatcher does not accept arbitrary strings.
        """
        self._validate_entity_id(entity_id)
        handler = self._DISPATCH[activity_type]
        return handler(self, entity_id)

    # -- per-type handlers ---------------------------------------------

    def _get_fire_danger_detail(self, entity_id: int) -> FireDangerActivityDetail | None:
        detail = self._fire_danger_query_service.get_assessment_detail(entity_id)
        if detail is None:
            return None
        return FireDangerActivityDetail(
            entity_id=entity_id,
            occurred_at=detail.assessed_at,
            title=f"Fire Danger Assessment - {detail.area_name}",
            location=OperationsActivityLocation(latitude=detail.area_latitude, longitude=detail.area_longitude),
            details=detail,
        )

    def _get_satellite_hotspot_detail(self, entity_id: int) -> SatelliteHotspotActivityDetail | None:
        stored = self._satellite_hotspot_repository.get_by_id(entity_id)
        if stored is None:
            return None
        hotspot = stored.hotspot
        return SatelliteHotspotActivityDetail(
            entity_id=entity_id,
            occurred_at=hotspot.detected_at,
            title="Satellite Hotspot",
            location=OperationsActivityLocation(latitude=hotspot.latitude, longitude=hotspot.longitude),
            details=hotspot,
        )

    def _get_news_report_detail(self, entity_id: int) -> NewsReportActivityDetail | None:
        stored = self._news_repository.get_by_id(entity_id)
        if stored is None:
            return None
        report = stored.report
        location = (
            OperationsActivityLocation(latitude=report.latitude, longitude=report.longitude)
            if report.latitude is not None and report.longitude is not None
            else None
        )
        return NewsReportActivityDetail(
            entity_id=entity_id,
            occurred_at=stored.observed_at,
            title=report.title,
            location=location,
            details=report,
        )

    def _get_fire_event_detail(self, entity_id: int) -> FireEventActivityDetail | None:
        stored = self._fire_event_repository.get_by_id(entity_id)
        if stored is None:
            return None
        event = stored.event

        satellite_ids = tuple(
            ref.evidence_id for ref in stored.supporting_evidence if ref.evidence_type is FireEvidenceType.SATELLITE
        )
        news_ids = tuple(
            ref.evidence_id for ref in stored.supporting_evidence if ref.evidence_type is FireEvidenceType.NEWS
        )

        stored_severity = self._fire_severity_assessment_repository.get_latest_for_event(entity_id)
        latest_severity = None
        if stored_severity is not None:
            severity = stored_severity.assessment
            latest_severity = FireEventSeverityReference(
                assessment_id=stored_severity.assessment_id,
                status=severity.status,
                score=severity.score,
                level=severity.level,
                assessed_at=severity.assessed_at,
            )

        return FireEventActivityDetail(
            entity_id=entity_id,
            occurred_at=event.detected_at,
            title=f"Fire Event #{entity_id}",
            location=OperationsActivityLocation(latitude=event.latitude, longitude=event.longitude),
            details=FireEventActivityDetails(
                fire_event=event,
                evidence=FireEventEvidenceRefs(satellite_hotspot_ids=satellite_ids, news_report_ids=news_ids),
                latest_severity=latest_severity,
                created_at=stored.created_at,
            ),
        )

    def _get_fire_severity_detail(self, entity_id: int) -> FireSeverityActivityDetail | None:
        stored = self._fire_severity_assessment_repository.get_by_id(entity_id)
        if stored is None:
            return None
        return FireSeverityActivityDetail(
            entity_id=entity_id,
            occurred_at=stored.assessment.assessed_at,
            title="Fire Severity Assessment",
            location=None,
            details=FireSeverityActivityDetails(
                assessment=stored.assessment,
                weather_observation_ids=stored.weather_observation_ids,
                satellite_hotspot_ids=stored.satellite_hotspot_ids,
                selected_frp_hotspot_id=stored.selected_frp_hotspot_id,
            ),
        )

    def _get_global_planning_run_detail(self, entity_id: int) -> GlobalPlanningRunActivityDetail | None:
        stored_run = self._global_planning_run_repository.get_by_id(entity_id)
        if stored_run is None:
            return None
        run = stored_run.run
        stored_members = self._global_planning_run_repository.get_members(entity_id)
        members = tuple(
            GlobalPlanningRunMemberSummary(
                fire_event_id=stored_member.member.fire_event_id,
                event_order=stored_member.member.event_order,
                result_status=stored_member.member.result_status,
                response_plan_id=stored_member.member.response_plan_id,
                severity_level=stored_member.member.severity_level,
                assigned_resources=stored_member.member.assigned_resources,
                coverage_score=stored_member.member.coverage_score,
                average_eta_seconds=stored_member.member.average_eta_seconds,
            )
            for stored_member in stored_members
        )
        occurred_at: datetime = run.completed_at if run.completed_at is not None else run.started_at

        return GlobalPlanningRunActivityDetail(
            entity_id=entity_id,
            occurred_at=occurred_at,
            title="Global Response Plan",
            location=None,
            details=GlobalPlanningRunActivityDetails(run=run, members=members),
        )

    def _get_weather_conditions_detail(self, entity_id: int) -> WeatherConditionsActivityDetail | None:
        """entity_id IS the FireDangerAssessmentDB id - there is no separate
        persisted WeatherAlert entity (Part G). Re-materializes the exact
        observation rows already traced for that assessment; performs no
        FFWI recalculation and no fresh observation selection."""
        stored = self._fire_danger_assessment_repository.get_by_id(entity_id)
        if stored is None or stored.assessment.level is None:
            return None
        assessment = stored.assessment

        stored_observations = self._weather_repository.get_observations_by_ids(stored.observation_ids)
        observations_by_id = {stored_obs.observation_id: stored_obs for stored_obs in stored_observations}
        readings = tuple(
            WeatherConditionsStationReading(
                station_id=observations_by_id[observation_id].station_id,
                station_name=observations_by_id[observation_id].station.name,
                observation_id=observation_id,
                observed_at=observations_by_id[observation_id].observation.timestamp,
                temperature=observations_by_id[observation_id].observation.temperature,
                relative_humidity=observations_by_id[observation_id].observation.relative_humidity,
                wind_speed=observations_by_id[observation_id].observation.wind_speed,
                wind_gust=observations_by_id[observation_id].observation.wind_gust,
            )
            for observation_id in stored.observation_ids
            if observation_id in observations_by_id
        )

        return WeatherConditionsActivityDetail(
            entity_id=entity_id,
            occurred_at=(max(reading.observed_at for reading in readings) if readings else assessment.assessed_at),
            title=f"Weather Conditions - {assessment.area_name}",
            location=OperationsActivityLocation(
                latitude=assessment.area_latitude, longitude=assessment.area_longitude
            ),
            details=WeatherConditionsActivityDetails(
                fire_danger_assessment_id=entity_id,
                area_name=assessment.area_name,
                fire_danger_level=assessment.level,
                assessed_at=assessment.assessed_at,
                readings=readings,
            ),
        )

    _DISPATCH = {
        OperationsActivityType.FIRE_DANGER: _get_fire_danger_detail,
        OperationsActivityType.SATELLITE_HOTSPOT: _get_satellite_hotspot_detail,
        OperationsActivityType.NEWS_REPORT: _get_news_report_detail,
        OperationsActivityType.FIRE_EVENT: _get_fire_event_detail,
        OperationsActivityType.FIRE_SEVERITY: _get_fire_severity_detail,
        OperationsActivityType.GLOBAL_PLANNING_RUN: _get_global_planning_run_detail,
        OperationsActivityType.WEATHER_CONDITIONS: _get_weather_conditions_detail,
    }

    @staticmethod
    def _validate_entity_id(entity_id: object) -> None:
        if isinstance(entity_id, bool) or not isinstance(entity_id, int) or entity_id <= 0:
            raise ValueError(f"entity_id must be a positive integer, got {entity_id!r}")
