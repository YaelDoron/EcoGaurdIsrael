"""Read-only FireEvent-details aggregation service (Epic 6, US 6.2, Task 1).

EventDetailsService assembles US 6.2's `EventDetailsResult` response DTO
purely by fetching data that Fire Detection, Fire Severity Assessment, Fire
Spread Prediction, Response Target Generation, and Response Planning have
already persisted. It contains no detection, no severity/danger/spread
calculation, no target generation, no routing, and no genetic-algorithm
optimization of its own, and it never writes to any repository - a strict
read/aggregate/map step, matching `ActiveFireEventsService`'s (US 6.1) and
`ResponsePlanDetailsService`'s (US 5.5) own read-only precedent. The current
response plan is not re-derived here: `ResponsePlanDetailsService` is reused
as-is (composition, not duplication) for that part of the assembly.

This service builds `src.api.schemas.event_details` Pydantic DTOs directly
(no intermediate `src.models` dataclass layer) - see that module's docstring
for why, and for the `danger` field's current always-`None` contract.

Current-state spread contract (critical): for each supported horizon, the
latest persisted `FireSpreadPrediction` is used exactly as stored. When its
status is `insufficient_data` or `inactive_event`, `cells` is `[]` - this is
guaranteed by `FireSpreadPrediction`'s own domain validation (a non-VALID
prediction can never carry cells), and this service never substitutes an
older VALID run to avoid showing an empty map layer. A horizon with no
persisted prediction at all is simply omitted from `spread_predictions`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.api.schemas.event_details import (
    BaselineComparisonResponse,
    CurrentResponsePlanResponse,
    DetectionEvidenceResponse,
    EventDetailsResult,
    FireEventMLAssessmentResponse,
    FireEventSummaryResponse,
    FirefightingResourceResponse,
    FireStationResponse,
    NewsEvidenceResponse,
    ResponseActionResponse,
    ResponseTargetResponse,
    SatelliteEvidenceResponse,
    SeverityAssessmentResponse,
    SpreadPredictionCellResponse,
    SpreadPredictionResponse,
    StationAllocationResponse,
    StationSummaryResponse,
)
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_event_ml_assessment import FireEventMLAssessment
from src.models.fire_spread_prediction import SUPPORTED_HORIZON_MINUTES
from src.models.resource_status import ResourceStatus
from src.models.response_plan_details import ResponseActionDetails, ResponsePlanDetails
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import (
    FireSpreadPredictionRepository,
    StoredFireSpreadPrediction,
)
from src.repositories.fire_station_repository import FireStationRepository
from src.repositories.firefighting_resource_repository import FirefightingResourceRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.services.response_planning.response_plan_details_service import ResponsePlanDetailsService

# ML Task 6 (API exposure): `FireEventMLAssessment.ml_failure_reason` is an
# internal diagnostic string - `MLFireDetectionClassifier` sometimes embeds
# raw exception text in it (e.g. "ML model load failed: [Errno 2] ... 'C:\\
# ...\\fire_detection_logistic_v3.joblib'"), which can include filesystem
# paths. That persisted text is intentionally left untouched (it is useful
# server-side/in logs), but it must never be forwarded to the public API
# verbatim. This categorizes the known failure-text prefixes (from
# src/calculators/fire_detection/fire_detection_ml_classifier.py) into a
# short, safe, generic reason - any unrecognized text (including future
# failure-text changes) falls back to a fully generic message rather than
# risk leaking something new.
_ML_FAILURE_REASON_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("ML model load failed", "ML model artifact could not be loaded."),
    ("feature extraction failed", "ML feature extraction failed for this event's evidence."),
    ("predict_proba failed", "ML inference failed for this event's evidence."),
    ("ML classifier not configured", "ML classifier is not configured for this deployment."),
)
_DEFAULT_SANITIZED_ML_FAILURE_REASON = "ML assessment unavailable."


def _sanitize_ml_failure_reason(raw_reason: str | None) -> str | None:
    if raw_reason is None:
        return None
    for prefix, sanitized in _ML_FAILURE_REASON_CATEGORIES:
        if raw_reason.startswith(prefix):
            return sanitized
    return _DEFAULT_SANITIZED_ML_FAILURE_REASON


class EventDetailsService:
    """Assemble `EventDetailsResult` from already-persisted per-event data."""

    def __init__(
        self,
        *,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        fire_spread_prediction_repository: FireSpreadPredictionRepository | None = None,
        response_target_repository: ResponseTargetRepository | None = None,
        fire_station_repository: FireStationRepository | None = None,
        firefighting_resource_repository: FirefightingResourceRepository | None = None,
        satellite_hotspot_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
        response_plan_details_service: ResponsePlanDetailsService | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._fire_spread_prediction_repository = (
            fire_spread_prediction_repository or FireSpreadPredictionRepository()
        )
        self._response_target_repository = response_target_repository or ResponseTargetRepository()
        self._fire_station_repository = fire_station_repository or FireStationRepository()
        self._firefighting_resource_repository = (
            firefighting_resource_repository or FirefightingResourceRepository()
        )
        self._satellite_hotspot_repository = satellite_hotspot_repository or SatelliteHotspotRepository()
        self._news_repository = news_repository or NewsRepository()
        self._response_plan_details_service = response_plan_details_service or ResponsePlanDetailsService()

    def get_event_details(
        self,
        fire_event_id: int,
        *,
        as_of: datetime | None = None,
    ) -> EventDetailsResult | None:
        """Return the FireEvent details snapshot, or None if the FireEvent does not exist.

        `as_of` only stamps the returned snapshot's `as_of` field and is the
        cutoff used to look up the latest response-target set; it does not
        change which FireEvent is looked up. Pass an explicit value in tests
        for a deterministic snapshot; defaults to the current time otherwise.
        """
        self._validate_positive_int("fire_event_id", fire_event_id)
        snapshot_time = as_of if as_of is not None else datetime.now(timezone.utc)
        self._validate_aware_datetime("as_of", snapshot_time)

        stored_event = self._fire_event_repository.get_by_id(fire_event_id)
        if stored_event is None:
            return None

        db_stations = self._fire_station_repository.get_all_stations()
        station_ids = [str(db_station.id) for db_station in db_stations]
        db_resources = self._firefighting_resource_repository.get_resources_for_stations(station_ids)
        plan_details = self._response_plan_details_service.get_current_plan_details(fire_event_id)

        return EventDetailsResult(
            as_of=snapshot_time,
            fire_event=self._to_fire_event_response(stored_event.id, stored_event.event),
            severity=self._load_severity(fire_event_id),
            ml_assessment=self._load_ml_assessment(fire_event_id),
            danger=None,
            detection_evidence=self._load_detection_evidence(stored_event.supporting_evidence),
            spread_predictions=self._load_spread_predictions(fire_event_id),
            targets=self._load_targets(fire_event_id, snapshot_time),
            stations=self._to_station_responses(db_stations),
            resources=self._to_resource_responses(db_resources),
            station_summaries=self._build_station_summaries(fire_event_id, db_stations, db_resources, plan_details),
            current_response_plan=(
                self._to_current_response_plan_response(plan_details) if plan_details is not None else None
            ),
        )

    def _load_severity(self, fire_event_id: int) -> SeverityAssessmentResponse | None:
        stored_severity = self._fire_severity_assessment_repository.get_latest_for_event(fire_event_id)
        if stored_severity is None:
            return None
        assessment = stored_severity.assessment
        return SeverityAssessmentResponse(
            assessment_id=stored_severity.assessment_id,
            status=assessment.status,
            score=assessment.score,
            level=assessment.level,
            assessed_at=assessment.assessed_at,
        )

    def _load_ml_assessment(self, fire_event_id: int) -> FireEventMLAssessmentResponse | None:
        """Return the FireEvent's persisted runtime ML/decision trace, or
        None when it was never evaluated with ML (RULE_ONLY mode, or a
        FireEvent created before ML Task 5's integration) - a pure read via
        the repository's existing `get_ml_assessment`, never ML inference."""
        assessment = self._fire_event_repository.get_ml_assessment(fire_event_id)
        if assessment is None:
            return None
        return self._to_ml_assessment_response(assessment)

    @staticmethod
    def _to_ml_assessment_response(assessment: FireEventMLAssessment) -> FireEventMLAssessmentResponse:
        return FireEventMLAssessmentResponse(
            available=assessment.ml_available,
            mode=assessment.decision_mode,
            rule_status=assessment.rule_status,
            rule_confidence=assessment.rule_confidence,
            model_score=assessment.ml_probability,
            agreement=assessment.agreement,
            model_name=assessment.ml_model_name,
            model_version=assessment.ml_model_version,
            feature_schema_version=assessment.ml_feature_schema_version,
            failure_reason=_sanitize_ml_failure_reason(assessment.ml_failure_reason),
            updated_at=assessment.updated_at,
        )

    def _load_detection_evidence(self, supporting_evidence) -> DetectionEvidenceResponse:  # noqa: ANN001
        satellite_evidence: list[SatelliteEvidenceResponse] = []
        news_evidence: list[NewsEvidenceResponse] = []
        for evidence_ref in supporting_evidence:
            if evidence_ref.evidence_type is FireEvidenceType.SATELLITE:
                stored_hotspot = self._satellite_hotspot_repository.get_by_id(evidence_ref.evidence_id)
                if stored_hotspot is not None:
                    satellite_evidence.append(self._to_satellite_evidence_response(stored_hotspot))
            elif evidence_ref.evidence_type is FireEvidenceType.NEWS:
                stored_report = self._news_repository.get_by_id(evidence_ref.evidence_id)
                if stored_report is not None:
                    news_evidence.append(self._to_news_evidence_response(stored_report))
        return DetectionEvidenceResponse(satellite=satellite_evidence, news=news_evidence)

    @staticmethod
    def _to_satellite_evidence_response(stored_hotspot) -> SatelliteEvidenceResponse:  # noqa: ANN001
        hotspot = stored_hotspot.hotspot
        return SatelliteEvidenceResponse(
            id=stored_hotspot.id,
            detected_at=hotspot.detected_at,
            latitude=hotspot.latitude,
            longitude=hotspot.longitude,
            confidence=hotspot.confidence,
            frp=hotspot.frp,
            brightness=hotspot.brightness,
            satellite=hotspot.satellite,
            instrument=hotspot.instrument,
            day_night=hotspot.day_night,
        )

    @staticmethod
    def _to_news_evidence_response(stored_report) -> NewsEvidenceResponse:  # noqa: ANN001
        report = stored_report.report
        return NewsEvidenceResponse(
            id=stored_report.id,
            title=report.title,
            summary=report.summary,
            source=report.source_feed,
            observed_at=stored_report.observed_at,
            location_name=report.location_name,
            latitude=report.latitude,
            longitude=report.longitude,
        )

    def _load_spread_predictions(self, fire_event_id: int) -> list[SpreadPredictionResponse]:
        predictions: list[SpreadPredictionResponse] = []
        for horizon_minutes in SUPPORTED_HORIZON_MINUTES:
            stored_prediction = self._fire_spread_prediction_repository.get_latest_for_event_and_horizon(
                fire_event_id, horizon_minutes
            )
            if stored_prediction is None:
                continue
            predictions.append(self._to_spread_prediction_response(stored_prediction))
        return predictions

    @staticmethod
    def _to_spread_prediction_response(
        stored_prediction: StoredFireSpreadPrediction,
    ) -> SpreadPredictionResponse:
        prediction = stored_prediction.prediction
        return SpreadPredictionResponse(
            horizon_minutes=prediction.horizon_minutes,
            status=prediction.status,
            predicted_at=prediction.predicted_at,
            cells=[
                SpreadPredictionCellResponse(
                    latitude=cell.latitude,
                    longitude=cell.longitude,
                    spread_probability=cell.spread_probability,
                    spread_risk_score=cell.spread_risk_score,
                    reached_step=cell.reached_step,
                    reached_minutes=cell.reached_minutes,
                )
                for cell in prediction.cells
            ],
        )

    def _load_targets(self, fire_event_id: int, as_of: datetime) -> list[ResponseTargetResponse]:
        stored_target_set = self._response_target_repository.get_latest_for_event_as_of(fire_event_id, as_of)
        if stored_target_set is None:
            return []
        return [
            ResponseTargetResponse(
                target_order=stored_target.target_order,
                target_type=stored_target.target.target_type,
                latitude=stored_target.target.latitude,
                longitude=stored_target.target.longitude,
                priority_score=stored_target.target.priority_score,
                prediction_horizon_minutes=stored_target.target.prediction_horizon_minutes,
            )
            for stored_target in stored_target_set.targets
        ]

    @staticmethod
    def _to_station_responses(db_stations) -> list[FireStationResponse]:  # noqa: ANN001
        return [
            FireStationResponse(
                station_id=str(db_station.id),
                name=db_station.name,
                latitude=db_station.latitude,
                longitude=db_station.longitude,
                station_type=db_station.station_type,
                address=db_station.address,
            )
            for db_station in db_stations
        ]

    @staticmethod
    def _to_resource_responses(db_resources) -> list[FirefightingResourceResponse]:  # noqa: ANN001
        return [
            FirefightingResourceResponse(
                resource_id=str(db_resource.id),
                station_id=str(db_resource.station_id),
                status=db_resource.status,
            )
            for db_resource in db_resources
        ]

    @staticmethod
    def _build_station_summaries(
        fire_event_id: int,
        db_stations,  # noqa: ANN001
        db_resources,  # noqa: ANN001
        plan_details: ResponsePlanDetails | None,
    ) -> list[StationSummaryResponse]:
        resources_by_station_id: dict[str, list] = {str(db_station.id): [] for db_station in db_stations}
        for db_resource in db_resources:
            resources_by_station_id.setdefault(str(db_resource.station_id), []).append(db_resource)

        allocations_by_station_id: dict[str, list[StationAllocationResponse]] = {}
        if plan_details is not None:
            for action in plan_details.actions:
                allocations_by_station_id.setdefault(action.station_id, []).append(
                    StationAllocationResponse(
                        resource_id=action.resource_id,
                        fire_event_id=fire_event_id,
                        response_plan_id=plan_details.plan_id,
                    )
                )

        summaries: list[StationSummaryResponse] = []
        for db_station in db_stations:
            station_id = str(db_station.id)
            station_resources = resources_by_station_id.get(station_id, [])
            summaries.append(
                StationSummaryResponse(
                    station_id=station_id,
                    total_resources=len(station_resources),
                    available=sum(
                        1 for db_resource in station_resources if db_resource.status is ResourceStatus.AVAILABLE
                    ),
                    assigned_status=sum(
                        1 for db_resource in station_resources if db_resource.status is ResourceStatus.ASSIGNED
                    ),
                    unavailable=sum(
                        1 for db_resource in station_resources if db_resource.status is ResourceStatus.UNAVAILABLE
                    ),
                    current_global_plan_allocations=allocations_by_station_id.get(station_id, []),
                )
            )
        return summaries

    @classmethod
    def _to_current_response_plan_response(cls, plan_details: ResponsePlanDetails) -> CurrentResponsePlanResponse:
        return CurrentResponsePlanResponse(
            plan_id=plan_details.plan_id,
            generated_at=plan_details.generated_at,
            methodology=plan_details.methodology,
            methodology_version=plan_details.methodology_version,
            plan_score=plan_details.plan_score,
            coverage_score=plan_details.coverage_score,
            average_eta_seconds=plan_details.average_eta_seconds,
            actions=[cls._to_response_action_response(action) for action in plan_details.actions],
            uncovered_target_ids=list(plan_details.uncovered_target_ids),
            baseline_comparison=(
                BaselineComparisonResponse(
                    baseline_score=plan_details.baseline_comparison.baseline_score,
                    baseline_coverage_score=plan_details.baseline_comparison.baseline_coverage_score,
                    baseline_average_eta_seconds=plan_details.baseline_comparison.baseline_average_eta_seconds,
                    score_difference=plan_details.baseline_comparison.score_difference,
                    improvement_percentage=plan_details.baseline_comparison.improvement_percentage,
                )
                if plan_details.baseline_comparison is not None
                else None
            ),
        )

    @staticmethod
    def _to_response_action_response(action: ResponseActionDetails) -> ResponseActionResponse:
        return ResponseActionResponse(
            resource_id=action.resource_id,
            station_id=action.station_id,
            response_target_id=action.response_target_id,
            target_type=action.target_type,
            target_priority=action.target_priority,
            eta_seconds=action.eta_seconds,
            route_distance_meters=action.route_distance_meters,
            node_path=list(action.node_path) if action.node_path is not None else None,
        )

    @staticmethod
    def _to_fire_event_response(fire_event_id: int, event) -> FireEventSummaryResponse:  # noqa: ANN001
        return FireEventSummaryResponse(
            fire_event_id=fire_event_id,
            status=event.status,
            latitude=event.latitude,
            longitude=event.longitude,
            detection_confidence=event.detection_confidence,
            detected_at=event.detected_at,
            updated_at=event.updated_at,
            methodology=event.methodology,
            methodology_version=event.methodology_version,
        )

    @staticmethod
    def _validate_positive_int(field_name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
