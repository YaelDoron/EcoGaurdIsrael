"""Read-only Operations Overview query service (Task A6).

OperationsOverviewQueryService assembles ONE coherent dashboard snapshot -
current Fire Danger areas, currently active FireEvents (each with latest
persisted Severity), a chronological Activity Feed, and the A3
simulation-control summary - purely by composing already-existing,
already-tested read services/repositories:

    Fire Danger areas   -> FireDangerQueryService.get_latest_for_all_areas()      (A4, unchanged)
    Active fires + Severity -> ActiveFireEventsService.get_active_events()         (US 6.1, unchanged:
                               canonical active = SUSPECTED/CONFIRMED, batched
                               latest-Severity via FireSeverityAssessmentRepository.get_latest_for_events)
    Simulation summary  -> the INJECTED, process-local A3 SimulationRunManager's
                            own get_current_snapshot() (never a fresh instance -
                            see __init__'s docstring)
    Activity Feed        -> seven bounded "get_recent(limit)"-shaped repository
                            reads, merged/sorted/trimmed in application memory
                            (see _build_activity_feed's docstring for the
                            exact strategy)

It never runs FireDangerAssessmentAgent/FFWICalculator, FireDetectionAgent/
FireDetectionCalculator, FireSeverityAssessmentAgent, a spread predictor,
SimulationRefreshCoordinator, routing/Dijkstra, or the Global Optimizer/GA -
only plain repository/service reads, and it never writes anything.
WEATHER_CONDITIONS is no exception: it only re-averages the exact weather
observations FireDangerAssessmentRepository already traced for a HIGH+
assessment - never a fresh FFWI calculation.

occurred_at mapping for the Activity Feed (Task A6, Part 13) - identical to
A5's own mapping (src/services/operations/operations_activity_query_service.py),
so a feed item's timestamp always matches what its A5 detail would show:
    FIRE_DANGER          -> assessment.assessed_at
    SATELLITE_HOTSPOT    -> hotspot.detected_at (TIMESTAMPTZ, tz-aware)
    NEWS_REPORT          -> NewsRepository's own observed_at (already tz-aware)
    FIRE_EVENT           -> event.detected_at
    FIRE_SEVERITY        -> assessment.assessed_at
    GLOBAL_PLANNING_RUN  -> run.completed_at if finished, else run.started_at
    WEATHER_CONDITIONS   -> the latest contributing observation's timestamp

available_at mapping (Task 4 - "when did this activity become available in
EcoGuard", used for the LIVE feed's display/sort timestamp, never a fake
frontend "first seen" time):
    FIRE_DANGER          -> the assessment row's own created_at (DB-insert time)
    WEATHER_CONDITIONS   -> the SAME FireDangerAssessment's created_at (Part H:
                            the two are always derived from one assessment and
                            therefore share one availability instant)
    SATELLITE_HOTSPOT    -> the hotspot row's own created_at (added by this
                            task - see SatelliteHotspotDB.created_at), falling
                            back to detected_at only for a genuine
                            pre-migration row with no recorded value
    NEWS_REPORT          -> report.fetched_at (NewsRepository's existing
                            ingestion timestamp - already exactly this
                            meaning, no new column needed)
    FIRE_EVENT           -> the event row's own created_at, falling back to
                            detected_at if ever absent (excluded from the
                            visible feed; kept consistent for the shared sort)
    FIRE_SEVERITY        -> the assessment row's own created_at, falling back
                            to assessed_at (excluded from the visible feed)
    GLOBAL_PLANNING_RUN  -> the run row's own created_at, falling back to its
                            occurred_at (excluded from the visible feed)

Activity Feed query strategy (Task A6, Part 16, option B): each of the seven
sources is read via its own bounded `get_recent(activity_limit)`-shaped call
(added to each repository for this task - see e.g.
FireDangerAssessmentRepository.get_recent/get_recent_with_level_in;
SatelliteHotspotRepository already had limit-bounded reads, but none that
also carried the database id the feed needs, so `get_recent` was added
there too) - at most `7 * activity_limit` candidate rows total, never a
full-table scan and never one query per feed item. Candidates are merged, sorted once by
(occurred_at desc, activity_type asc, entity_id desc), and trimmed to
`activity_limit`. This does NOT call
OperationsActivityQueryService.get_activity_detail() per item (Task A6,
Part 25) - detail lookup is a separate, click-triggered concern (A5); this
service only builds small, typed `*ActivityPreview` projections.

FireEvent history limitation (Task A6, Part 14): FireEventDB stores only
the current row per event, with no historical change-audit table. One
persisted FireEvent therefore contributes exactly one feed item (via
get_recent, keyed by detected_at) - this service does not fabricate
separate "FireEvent updated" activities for the same row.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.fire_danger_areas import FireDangerAreaSnapshot
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.operations_activity import OperationsActivityLocation, OperationsActivityType
from src.models.operations_overview import (
    FireDangerActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    NewsReportActivityPreview,
    OperationsActivityFeed,
    OperationsActivityFeedItem,
    OperationsOverviewSnapshot,
    OperationsSimulationSummary,
    SatelliteHotspotActivityPreview,
    WeatherConditionsActivityPreview,
)
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.simulation_control.simulation_run_manager import SimulationRunManager
from src.services.weather.weather_conditions_query_service import WeatherConditionsQueryService
from src.utils.geo import resolve_nearest_containing_area_name

MIN_ACTIVITY_LIMIT = 1
MAX_ACTIVITY_LIMIT = 100
DEFAULT_ACTIVITY_LIMIT = 30

# Every successfully calculated (VALID) Fire Danger level - the Weather
# Conditions Activity Feed signal now appears for EVERY one of these, not
# just HIGH+. Passing all five to get_recent_with_level_in's danger_level.in_()
# filter is equivalent to "status == VALID" (INSUFFICIENT_DATA rows have
# danger_level IS NULL, which never matches any of these five values) - no
# separate INSUFFICIENT_DATA check is needed.
ALL_FIRE_DANGER_LEVELS = (
    FireDangerLevel.LOW,
    FireDangerLevel.MODERATE,
    FireDangerLevel.HIGH,
    FireDangerLevel.VERY_HIGH,
    FireDangerLevel.EXTREME,
)

# Feed sort tie-break order (Part I): plain alphabetical-by-value for every
# type EXCEPT the Weather/Fire Danger causal pair, where WEATHER_CONDITIONS
# must win an exact occurred_at tie over FIRE_DANGER (cause before effect).
# All other relative positions are unchanged from the previous pure
# alphabetical tie-break.
_ACTIVITY_TYPE_TIE_BREAK_ORDER = {
    OperationsActivityType.WEATHER_CONDITIONS: "0_weather_conditions",
    OperationsActivityType.FIRE_DANGER: "1_fire_danger",
    OperationsActivityType.FIRE_EVENT: "2_fire_event",
    OperationsActivityType.FIRE_SEVERITY: "2_fire_severity",
    OperationsActivityType.GLOBAL_PLANNING_RUN: "2_global_planning_run",
    OperationsActivityType.NEWS_REPORT: "2_news_report",
    OperationsActivityType.SATELLITE_HOTSPOT: "2_satellite_hotspot",
}


class OperationsOverviewQueryService:
    """Assemble the Operations Overview dashboard snapshot from already-persisted/already-computed data."""

    def __init__(
        self,
        *,
        fire_danger_query_service: FireDangerQueryService | None = None,
        active_fire_events_service: ActiveFireEventsService | None = None,
        fire_danger_assessment_repository: FireDangerAssessmentRepository | None = None,
        satellite_hotspot_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
        weather_repository: WeatherRepository | None = None,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
        global_planning_run_repository: GlobalPlanningRunRepository | None = None,
        simulation_run_manager: SimulationRunManager | None = None,
        weather_conditions_query_service: WeatherConditionsQueryService | None = None,
    ) -> None:
        """Wire every collaborator this snapshot composes.

        `simulation_run_manager` is the one exception to "a fresh default
        instance is fine": SimulationRunManager owns in-memory, process-
        local run state (Task A3) - a fresh, unwired instance is always
        IDLE and would silently misreport a real running simulation as "no
        run". Production wiring (src/api/dependencies.py) MUST inject the
        same process-local singleton `get_simulation_run_manager()`
        returns; the default here exists only so this service remains
        constructible (and correctly reports `run=None`) in contexts that
        do not care about simulation state.
        """
        self._fire_danger_query_service = fire_danger_query_service or FireDangerQueryService()
        self._active_fire_events_service = active_fire_events_service or ActiveFireEventsService()
        self._fire_danger_assessment_repository = (
            fire_danger_assessment_repository or FireDangerAssessmentRepository()
        )
        self._satellite_hotspot_repository = satellite_hotspot_repository or SatelliteHotspotRepository()
        self._news_repository = news_repository or NewsRepository()
        self._weather_repository = weather_repository or WeatherRepository()
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )
        self._global_planning_run_repository = global_planning_run_repository or GlobalPlanningRunRepository()
        self._simulation_run_manager = simulation_run_manager or SimulationRunManager()
        # Built from this service's own repositories by default, so the
        # Weather Conditions rows read exactly the same data as before.
        self._weather_conditions_query_service = weather_conditions_query_service or WeatherConditionsQueryService(
            fire_danger_assessment_repository=self._fire_danger_assessment_repository,
            weather_repository=self._weather_repository,
        )

    def get_overview(
        self,
        *,
        activity_limit: int = DEFAULT_ACTIVITY_LIMIT,
        as_of: datetime | None = None,
    ) -> OperationsOverviewSnapshot:
        """Return one dashboard snapshot. Never raises merely because the DB is empty/just reset."""
        self._validate_activity_limit(activity_limit)
        generated_at = as_of if as_of is not None else datetime.now(timezone.utc)
        self._validate_aware_datetime("as_of", generated_at)

        fire_danger_areas = self._fire_danger_query_service.get_latest_for_all_areas(as_of=generated_at).areas

        return OperationsOverviewSnapshot(
            generated_at=generated_at,
            simulation=self._build_simulation_summary(),
            fire_danger_areas=fire_danger_areas,
            active_fires=self._active_fire_events_service.get_active_events(
                as_of=generated_at, fire_danger_areas=fire_danger_areas
            ).items,
            activity_feed=self._build_activity_feed(activity_limit, fire_danger_areas),
        )

    # -- simulation summary ----------------------------------------------

    def _build_simulation_summary(self) -> OperationsSimulationSummary:
        # Imported lazily (not at module top) so a test/deployment can flip
        # the flag between calls without this module having cached a stale
        # value at import time - matches SimulationRunManager.start_run's
        # own lazy-settings-import rationale.
        from src.config.settings import settings

        if not settings.ENABLE_SIMULATION_CONTROL_API:
            return OperationsSimulationSummary(enabled=False, run=None)

        snapshot = self._simulation_run_manager.get_current_snapshot()
        run = snapshot if snapshot.run_id is not None else None
        return OperationsSimulationSummary(enabled=True, run=run)

    # -- activity feed ------------------------------------------------------

    def _build_activity_feed(
        self, limit: int, fire_danger_areas: tuple[FireDangerAreaSnapshot, ...]
    ) -> OperationsActivityFeed:
        candidates: list[OperationsActivityFeedItem] = []
        candidates.extend(self._fire_danger_candidates(limit))
        candidates.extend(self._satellite_candidates(limit, fire_danger_areas))
        candidates.extend(self._news_candidates(limit))
        candidates.extend(self._fire_event_candidates(limit))
        candidates.extend(self._severity_candidates(limit))
        candidates.extend(self._global_planning_candidates(limit))
        candidates.extend(self._weather_conditions_candidates(limit))

        candidates.sort(key=self._feed_sort_key)
        return OperationsActivityFeed(items=tuple(candidates[:limit]), limit=limit)

    @staticmethod
    def _feed_sort_key(item: OperationsActivityFeedItem) -> tuple:
        """available_at DESC, then a type tie-break, then entity_id DESC (Task 4, Part L).

        The LIVE Activity Feed is now ordered by `available_at` (when
        EcoGuard actually persisted/could expose the activity), not by
        `occurred_at` (its source/domain time) - a satellite pass observed a
        few minutes ago must not visually jump ahead of activities EcoGuard
        exposed more recently. `available_at` is required and
        timezone-aware end to end (see OperationsActivityFeedItem.__post_init__);
        this naive-as-UTC guard remains only as defense-in-depth, comparison
        ranking ONLY, never mutating the exposed value - the same convention
        already used by FireDangerAssessmentRepository._ensure_aware_datetime
        elsewhere.

        The type tie-break is normally plain alphabetical-by-value (stable,
        deterministic). WEATHER_CONDITIONS/FIRE_DANGER are the one
        exception: on an exact `available_at` tie (expected now, since both
        share the same assessment's availability instant - Part H),
        WEATHER_CONDITIONS must sort first (i.e. "newer" in this DESC
        ordering) so a same-assessment causal pair reads cause-before-effect
        (Weather input, then its Fire Danger result) - never by mutating
        either timestamp.
        """
        available_at = item.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=timezone.utc)
        return (-available_at.timestamp(), _ACTIVITY_TYPE_TIE_BREAK_ORDER[item.activity_type], -item.entity_id)

    def _fire_danger_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        items = []
        for stored in self._fire_danger_assessment_repository.get_recent(limit):
            assessment = stored.assessment
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.FIRE_DANGER.value}:{stored.assessment_id}",
                    activity_type=OperationsActivityType.FIRE_DANGER,
                    entity_id=stored.assessment_id,
                    occurred_at=assessment.assessed_at,
                    available_at=stored.created_at,
                    title=f"Fire Danger Assessment - {assessment.area_name}",
                    location=OperationsActivityLocation(
                        latitude=assessment.area_latitude, longitude=assessment.area_longitude
                    ),
                    preview=FireDangerActivityPreview(
                        area_name=assessment.area_name,
                        status=assessment.status,
                        level=assessment.level,
                        score=assessment.score,
                    ),
                )
            )
        return items

    def _satellite_candidates(
        self, limit: int, fire_danger_areas: tuple[FireDangerAreaSnapshot, ...]
    ) -> list[OperationsActivityFeedItem]:
        items = []
        for stored in self._satellite_hotspot_repository.get_recent(limit):
            hotspot = stored.hotspot
            location_name = self._resolve_hotspot_location_name(
                hotspot.latitude, hotspot.longitude, fire_danger_areas
            )
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.SATELLITE_HOTSPOT.value}:{stored.id}",
                    activity_type=OperationsActivityType.SATELLITE_HOTSPOT,
                    entity_id=stored.id,
                    occurred_at=hotspot.detected_at,
                    available_at=stored.created_at if stored.created_at is not None else hotspot.detected_at,
                    title="Satellite Hotspot",
                    location=OperationsActivityLocation(latitude=hotspot.latitude, longitude=hotspot.longitude),
                    preview=SatelliteHotspotActivityPreview(
                        confidence=hotspot.confidence, frp=hotspot.frp, location_name=location_name
                    ),
                )
            )
        return items

    @staticmethod
    def _resolve_hotspot_location_name(
        latitude: float,
        longitude: float,
        fire_danger_areas: tuple[FireDangerAreaSnapshot, ...],
    ) -> str | None:
        """Read/presentation-only location label for a satellite hotspot.

        Reuses the already-fetched Fire Danger area snapshots for this same
        overview request (no extra query) - a hotspot's persisted
        coordinates are compared against each area's persisted circle
        (`area_latitude`/`area_longitude`/`area_radius_km`).
        When multiple areas' circles contain the point, the nearest center
        wins, with `area_id` as a deterministic tiebreaker for an exact
        distance tie. Returns `None` when no area's circle contains the
        point - being merely the geographically nearest area is NOT enough
        to claim the hotspot belongs to it. This never affects Fire
        Detection, evidence correlation, FireEvent, confidence, or
        Severity, and performs no database write - it is presentation
        metadata for the Activity Feed only.

        Delegates to the shared `resolve_nearest_containing_area_name`
        helper (src/utils/geo.py) - the exact same logic
        `ActiveFireEventsService` reuses for active FireEvents' own
        `location_name`, so this distance/tie-break rule is never
        duplicated.
        """
        return resolve_nearest_containing_area_name(latitude, longitude, fire_danger_areas)

    def _news_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        items = []
        for stored in self._news_repository.get_recent(limit):
            report = stored.report
            location = (
                OperationsActivityLocation(latitude=report.latitude, longitude=report.longitude)
                if report.latitude is not None and report.longitude is not None
                else None
            )
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.NEWS_REPORT.value}:{stored.id}",
                    activity_type=OperationsActivityType.NEWS_REPORT,
                    entity_id=stored.id,
                    occurred_at=stored.observed_at,
                    available_at=report.fetched_at,
                    title=report.title,
                    location=location,
                    preview=NewsReportActivityPreview(source=report.source_feed, headline=report.title),
                )
            )
        return items

    def _fire_event_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        items = []
        for stored in self._fire_event_repository.get_recent(limit):
            event = stored.event
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.FIRE_EVENT.value}:{stored.id}",
                    activity_type=OperationsActivityType.FIRE_EVENT,
                    entity_id=stored.id,
                    occurred_at=event.detected_at,
                    available_at=stored.created_at if stored.created_at is not None else event.detected_at,
                    title=f"Fire Event #{stored.id}",
                    location=OperationsActivityLocation(latitude=event.latitude, longitude=event.longitude),
                    preview=FireEventActivityPreview(status=event.status, confidence=event.detection_confidence),
                )
            )
        return items

    def _severity_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        items = []
        for stored in self._fire_severity_assessment_repository.get_recent(limit):
            assessment = stored.assessment
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.FIRE_SEVERITY.value}:{stored.assessment_id}",
                    activity_type=OperationsActivityType.FIRE_SEVERITY,
                    entity_id=stored.assessment_id,
                    occurred_at=assessment.assessed_at,
                    available_at=stored.created_at if stored.created_at is not None else assessment.assessed_at,
                    title="Fire Severity Assessment",
                    location=None,
                    preview=FireSeverityActivityPreview(
                        fire_event_id=assessment.fire_event_id,
                        level=assessment.level,
                        score=assessment.score,
                    ),
                )
            )
        return items

    def _global_planning_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        recent_runs = self._global_planning_run_repository.get_recent(limit)
        # Batched (one query for every candidate run) - NOT one get_members()
        # call per run, which would be an N+1 (see get_member_counts's own
        # docstring; this materially affects latency against remote Postgres/Neon).
        member_counts = self._global_planning_run_repository.get_member_counts(
            stored.id for stored in recent_runs
        )

        items = []
        for stored in recent_runs:
            run = stored.run
            occurred_at = run.completed_at if run.completed_at is not None else run.started_at
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.GLOBAL_PLANNING_RUN.value}:{stored.id}",
                    activity_type=OperationsActivityType.GLOBAL_PLANNING_RUN,
                    entity_id=stored.id,
                    occurred_at=occurred_at,
                    available_at=stored.created_at if stored.created_at is not None else occurred_at,
                    title="Global Response Plan",
                    location=None,
                    preview=GlobalPlanningRunActivityPreview(
                        status=run.status, fire_event_count=member_counts.get(stored.id, 0)
                    ),
                )
            )
        return items

    def _weather_conditions_candidates(self, limit: int) -> list[OperationsActivityFeedItem]:
        """"Weather Conditions" signal - one row per recent successfully
        calculated (VALID) Fire Danger assessment of ANY level, showing the
        mean of the exact weather observations already traced for that
        assessment. No FFWI recalculation, no fresh observation selection,
        no external weather call - purely a read/aggregate/map over
        already-persisted rows. INSUFFICIENT_DATA assessments (danger_level
        IS NULL) never match any of the five named levels, so they are
        excluded automatically - no separate check needed.

        `occurred_at` is the LATEST of the exact contributing
        WeatherObservation timestamps (a real domain/source timestamp) -
        deliberately NOT the assessment row's `created_at` (a DB-insert/
        availability timestamp): both Weather Conditions and Fire Danger's
        own `occurred_at` (`assessed_at`) use the same kind of timestamp (a
        real domain instant), so their relative source-time order reflects
        genuine chronology.

        `available_at` is that same assessment's `created_at` (Task 4, Part
        D/H) - the moment EcoGuard actually persisted the assessment this
        Weather Conditions row is a projection of, which is exactly when
        both this item and its paired FIRE_DANGER item became available.
        Reusing the SAME stored assessment's created_at for both is what
        guarantees they always tie (see _feed_sort_key's WEATHER_CONDITIONS-
        before-FIRE_DANGER tie-break).

        `title` is deliberately level-neutral ("Weather Conditions - {area}")
        - the frontend also renders it with fully neutral wording (Part J:
        never implying WeatherAgent itself classified the weather as
        dangerous - that interpretation belongs solely to the Fire Danger
        row). `entity_id`/`fire_danger_assessment_id` is the same assessment
        id the paired FIRE_DANGER item uses - the two are always derived
        from exactly one assessment, so they can never reference different
        assessments.

        The averaging itself is WeatherConditionsQueryService's (shared with
        ChatbotAgent); this method only selects the assessments and maps the
        result to feed items.
        """
        stored_assessments = self._fire_danger_assessment_repository.get_recent_with_level_in(
            ALL_FIRE_DANGER_LEVELS, limit
        )
        if not stored_assessments:
            return []

        items = []
        for result in self._weather_conditions_query_service.summarize_assessments(stored_assessments):
            stored = result.stored_assessment
            assessment = stored.assessment
            conditions = result.conditions
            items.append(
                OperationsActivityFeedItem(
                    activity_id=f"{OperationsActivityType.WEATHER_CONDITIONS.value}:{stored.assessment_id}",
                    activity_type=OperationsActivityType.WEATHER_CONDITIONS,
                    entity_id=stored.assessment_id,
                    occurred_at=conditions.observed_at,
                    available_at=stored.created_at,
                    title=f"Weather Conditions - {assessment.area_name}",
                    location=OperationsActivityLocation(
                        latitude=assessment.area_latitude, longitude=assessment.area_longitude
                    ),
                    preview=WeatherConditionsActivityPreview(
                        area_name=assessment.area_name,
                        fire_danger_level=assessment.level,
                        fire_danger_assessment_id=stored.assessment_id,
                        temperature_c=conditions.temperature_c,
                        relative_humidity_pct=conditions.relative_humidity_pct,
                        wind_speed_kmh=conditions.wind_speed_kmh,
                        wind_gust_kmh=conditions.wind_gust_kmh,
                    ),
                )
            )
        return items

    @staticmethod
    def _validate_activity_limit(activity_limit: object) -> None:
        if (
            isinstance(activity_limit, bool)
            or not isinstance(activity_limit, int)
            or not MIN_ACTIVITY_LIMIT <= activity_limit <= MAX_ACTIVITY_LIMIT
        ):
            raise ValueError(
                f"activity_limit must be an integer within [{MIN_ACTIVITY_LIMIT}, {MAX_ACTIVITY_LIMIT}], "
                f"got {activity_limit!r}"
            )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
