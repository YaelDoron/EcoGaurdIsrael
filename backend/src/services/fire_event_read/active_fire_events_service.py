"""Read-only active-FireEvents aggregation service (Epic 6, US 6.1, Task 2).

ActiveFireEventsService assembles US 6.1's `ActiveFireEventsResult` read
model purely by fetching data Fire Detection and Fire Severity Assessment
have already persisted. It contains no detection, no severity calculation,
and never invokes an agent or an external provider (IMS/FIRMS/Copernicus) -
a strict read/aggregate/map step, matching `ResponsePlanDetailsService`'s
own read-only precedent. It never writes to any repository.

An active FireEvent is one whose status is SUSPECTED or CONFIRMED - the same
definition FireEventRepository already enforces internally (see
_ACTIVE_STATUSES there); this service does not introduce a second lifecycle
rule. Each event's severity summary is its single latest persisted
FireSeverityAssessment, reported honestly regardless of status - never a
fallback to an older VALID assessment, and never one belonging to a
different FireEvent (get_latest_for_events keys strictly by fire_event_id).
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.active_fire_events import (
    ActiveFireEventSeveritySummary,
    ActiveFireEventsResult,
    ActiveFireEventSummary,
)
from src.models.fire_danger_areas import FireDangerAreaSnapshot
from src.models.fire_event import FireEvent
from src.repositories.fire_event_repository import FireEventRepository, StoredFireEvent
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.utils.geo import resolve_nearest_containing_area_name


class ActiveFireEventsService:
    """Assemble the active-FireEvents dashboard snapshot from already-persisted data."""

    def __init__(
        self,
        *,
        fire_event_repository: FireEventRepository | None = None,
        fire_severity_assessment_repository: FireSeverityAssessmentRepository | None = None,
    ) -> None:
        self._fire_event_repository = fire_event_repository or FireEventRepository()
        self._fire_severity_assessment_repository = (
            fire_severity_assessment_repository or FireSeverityAssessmentRepository()
        )

    def get_active_events(
        self,
        *,
        as_of: datetime | None = None,
        fire_danger_areas: tuple[FireDangerAreaSnapshot, ...] = (),
    ) -> ActiveFireEventsResult:
        """Return all currently active FireEvents with their latest severity, if any.

        Ordering is deterministic and comes entirely from
        FireEventRepository.get_active_events() (most-recently-updated
        first, tie-broken by id) - this method does not re-sort.

        `as_of` only stamps the returned snapshot's `as_of` field (when this
        read was taken); it does not filter which FireEvents are considered
        active, which is governed purely by persisted status. Pass an
        explicit value in tests for a deterministic snapshot timestamp;
        defaults to the current time otherwise.

        `fire_danger_areas` is optional, already-loaded Fire Danger area
        geometry (never re-queried here) used to resolve each event's
        `location_name` - the same read/presentation-only enrichment
        already used for satellite hotspots (see
        OperationsOverviewQueryService and src.utils.geo). Omitting it
        (the default) simply leaves every `location_name` as `None` - it
        never changes which events are returned or their ordering.
        """
        snapshot_time = as_of if as_of is not None else datetime.now(timezone.utc)
        self._validate_aware_datetime("as_of", snapshot_time)

        stored_events = self._fire_event_repository.get_active_events()
        severity_by_event_id = self._fire_severity_assessment_repository.get_latest_for_events(
            stored_event.id for stored_event in stored_events
        )

        items = tuple(
            self._to_summary(stored_event, severity_by_event_id.get(stored_event.id), fire_danger_areas)
            for stored_event in stored_events
        )
        return ActiveFireEventsResult(as_of=snapshot_time, items=items)

    @staticmethod
    def _to_summary(
        stored_event: StoredFireEvent,
        stored_severity: StoredFireSeverityAssessment | None,
        fire_danger_areas: tuple[FireDangerAreaSnapshot, ...],
    ) -> ActiveFireEventSummary:
        event = stored_event.event
        return ActiveFireEventSummary(
            fire_event_id=stored_event.id,
            status=event.status,
            latitude=event.latitude,
            longitude=event.longitude,
            detection_confidence=event.detection_confidence,
            detected_at=event.detected_at,
            updated_at=event.updated_at,
            created_at=stored_event.created_at,
            severity=(
                ActiveFireEventsService._to_severity_summary(stored_severity)
                if stored_severity is not None
                else None
            ),
            location_name=ActiveFireEventsService._resolve_location_name(event, fire_danger_areas),
        )

    @staticmethod
    def _resolve_location_name(
        event: FireEvent, fire_danger_areas: tuple[FireDangerAreaSnapshot, ...]
    ) -> str | None:
        """Priority (never overwrite a trusted value with a guess):

        1. `event.location_name` - trusted provenance set once at FireEvent
           creation (simulation's own canonical location, or a real,
           verified label - see FireEvent's own docstring). Always wins
           when present.
        2. The safe containing-area read-side fallback (for historical
           FireEvents created before this provenance existed, or real
           evidence with no trusted label).
        3. `None` - never invented.
        """
        if event.location_name is not None:
            return event.location_name
        return resolve_nearest_containing_area_name(event.latitude, event.longitude, fire_danger_areas)

    @staticmethod
    def _to_severity_summary(
        stored_severity: StoredFireSeverityAssessment,
    ) -> ActiveFireEventSeveritySummary:
        assessment = stored_severity.assessment
        return ActiveFireEventSeveritySummary(
            assessment_id=stored_severity.assessment_id,
            status=assessment.status,
            score=assessment.score,
            level=assessment.level,
            assessed_at=assessment.assessed_at,
        )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
