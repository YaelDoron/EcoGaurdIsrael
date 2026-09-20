"""Read-only Operations Activity detail DTOs (Task A5).

One stable read contract: "I have an activity type and entity id; give me
its persisted details." Each `*ActivityDetail` variant below is a pure
presentation wrapper - common fields (`entity_id`/`occurred_at`/`title`/
`location`) plus a `details` payload that, wherever an existing domain/
repository object already carries the needed shape, is that EXACT object
reused unchanged (never a re-derivation): `FireDangerActivityDetail.details`
is A4's own `FireDangerAssessmentDetail`; `SatelliteHotspotActivityDetail.details`
is the plain `SatelliteHotspot` domain dataclass; `NewsReportActivityDetail.details`
is the plain `WildfireReport` domain dataclass; `FireSeverityActivityDetail.details`
wraps the plain `FireSeverityAssessment` domain dataclass plus its persisted
input-trace ids; `GlobalPlanningRunActivityDetail.details` wraps the plain
`GlobalPlanningRun` domain dataclass plus its persisted per-FireEvent
membership rows. `FireEventActivityDetail.details` wraps the plain
`FireEvent` domain dataclass plus its persisted evidence refs and latest
severity reference.

No calculation happens anywhere in this module - it only shapes already-
persisted domain objects into one envelope per Task A5's contract.

Task A5 audit finding (see the query service's module docstring for the
full per-type source mapping): all six entity id spaces (FireDangerAssessment,
SatelliteHotspot, WildfireReport, FireEvent, FireSeverityAssessment,
GlobalPlanningRun) are plain positive integers - `entity_id: int`
uniformly, no UUID/string ids exist among these types today.

FireSeverityAssessment carries no coordinates of its own (only
`fire_event_id`) - `FireSeverityActivityDetail.location` is therefore always
`None`, by design, not an oversight (see Part 4: "do not invent location for
resources that lack reliable coordinates"). A GlobalPlanningRun spans
multiple FireEvents/areas, so it has no single coordinate either -
`GlobalPlanningRunActivityDetail.location` is always `None` for the same
reason. `FireEventActivityDetail` deliberately has no area/place name in its
`title` (`"Fire Event #{id}"` only) - FireEvent persists no trustworthy area
label (see src/models/active_fire_events.py's own precedent for the same
rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Union

from src.models.fire_danger_assessment_detail import FireDangerAssessmentDetail
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event import FireEvent
from src.models.fire_report import WildfireReport
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int
from src.models.satellite_hotspot import SatelliteHotspot


class OperationsActivityType(Enum):
    """The stable, closed set of activity kinds the drawer can request a detail for.

    Values are the exact public strings used in the URL path and API
    response - never a table/model name accepted verbatim from the request.
    """

    FIRE_DANGER = "fire_danger"
    SATELLITE_HOTSPOT = "satellite_hotspot"
    NEWS_REPORT = "news_report"
    FIRE_EVENT = "fire_event"
    FIRE_SEVERITY = "fire_severity"
    GLOBAL_PLANNING_RUN = "global_planning_run"
    WEATHER_CONDITIONS = "weather_conditions"


@dataclass(frozen=True)
class OperationsActivityLocation:
    """A point location for the drawer/map - never fabricated for an entity without one."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if isinstance(self.latitude, bool) or not isinstance(self.latitude, (int, float)):
            raise ValueError(f"latitude must be numeric, got {self.latitude!r}")
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude must be within [-90, 90], got {self.latitude!r}")
        if isinstance(self.longitude, bool) or not isinstance(self.longitude, (int, float)):
            raise ValueError(f"longitude must be numeric, got {self.longitude!r}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude must be within [-180, 180], got {self.longitude!r}")


def _validate_common(entity_id: object, occurred_at: object, title: object) -> None:
    validate_positive_int("entity_id", entity_id)
    if not isinstance(occurred_at, datetime):
        raise ValueError(f"occurred_at must be a datetime, got {occurred_at!r}")
    validate_non_empty_string("title", title)


# ---------------------------------------------------------------------------
# FIRE_DANGER - details is A4's own FireDangerAssessmentDetail, unchanged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireDangerActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: FireDangerAssessmentDetail
    activity_type: OperationsActivityType = OperationsActivityType.FIRE_DANGER

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.FIRE_DANGER:
            raise ValueError("activity_type must be FIRE_DANGER for FireDangerActivityDetail")
        if not isinstance(self.details, FireDangerAssessmentDetail):
            raise ValueError(f"details must be a FireDangerAssessmentDetail, got {self.details!r}")


# ---------------------------------------------------------------------------
# SATELLITE_HOTSPOT - details is the plain SatelliteHotspot domain dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SatelliteHotspotActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: SatelliteHotspot
    activity_type: OperationsActivityType = OperationsActivityType.SATELLITE_HOTSPOT

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.SATELLITE_HOTSPOT:
            raise ValueError("activity_type must be SATELLITE_HOTSPOT for SatelliteHotspotActivityDetail")
        if not isinstance(self.details, SatelliteHotspot):
            raise ValueError(f"details must be a SatelliteHotspot, got {self.details!r}")


# ---------------------------------------------------------------------------
# NEWS_REPORT - details is the plain WildfireReport domain dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsReportActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: WildfireReport
    activity_type: OperationsActivityType = OperationsActivityType.NEWS_REPORT

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.NEWS_REPORT:
            raise ValueError("activity_type must be NEWS_REPORT for NewsReportActivityDetail")
        if not isinstance(self.details, WildfireReport):
            raise ValueError(f"details must be a WildfireReport, got {self.details!r}")


# ---------------------------------------------------------------------------
# FIRE_EVENT - details wraps FireEvent + its evidence refs + a severity reference.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireEventEvidenceRefs:
    """Ids of the persisted evidence directly attached to one FireEvent."""

    satellite_hotspot_ids: tuple[int, ...]
    news_report_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "satellite_hotspot_ids", tuple(self.satellite_hotspot_ids))
        object.__setattr__(self, "news_report_ids", tuple(self.news_report_ids))


@dataclass(frozen=True)
class FireEventSeverityReference:
    """A lightweight pointer to a FireEvent's latest severity assessment.

    The full severity detail (trace inputs etc.) belongs to the FIRE_SEVERITY
    activity type (see FireSeverityActivityDetail) - this reference exists
    only so the FireEvent drawer can show/link the current severity without
    duplicating that detail here.
    """

    assessment_id: int
    status: FireSeverityAssessmentStatus
    score: float | None
    level: FireSeverityLevel | None
    assessed_at: datetime

    def __post_init__(self) -> None:
        validate_positive_int("assessment_id", self.assessment_id)
        if not isinstance(self.status, FireSeverityAssessmentStatus):
            raise ValueError(f"status must be a FireSeverityAssessmentStatus, got {self.status!r}")
        if not isinstance(self.assessed_at, datetime) or self.assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {self.assessed_at!r}")


@dataclass(frozen=True)
class FireEventActivityDetails:
    """`created_at` is the row's own DB-insert ("Opened") timestamp - see
    StoredFireEvent's own docstring - distinct from `fire_event.detected_at`
    (source evidence time, which may be earlier). Event Details exposes
    both, per this task's requirement not to remove historical evidence
    time when adding the operator-facing opened/created time."""

    fire_event: FireEvent
    evidence: FireEventEvidenceRefs
    latest_severity: FireEventSeverityReference | None
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.fire_event, FireEvent):
            raise ValueError(f"fire_event must be a FireEvent, got {self.fire_event!r}")
        if not isinstance(self.evidence, FireEventEvidenceRefs):
            raise ValueError(f"evidence must be a FireEventEvidenceRefs, got {self.evidence!r}")
        if self.latest_severity is not None and not isinstance(self.latest_severity, FireEventSeverityReference):
            raise ValueError(
                f"latest_severity must be a FireEventSeverityReference or None, got {self.latest_severity!r}"
            )
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ValueError(f"created_at must be a timezone-aware datetime, got {self.created_at!r}")


@dataclass(frozen=True)
class FireEventActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: FireEventActivityDetails
    activity_type: OperationsActivityType = OperationsActivityType.FIRE_EVENT

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.FIRE_EVENT:
            raise ValueError("activity_type must be FIRE_EVENT for FireEventActivityDetail")
        if not isinstance(self.details, FireEventActivityDetails):
            raise ValueError(f"details must be a FireEventActivityDetails, got {self.details!r}")


# ---------------------------------------------------------------------------
# FIRE_SEVERITY - details wraps FireSeverityAssessment + its persisted input trace.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireSeverityActivityDetails:
    assessment: FireSeverityAssessment
    weather_observation_ids: tuple[int, ...]
    satellite_hotspot_ids: tuple[int, ...]
    selected_frp_hotspot_id: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.assessment, FireSeverityAssessment):
            raise ValueError(f"assessment must be a FireSeverityAssessment, got {self.assessment!r}")
        object.__setattr__(self, "weather_observation_ids", tuple(self.weather_observation_ids))
        object.__setattr__(self, "satellite_hotspot_ids", tuple(self.satellite_hotspot_ids))


@dataclass(frozen=True)
class FireSeverityActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: FireSeverityActivityDetails
    activity_type: OperationsActivityType = OperationsActivityType.FIRE_SEVERITY

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.FIRE_SEVERITY:
            raise ValueError("activity_type must be FIRE_SEVERITY for FireSeverityActivityDetail")
        if not isinstance(self.details, FireSeverityActivityDetails):
            raise ValueError(f"details must be a FireSeverityActivityDetails, got {self.details!r}")


# ---------------------------------------------------------------------------
# GLOBAL_PLANNING_RUN - details wraps GlobalPlanningRun + its FireEvent membership.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalPlanningRunMemberSummary:
    """One FireEvent's membership/outcome within a GlobalPlanningRun, for the drawer."""

    fire_event_id: int
    event_order: int
    result_status: GlobalPlanningRunEventStatus | None
    response_plan_id: int | None
    severity_level: FireSeverityLevel | None
    assigned_resources: int | None
    coverage_score: float | None
    average_eta_seconds: float | None

    def __post_init__(self) -> None:
        validate_positive_int("fire_event_id", self.fire_event_id)
        if self.result_status is not None and not isinstance(self.result_status, GlobalPlanningRunEventStatus):
            raise ValueError(f"result_status must be a GlobalPlanningRunEventStatus or None, got {self.result_status!r}")
        if self.severity_level is not None and not isinstance(self.severity_level, FireSeverityLevel):
            raise ValueError(f"severity_level must be a FireSeverityLevel or None, got {self.severity_level!r}")


@dataclass(frozen=True)
class GlobalPlanningRunActivityDetails:
    """The global multi-fire optimization run - NEVER presented as belonging to one FireEvent.

    `members` preserves every FireEvent this cycle actually considered
    (Task A5, Part 10) - a run covering 2+ active fires must show all of
    them here, not a single-event projection.
    """

    run: GlobalPlanningRun
    members: tuple[GlobalPlanningRunMemberSummary, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, GlobalPlanningRun):
            raise ValueError(f"run must be a GlobalPlanningRun, got {self.run!r}")
        object.__setattr__(self, "members", tuple(self.members))

    @property
    def fire_event_ids(self) -> tuple[int, ...]:
        return tuple(member.fire_event_id for member in self.members)

    @property
    def response_plan_ids(self) -> tuple[int, ...]:
        return tuple(
            member.response_plan_id for member in self.members if member.response_plan_id is not None
        )


@dataclass(frozen=True)
class GlobalPlanningRunActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: GlobalPlanningRunActivityDetails
    activity_type: OperationsActivityType = OperationsActivityType.GLOBAL_PLANNING_RUN

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.GLOBAL_PLANNING_RUN:
            raise ValueError("activity_type must be GLOBAL_PLANNING_RUN for GlobalPlanningRunActivityDetail")
        if not isinstance(self.details, GlobalPlanningRunActivityDetails):
            raise ValueError(f"details must be a GlobalPlanningRunActivityDetails, got {self.details!r}")


# ---------------------------------------------------------------------------
# WEATHER_CONDITIONS - details wraps the exact persisted weather observations
# a HIGH+ Fire Danger assessment already traced (FireDangerAssessmentWeatherInputDB).
# No FFWI recalculation, no new independently-persisted WeatherAlert entity -
# entity_id IS the FireDangerAssessmentDB id (see Part G).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherConditionsStationReading:
    """One station's raw weather reading that fed a HIGH+ Fire Danger assessment."""

    station_id: int
    station_name: str
    observation_id: int
    observed_at: datetime
    temperature: float | None
    relative_humidity: float | None
    wind_speed: float | None
    wind_gust: float | None

    def __post_init__(self) -> None:
        validate_positive_int("station_id", self.station_id)
        validate_non_empty_string("station_name", self.station_name)
        validate_positive_int("observation_id", self.observation_id)
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise ValueError(f"observed_at must be a timezone-aware datetime, got {self.observed_at!r}")


@dataclass(frozen=True)
class WeatherConditionsActivityDetails:
    fire_danger_assessment_id: int
    area_name: str
    fire_danger_level: FireDangerLevel
    assessed_at: datetime
    readings: tuple[WeatherConditionsStationReading, ...]

    def __post_init__(self) -> None:
        validate_positive_int("fire_danger_assessment_id", self.fire_danger_assessment_id)
        validate_non_empty_string("area_name", self.area_name)
        if not isinstance(self.fire_danger_level, FireDangerLevel):
            raise ValueError(f"fire_danger_level must be a FireDangerLevel, got {self.fire_danger_level!r}")
        if not isinstance(self.assessed_at, datetime) or self.assessed_at.tzinfo is None:
            raise ValueError(f"assessed_at must be a timezone-aware datetime, got {self.assessed_at!r}")
        readings = tuple(self.readings)
        for reading in readings:
            if not isinstance(reading, WeatherConditionsStationReading):
                raise ValueError(f"readings must contain WeatherConditionsStationReading entries, got {reading!r}")
        object.__setattr__(self, "readings", readings)


@dataclass(frozen=True)
class WeatherConditionsActivityDetail:
    entity_id: int
    occurred_at: datetime
    title: str
    location: OperationsActivityLocation | None
    details: WeatherConditionsActivityDetails
    activity_type: OperationsActivityType = OperationsActivityType.WEATHER_CONDITIONS

    def __post_init__(self) -> None:
        _validate_common(self.entity_id, self.occurred_at, self.title)
        if self.activity_type is not OperationsActivityType.WEATHER_CONDITIONS:
            raise ValueError("activity_type must be WEATHER_CONDITIONS for WeatherConditionsActivityDetail")
        if not isinstance(self.details, WeatherConditionsActivityDetails):
            raise ValueError(f"details must be a WeatherConditionsActivityDetails, got {self.details!r}")


OperationsActivityDetail = Union[
    FireDangerActivityDetail,
    SatelliteHotspotActivityDetail,
    NewsReportActivityDetail,
    FireEventActivityDetail,
    FireSeverityActivityDetail,
    GlobalPlanningRunActivityDetail,
    WeatherConditionsActivityDetail,
]
