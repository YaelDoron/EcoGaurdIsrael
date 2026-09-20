"""Read-only Operations Overview dashboard DTOs (Task A6).

One coherent snapshot of the current operational state: current Fire Danger
areas, currently active FireEvents (each with its latest persisted
Severity), a chronological Activity Feed, and the A3 simulation-control
summary. No calculation happens anywhere in this module - it only shapes
already-persisted/already-computed domain objects into one envelope.

Maximal reuse, never duplication, of prior tasks' own domain types:
    fire_danger_areas   -> tuple[FireDangerAreaSnapshot, ...]      (A4, src.models.fire_danger_areas)
    active_fires         -> tuple[ActiveFireEventSummary, ...]      (US 6.1, src.models.active_fire_events)
    simulation.run        -> SimulationRunSnapshot | None            (A3, src.services.simulation_control.simulation_run_manager)

`SimulationRunSnapshot`/`SimulationRunState` are reused from A3's services
module rather than duplicated here - they are plain data (a frozen
dataclass and an Enum), not executable business logic, so this remains a
pure read/presentation module despite the cross-layer reference (see Part
33's "Domain/result enums are fine" - the same principle applies to a plain
result snapshot dataclass). The import itself is deferred to inside
OperationsSimulationSummary.__post_init__ (not at module level): A3's
`src.simulation` package transitively imports `src.models` (its analysis
coordinators build response-target/severity/etc. domain objects), so an
eager top-level import here would be a real circular import at package-load
time, not just a style preference - see SimulationRunManager.start_run's
own lazy-settings-import for the identical, already-established pattern in
this codebase.

Activity Feed items are deliberately NOT the full A5
`OperationsActivityDetail` - those embed full per-type detail (weather
traces, vegetation traces, evidence collections) that would make a
2-second-polled feed heavy. Each `OperationsActivityFeedItem.preview` is a
small, typed, per-type projection - see the six `*ActivityPreview`
dataclasses below. `activity_id` is the stable composite
"{activity_type}:{entity_id}" identity (Task A6, Part 10) - entity_id alone
is not globally unique across the six source tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from src.models.fire_danger_areas import FireDangerAreaSnapshot
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event_status import FireEventStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import OperationsActivityLocation, OperationsActivityType
from src.models.optimization_validation import validate_non_empty_string, validate_positive_int

if TYPE_CHECKING:
    # Static-typing only - the real import is deferred to inside
    # OperationsSimulationSummary.__post_init__ to avoid a real circular
    # import at package-load time (see the module docstring above).
    from src.services.simulation_control.simulation_run_manager import SimulationRunSnapshot


@dataclass(frozen=True)
class OperationsSimulationSummary:
    """The A3 simulation-control state, exposed for the dashboard without a second request.

    `run` is `None` whenever there is nothing meaningful to report: the
    control API is disabled (`enabled=False`), or it is enabled but no run
    has ever started (the manager's own IDLE sentinel, `run_id is None`).
    Reuses A3's own `SimulationRunSnapshot` unchanged when a run exists -
    never a re-derived summary.
    """

    enabled: bool
    run: "SimulationRunSnapshot | None"

    def __post_init__(self) -> None:
        from src.services.simulation_control.simulation_run_manager import SimulationRunSnapshot

        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be a bool, got {self.enabled!r}")
        if self.run is not None and not isinstance(self.run, SimulationRunSnapshot):
            raise ValueError(f"run must be a SimulationRunSnapshot or None, got {self.run!r}")


# ---------------------------------------------------------------------------
# Activity Feed item previews - small, typed, per-type projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireDangerActivityPreview:
    area_name: str
    status: FireDangerAssessmentStatus
    level: FireDangerLevel | None
    score: float | None


@dataclass(frozen=True)
class SatelliteHotspotActivityPreview:
    confidence: str | None
    frp: float | None
    # Read/presentation-only enrichment (never persisted, never used by Fire
    # Detection/Severity): the real persisted `area_name` of a currently-known
    # Fire Danger area whose circle (center + radius_km) contains this
    # hotspot's own persisted coordinates, or None when no such area exists.
    # See OperationsOverviewQueryService._resolve_hotspot_location_name.
    location_name: str | None = None


@dataclass(frozen=True)
class NewsReportActivityPreview:
    source: str
    headline: str


@dataclass(frozen=True)
class FireEventActivityPreview:
    status: FireEventStatus
    confidence: float


@dataclass(frozen=True)
class FireSeverityActivityPreview:
    fire_event_id: int
    level: FireSeverityLevel | None
    score: float | None


@dataclass(frozen=True)
class GlobalPlanningRunActivityPreview:
    status: GlobalPlanningRunStatus
    fire_event_count: int


@dataclass(frozen=True)
class WeatherConditionsActivityPreview:
    """Read/presentation-only projection of the persisted weather inputs that
    fed a HIGH+ Fire Danger assessment (Part F/G) - never a re-derived FFWI
    score, never a new independently-persisted WeatherAlert entity. The
    temperature/humidity/wind values are the mean across the exact station
    observations FireDangerAssessmentRepository already traced for this
    assessment (FireDangerAssessmentWeatherInputDB) - the same observations
    FireDangerInputService averaged for the FFWI calculation, never a fresh
    selection. `wind_gust_kmh` is None whenever no contributing observation
    persisted a gust value.
    """

    area_name: str
    fire_danger_level: FireDangerLevel
    fire_danger_assessment_id: int
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    wind_gust_kmh: float | None = None


ActivityPreview = (
    FireDangerActivityPreview,
    SatelliteHotspotActivityPreview,
    NewsReportActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    WeatherConditionsActivityPreview,
)


@dataclass(frozen=True)
class OperationsActivityFeedItem:
    """One lightweight Activity Feed entry - never the full A5 detail payload.

    `occurred_at` is the source/domain event time (observation, publication,
    assessment, detection - see each `_X_candidates` builder in
    OperationsOverviewQueryService for the exact per-type meaning). It never
    changes and is preserved for A5 detail/history purposes.

    `available_at` is the real persisted moment EcoGuard could reliably
    expose this activity (an assessment/insert/ingestion timestamp - never a
    frontend/browser "first seen" time, never fabricated). It is what the
    live Activity Feed timeline is sorted and displayed by (Task 4: "Make
    Activity Feed timestamps represent when items became available"),
    precisely because a source timestamp (e.g. a satellite pass a few
    minutes ago) can lag well behind when the item actually appeared on
    screen.
    """

    activity_id: str
    activity_type: OperationsActivityType
    entity_id: int
    occurred_at: datetime
    available_at: datetime
    title: str
    location: OperationsActivityLocation | None
    preview: object

    def __post_init__(self) -> None:
        expected_id = f"{self.activity_type.value}:{self.entity_id}"
        if self.activity_id != expected_id:
            raise ValueError(f"activity_id must be {expected_id!r}, got {self.activity_id!r}")
        validate_positive_int("entity_id", self.entity_id)
        if not isinstance(self.occurred_at, datetime):
            raise ValueError(f"occurred_at must be a datetime, got {self.occurred_at!r}")
        if not isinstance(self.available_at, datetime) or self.available_at.tzinfo is None:
            raise ValueError(f"available_at must be a timezone-aware datetime, got {self.available_at!r}")
        validate_non_empty_string("title", self.title)
        if not isinstance(self.preview, ActivityPreview):
            raise ValueError(f"preview must be one of the typed *ActivityPreview dataclasses, got {self.preview!r}")


@dataclass(frozen=True)
class OperationsActivityFeed:
    """The bounded, globally-sorted Activity Feed for the dashboard."""

    items: tuple[OperationsActivityFeedItem, ...]
    limit: int

    def __post_init__(self) -> None:
        items = _coerce_tuple("items", self.items)
        for item in items:
            if not isinstance(item, OperationsActivityFeedItem):
                raise ValueError(f"items must contain OperationsActivityFeedItem entries, got {item!r}")
        object.__setattr__(self, "items", items)
        validate_positive_int("limit", self.limit)
        if len(items) > self.limit:
            raise ValueError(f"items ({len(items)}) must not exceed limit ({self.limit})")


@dataclass(frozen=True)
class OperationsOverviewSnapshot:
    """Root read model: the Operations Overview dashboard snapshot at one instant."""

    generated_at: datetime
    simulation: OperationsSimulationSummary
    fire_danger_areas: tuple[FireDangerAreaSnapshot, ...]
    active_fires: tuple  # tuple[ActiveFireEventSummary, ...] - see src.models.active_fire_events
    activity_feed: OperationsActivityFeed

    def __post_init__(self) -> None:
        if not isinstance(self.generated_at, datetime) or self.generated_at.tzinfo is None:
            raise ValueError(f"generated_at must be a timezone-aware datetime, got {self.generated_at!r}")
        if not isinstance(self.simulation, OperationsSimulationSummary):
            raise ValueError(f"simulation must be an OperationsSimulationSummary, got {self.simulation!r}")
        object.__setattr__(self, "fire_danger_areas", _coerce_tuple("fire_danger_areas", self.fire_danger_areas))
        object.__setattr__(self, "active_fires", _coerce_tuple("active_fires", self.active_fires))
        if not isinstance(self.activity_feed, OperationsActivityFeed):
            raise ValueError(f"activity_feed must be an OperationsActivityFeed, got {self.activity_feed!r}")


def _coerce_tuple(field_name: str, value: object) -> tuple:
    try:
        return tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be iterable, got {value!r}") from exc
