"""Simulation scenario definitions and builders."""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.models.resource_status import ResourceStatus
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulated_incident import SimulatedIncident
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_resource_status_change import SimulationResourceStatusChange
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import (
    CARMEL_LOCATION,
    DEFAULT_CARMEL_LOCATION,
    GALILEE_LOCATION,
    GOLAN_LOCATION,
    JERUSALEM_FOREST_LOCATION,
    JUDEAN_HILLS_LOCATION,
    SIMULATION_LOCATIONS,
    get_simulation_location_key,
)

# The hard validation ceiling for ANY scenario's duration_seconds (see
# SimulationScenario.__init__). Raised well above the old fixed 120s so the
# seeded `operations_demo` family (Part I/Q) can accommodate up to 4
# ACTIVE_FIRE incidents' worth of 10-30s-spaced source events without
# hitting the ceiling - see build_operations_demo_scenario's own worst-case
# math. This is a ceiling, not a default: every OTHER preset below
# continues to request its own unchanged, literal
# LEGACY_FIXED_SCENARIO_DURATION_SECONDS value, so raising this ceiling has
# zero effect on their reported/validated duration.
MAX_SCENARIO_DURATION_SECONDS = 900

# The exact, unchanged duration every non-operations_demo builder below has
# always used - kept as its own named constant (rather than reusing the
# now-larger MAX_SCENARIO_DURATION_SECONDS) so this task's ceiling increase
# cannot silently change any other preset's behavior/reported duration.
LEGACY_FIXED_SCENARIO_DURATION_SECONDS = 120

_WEATHER_ONLY_EVENT_PLAN = (
    (0, SimulationEventType.WEATHER),
    (60, SimulationEventType.WEATHER),
)
_ACTIVE_FIRE_EVENT_PLAN = (
    (0, SimulationEventType.WEATHER),
    (20, SimulationEventType.SATELLITE),
    (40, SimulationEventType.NEWS),
    (65, SimulationEventType.WEATHER),
    (80, SimulationEventType.SATELLITE),
    (100, SimulationEventType.NEWS),
)
_ACTIVE_FIRE_RESOURCE_REFRESH_EVENT_PLAN = (
    *_ACTIVE_FIRE_EVENT_PLAN,
    (60, SimulationEventType.RESOURCE_STATUS),
    (90, SimulationEventType.RESOURCE_STATUS),
)


@dataclass(frozen=True)
class SimulationScenario:
    """An immutable simulation timeline containing one or more incidents."""

    duration_seconds: int
    seed: int
    incidents: tuple[SimulatedIncident, ...]
    events: tuple[SimulationEvent, ...]

    def __init__(
        self,
        duration_seconds: int,
        seed: int,
        incidents: Iterable[SimulatedIncident],
        events: Iterable[SimulationEvent],
    ) -> None:
        if (
            isinstance(duration_seconds, bool)
            or not isinstance(duration_seconds, int)
            or duration_seconds <= 0
            or duration_seconds > MAX_SCENARIO_DURATION_SECONDS
        ):
            raise ValueError(
                "duration_seconds must be an integer in "
                f"[1, {MAX_SCENARIO_DURATION_SECONDS}], got {duration_seconds!r}"
            )

        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an integer, got {seed!r}")

        incident_tuple = tuple(incidents)
        incident_ids: set[str] = set()
        for incident in incident_tuple:
            if not isinstance(incident, SimulatedIncident):
                raise ValueError(f"incidents must contain SimulatedIncident instances, got {incident!r}")
            if incident.incident_id in incident_ids:
                raise ValueError(f"duplicate incident_id: {incident.incident_id!r}")
            incident_ids.add(incident.incident_id)

        event_tuple = tuple(events)
        for event in event_tuple:
            if not isinstance(event, SimulationEvent):
                raise ValueError(f"events must contain SimulationEvent instances, got {event!r}")
            if event.offset_seconds > duration_seconds:
                raise ValueError(
                    f"event offset {event.offset_seconds} exceeds scenario duration {duration_seconds}"
                )
            if event.incident_id not in incident_ids:
                raise ValueError(f"event references unknown incident_id: {event.incident_id!r}")

        sorted_events = tuple(sorted(event_tuple, key=lambda event: event.offset_seconds))

        object.__setattr__(self, "duration_seconds", duration_seconds)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "incidents", incident_tuple)
        object.__setattr__(self, "events", sorted_events)

    def get_incident(self, incident_id: str) -> SimulatedIncident:
        """Return the incident for an event incident_id."""
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                return incident
        raise KeyError(f"Unknown incident_id: {incident_id!r}")


def build_active_fire_scenario(
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build the initial 120-second ACTIVE_FIRE demo timeline."""
    incident = SimulatedIncident(
        incident_id=incident_id or _default_incident_id(ScenarioType.ACTIVE_FIRE, location),
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=location,
    )
    return build_multi_incident_scenario(
        incidents=(incident,),
        events=_build_events_for_incident(incident.incident_id, _ACTIVE_FIRE_EVENT_PLAN),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_low_risk_no_fire_scenario(
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build the LOW_RISK_NO_FIRE demo timeline."""
    incident = SimulatedIncident(
        incident_id=incident_id or _default_incident_id(ScenarioType.LOW_RISK_NO_FIRE, location),
        scenario_type=ScenarioType.LOW_RISK_NO_FIRE,
        location=location,
    )
    return build_multi_incident_scenario(
        incidents=(incident,),
        events=_build_events_for_incident(incident.incident_id, _WEATHER_ONLY_EVENT_PLAN),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_high_risk_no_fire_scenario(
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build the HIGH_RISK_NO_FIRE demo timeline."""
    incident = SimulatedIncident(
        incident_id=incident_id or _default_incident_id(ScenarioType.HIGH_RISK_NO_FIRE, location),
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=location,
    )
    return build_multi_incident_scenario(
        incidents=(incident,),
        events=_build_events_for_incident(incident.incident_id, _WEATHER_ONLY_EVENT_PLAN),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_moderate_risk_no_fire_scenario(
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build the MODERATE_RISK_NO_FIRE demo timeline."""
    incident = SimulatedIncident(
        incident_id=incident_id or _default_incident_id(ScenarioType.MODERATE_RISK_NO_FIRE, location),
        scenario_type=ScenarioType.MODERATE_RISK_NO_FIRE,
        location=location,
    )
    return build_multi_incident_scenario(
        incidents=(incident,),
        events=_build_events_for_incident(incident.incident_id, _WEATHER_ONLY_EVENT_PLAN),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_scenario(
    scenario_type: ScenarioType,
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build the default scenario timeline for a ScenarioType."""
    if scenario_type is ScenarioType.LOW_RISK_NO_FIRE:
        return build_low_risk_no_fire_scenario(location=location, seed=seed, incident_id=incident_id)
    if scenario_type is ScenarioType.MODERATE_RISK_NO_FIRE:
        return build_moderate_risk_no_fire_scenario(location=location, seed=seed, incident_id=incident_id)
    if scenario_type is ScenarioType.HIGH_RISK_NO_FIRE:
        return build_high_risk_no_fire_scenario(location=location, seed=seed, incident_id=incident_id)
    if scenario_type is ScenarioType.ACTIVE_FIRE:
        return build_active_fire_scenario(location=location, seed=seed, incident_id=incident_id)
    raise ValueError(f"Unsupported scenario_type: {scenario_type!r}")


def build_active_fire_resource_refresh_scenario(
    location: SimulationLocation = DEFAULT_CARMEL_LOCATION,
    seed: int = 42,
    incident_id: str | None = None,
) -> SimulationScenario:
    """Build an ACTIVE_FIRE timeline with deterministic resource-status changes."""
    incident = SimulatedIncident(
        incident_id=incident_id or "incident-active-fire-resource-refresh-carmel-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=location,
    )
    return build_multi_incident_scenario(
        incidents=(incident,),
        events=_build_events_for_incident(
            incident.incident_id,
            _ACTIVE_FIRE_RESOURCE_REFRESH_EVENT_PLAN,
        ),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_multi_incident_scenario(
    incidents: Iterable[SimulatedIncident],
    events: Iterable[SimulationEvent],
    duration_seconds: int = LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
    seed: int = 42,
) -> SimulationScenario:
    """Build a scenario from already-defined incidents and event timeline."""
    return SimulationScenario(
        duration_seconds=duration_seconds,
        seed=seed,
        incidents=incidents,
        events=events,
    )


def build_carmel_golan_active_fire_scenario(seed: int = 42) -> SimulationScenario:
    """Build a deterministic two-incident active fire scenario for demos and tests."""
    carmel_incident = SimulatedIncident(
        incident_id="incident-carmel-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=CARMEL_LOCATION,
    )
    golan_incident = SimulatedIncident(
        incident_id="incident-golan-01",
        scenario_type=ScenarioType.ACTIVE_FIRE,
        location=GOLAN_LOCATION,
    )
    golan_plan = tuple(
        (offset + 10 if event_type is SimulationEventType.WEATHER else offset + 15, event_type)
        for offset, event_type in _ACTIVE_FIRE_EVENT_PLAN
    )
    return build_multi_incident_scenario(
        incidents=(carmel_incident, golan_incident),
        events=(
            *_build_events_for_incident(carmel_incident.incident_id, _ACTIVE_FIRE_EVENT_PLAN),
            *_build_events_for_incident(golan_incident.incident_id, golan_plan),
        ),
        duration_seconds=LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


MIN_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT = 1
MAX_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT = 4

_RISK_ONLY_SCENARIO_TYPES = (
    ScenarioType.LOW_RISK_NO_FIRE,
    ScenarioType.MODERATE_RISK_NO_FIRE,
    ScenarioType.HIGH_RISK_NO_FIRE,
)
_RISK_ONLY_SCENARIO_TYPE_LABEL = {
    ScenarioType.LOW_RISK_NO_FIRE: "low",
    ScenarioType.MODERATE_RISK_NO_FIRE: "moderate",
    ScenarioType.HIGH_RISK_NO_FIRE: "high",
}

# Two waves each (an initial detection + a later confirmation/escalation
# wave), matching the evidence depth the original fixed 2-fire demo already
# used - this is what lets Severity visibly progress as more evidence
# arrives, not just a bare minimum single satellite+news pair.
_OPERATIONS_DEMO_ACTIVE_FIRE_EVENT_TYPES = (
    SimulationEventType.WEATHER,
    SimulationEventType.SATELLITE,
    SimulationEventType.NEWS,
) * 2
_OPERATIONS_DEMO_RISK_ONLY_EVENT_TYPES = (SimulationEventType.WEATHER,) * 2

# Task: consecutive scheduled SOURCE events (WEATHER/SATELLITE/NEWS) should
# normally be spaced 10-30 simulated seconds apart, after the first event
# (which may land at T+0).
_SOURCE_EVENT_MIN_GAP_SECONDS = 10
_SOURCE_EVENT_MAX_GAP_SECONDS = 30
# Added after the final scheduled event so downstream reactions (Fire
# Danger/Detection/Severity/Global Planning) have room to complete within
# the scenario's own duration - these are NOT additional sleeps or extra
# scheduled events, just how long the scenario is considered to run.
_OPERATIONS_DEMO_COMPLETION_MARGIN_SECONDS = 20


def build_operations_demo_scenario(seed: int = 42) -> SimulationScenario:
    """Build the seeded `operations_demo` scenario family.

    Same seed -> byte-identical scenario (active-fire count, which
    canonical locations are active-fire vs risk-only, risk-level
    assignment for the risk-only locations, and source-event
    order/timing). A different seed may vary any/all of those - see
    build_operations_demo_scenario_signature for a lightweight summary
    used by tests to prove this.

    This builder only emits INPUT events (weather/satellite/news); it
    never fabricates fire danger, detection, severity, spread, or planning
    results, and it never decides which locations become FireEvents -
    FireDetectionAgent alone decides that from the emitted evidence. It
    also never hardcodes/duplicates canonical location coordinates -
    every location comes from the existing SIMULATION_LOCATIONS registry.

    Composition algorithm (all draws from one `random.Random(seed)`, in
    this fixed order, so the same seed always reproduces the same result):
      1. Pick `active_fire_count` uniformly in
         [MIN_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT, MAX_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT].
      2. Sample that many canonical location keys WITHOUT replacement from
         the registry (sorted first for a stable base ordering) - these
         become ACTIVE_FIRE incidents; every other canonical location
         becomes a risk-only incident with an independently-drawn risk
         ScenarioType (LOW/MODERATE/HIGH_RISK_NO_FIRE).
      3. Give each incident its fixed-length, fixed-INTERNAL-order source
         event plan (risk-only: WEATHER x2; active-fire: WEATHER/
         SATELLITE/NEWS x2), then repeatedly pick a random incident with
         events still pending and emit its next event - this interleaves
         *across* incidents (which incident's turn is next varies by seed)
         while preserving each incident's own natural event order.
      4. Assign offsets: the first emitted event lands at T+0; every
         following event's offset is the previous one's plus a fresh
         `randint(10, 30)` - so gaps vary and are never identical, but stay
         in the requested 10-30s range.
      5. `duration_seconds` covers the final scheduled offset plus a fixed
         completion margin (Part I) - never the old fixed 120s once a
         larger scenario needs more room (see MAX_SCENARIO_DURATION_SECONDS's
         own docstring for the worst-case bound this stays safely under).

    Fire Detection validity (Part F/Part W - correlation rules themselves
    are NEVER modified here): every ACTIVE_FIRE incident's SATELLITE/NEWS
    events all reference that SAME incident's location, so the data
    generators (which jitter coordinates within ~2km of the given
    location) always stay inside FireDetectionCalculator's existing
    MAX_EVIDENCE_DISTANCE_KM (5.0km); and the whole scenario fits well
    inside MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES (60 min), so evidence for
    the same incident always correlates regardless of the randomized
    ordering/timing.
    """
    rng = random.Random(seed)

    location_keys = sorted(SIMULATION_LOCATIONS.keys())
    active_fire_count = rng.randint(
        MIN_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT,
        min(MAX_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT, len(location_keys)),
    )
    active_location_keys = set(rng.sample(location_keys, active_fire_count))
    risk_only_location_keys = [key for key in location_keys if key not in active_location_keys]

    incidents: list[SimulatedIncident] = []
    incident_event_types: dict[str, list[SimulationEventType]] = {}

    for location_key in sorted(active_location_keys):
        incident_id = f"incident-{location_key.replace('_', '-')}-01"
        incidents.append(
            SimulatedIncident(
                incident_id=incident_id,
                scenario_type=ScenarioType.ACTIVE_FIRE,
                location=SIMULATION_LOCATIONS[location_key],
            )
        )
        incident_event_types[incident_id] = list(_OPERATIONS_DEMO_ACTIVE_FIRE_EVENT_TYPES)

    for location_key in risk_only_location_keys:
        risk_scenario_type = rng.choice(_RISK_ONLY_SCENARIO_TYPES)
        incident_id = f"incident-{location_key.replace('_', '-')}-{_RISK_ONLY_SCENARIO_TYPE_LABEL[risk_scenario_type]}-01"
        incidents.append(
            SimulatedIncident(
                incident_id=incident_id,
                scenario_type=risk_scenario_type,
                location=SIMULATION_LOCATIONS[location_key],
            )
        )
        incident_event_types[incident_id] = list(_OPERATIONS_DEMO_RISK_ONLY_EVENT_TYPES)

    # Interleave across incidents while preserving each incident's own
    # internal event order (Part G): repeatedly draw a random incident that
    # still has events pending and emit its next one.
    pending_incident_ids = [incident.incident_id for incident in incidents]
    ordered_slots: list[tuple[str, SimulationEventType]] = []
    while pending_incident_ids:
        chosen_incident_id = rng.choice(pending_incident_ids)
        event_type = incident_event_types[chosen_incident_id].pop(0)
        ordered_slots.append((chosen_incident_id, event_type))
        if not incident_event_types[chosen_incident_id]:
            pending_incident_ids.remove(chosen_incident_id)

    # Assign offsets: T+0 for the first event, then a fresh 10-30s gap per
    # subsequent event (Part H).
    events: list[SimulationEvent] = []
    per_incident_type_index: dict[tuple[str, SimulationEventType], int] = defaultdict(int)
    offset_seconds = 0
    for position, (incident_id, event_type) in enumerate(ordered_slots):
        if position > 0:
            offset_seconds += rng.randint(_SOURCE_EVENT_MIN_GAP_SECONDS, _SOURCE_EVENT_MAX_GAP_SECONDS)
        key = (incident_id, event_type)
        events.append(
            SimulationEvent(
                offset_seconds=offset_seconds,
                event_type=event_type,
                incident_id=incident_id,
                source_event_index=per_incident_type_index[key],
            )
        )
        per_incident_type_index[key] += 1

    duration_seconds = offset_seconds + _OPERATIONS_DEMO_COMPLETION_MARGIN_SECONDS

    return build_multi_incident_scenario(
        incidents=incidents,
        events=events,
        duration_seconds=duration_seconds,
        seed=seed,
    )


@dataclass(frozen=True)
class OperationsDemoScenarioSignature:
    """A lightweight, order-preserving summary of one operations_demo scenario build.

    Not a public API - exists so tests can assert "same seed -> identical
    signature" / "representative seeds -> meaningfully different
    signature" without re-deriving these facts from raw SimulationScenario
    internals each time (Part N).
    """

    active_fire_count: int
    active_location_keys: tuple[str, ...]
    location_scenario_types: tuple[tuple[str, str], ...]
    event_schedule: tuple[tuple[int, str, str], ...]
    duration_seconds: int


def build_operations_demo_scenario_signature(scenario: SimulationScenario) -> OperationsDemoScenarioSignature:
    """Derive a comparable signature from an already-built scenario (any SimulationScenario, not just operations_demo)."""
    location_scenario_types = tuple(
        sorted(
            (
                get_simulation_location_key(incident.location) or incident.location.name,
                incident.scenario_type.value,
            )
            for incident in scenario.incidents
        )
    )
    active_location_keys = tuple(
        sorted(
            get_simulation_location_key(incident.location) or incident.location.name
            for incident in scenario.incidents
            if incident.scenario_type is ScenarioType.ACTIVE_FIRE
        )
    )
    event_schedule = tuple(
        (event.offset_seconds, event.event_type.value, event.incident_id) for event in scenario.events
    )
    return OperationsDemoScenarioSignature(
        active_fire_count=len(active_location_keys),
        active_location_keys=active_location_keys,
        location_scenario_types=location_scenario_types,
        event_schedule=event_schedule,
        duration_seconds=scenario.duration_seconds,
    )


def _build_events_for_incident(
    incident_id: str,
    event_plan: Iterable[tuple[int, SimulationEventType]],
) -> tuple[SimulationEvent, ...]:
    indexes_by_type: dict[SimulationEventType, int] = defaultdict(int)
    events: list[SimulationEvent] = []
    for offset_seconds, event_type in event_plan:
        source_event_index = indexes_by_type[event_type]
        resource_status_change = None
        if event_type is SimulationEventType.RESOURCE_STATUS:
            resource_status_change = SimulationResourceStatusChange(
                new_status=(
                    ResourceStatus.UNAVAILABLE
                    if source_event_index == 0
                    else ResourceStatus.AVAILABLE
                ),
                selection_key=f"{incident_id}:primary-resource",
            )
        events.append(
            SimulationEvent(
                offset_seconds=offset_seconds,
                event_type=event_type,
                incident_id=incident_id,
                source_event_index=source_event_index,
                resource_status_change=resource_status_change,
            )
        )
        indexes_by_type[event_type] += 1
    return tuple(events)


def _default_incident_id(scenario_type: ScenarioType, location: SimulationLocation) -> str:
    location_key = get_simulation_location_key(location) or "custom"
    scenario_key = scenario_type.value.replace("_", "-")
    return f"incident-{scenario_key}-{location_key.replace('_', '-')}-01"
