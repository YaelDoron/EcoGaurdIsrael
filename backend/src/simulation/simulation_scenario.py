"""Simulation scenario definitions and builders."""
from __future__ import annotations

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
    GOLAN_LOCATION,
    get_simulation_location_key,
)

MAX_SCENARIO_DURATION_SECONDS = 120

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
        duration_seconds=MAX_SCENARIO_DURATION_SECONDS,
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
        duration_seconds=MAX_SCENARIO_DURATION_SECONDS,
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
        duration_seconds=MAX_SCENARIO_DURATION_SECONDS,
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
        duration_seconds=MAX_SCENARIO_DURATION_SECONDS,
        seed=seed,
    )


def build_multi_incident_scenario(
    incidents: Iterable[SimulatedIncident],
    events: Iterable[SimulationEvent],
    duration_seconds: int = MAX_SCENARIO_DURATION_SECONDS,
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
        duration_seconds=MAX_SCENARIO_DURATION_SECONDS,
        seed=seed,
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
