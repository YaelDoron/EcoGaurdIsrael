"""Tests for simulation scenario definitions."""

import pytest

from src.models.resource_status import ResourceStatus
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    LEGACY_FIXED_SCENARIO_DURATION_SECONDS,
    MAX_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT,
    MAX_SCENARIO_DURATION_SECONDS,
    MIN_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT,
    SIMULATION_LOCATIONS,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventType,
    SimulationScenario,
    build_active_fire_scenario,
    build_active_fire_resource_refresh_scenario,
    build_carmel_golan_active_fire_scenario,
    build_high_risk_no_fire_scenario,
    build_low_risk_no_fire_scenario,
    build_moderate_risk_no_fire_scenario,
    build_operations_demo_scenario,
    build_operations_demo_scenario_signature,
    build_scenario,
)

_SOURCE_EVENT_TYPES = (SimulationEventType.WEATHER, SimulationEventType.SATELLITE, SimulationEventType.NEWS)
_MIN_SOURCE_EVENT_GAP_SECONDS = 10
_MAX_SOURCE_EVENT_GAP_SECONDS = 30
_OPERATIONS_DEMO_COMPLETION_MARGIN_SECONDS = 20

# Representative seeds (discovered by scanning seeds 1-500 against this
# exact algorithm) covering all four possible active_fire_count values -
# not hardcoded assumptions about seed->count mapping in general (Part P),
# just fixed examples for deterministic coverage of every count.
_REPRESENTATIVE_SEED_BY_ACTIVE_FIRE_COUNT = {1: 2, 2: 1, 3: 5, 4: 9}

INCIDENT_ID = "incident-carmel-01"
INCIDENT = SimulatedIncident(
    incident_id=INCIDENT_ID,
    scenario_type=ScenarioType.ACTIVE_FIRE,
    location=CARMEL_LOCATION,
)


def make_event(
    offset_seconds: int,
    event_type: SimulationEventType,
    incident_id: str = INCIDENT_ID,
    source_event_index: int = 0,
) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=offset_seconds,
        event_type=event_type,
        incident_id=incident_id,
        source_event_index=source_event_index,
    )


def summarize_events(scenario: SimulationScenario) -> list[tuple[int, SimulationEventType, str, int]]:
    return [
        (event.offset_seconds, event.event_type, event.incident_id, event.source_event_index)
        for event in scenario.events
    ]


def test_valid_120_second_scenario():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=123,
        incidents=(INCIDENT,),
        events=(make_event(0, SimulationEventType.WEATHER),),
    )

    assert scenario.duration_seconds == 120
    assert scenario.seed == 123
    assert scenario.incidents == (INCIDENT,)


@pytest.mark.parametrize("duration_seconds", [0, -1, MAX_SCENARIO_DURATION_SECONDS + 1, None, "120", True])
def test_invalid_duration_is_rejected(duration_seconds):
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=duration_seconds,
            seed=42,
            incidents=(INCIDENT,),
            events=(),
        )


def test_invalid_seed_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=True,
            incidents=(INCIDENT,),
            events=(),
        )


def test_invalid_incident_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=("incident-carmel-01",),
            events=(),
        )


def test_duplicate_incident_ids_are_rejected():
    duplicate = SimulatedIncident(
        incident_id=INCIDENT_ID,
        scenario_type=ScenarioType.HIGH_RISK_NO_FIRE,
        location=GOLAN_LOCATION,
    )

    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=(INCIDENT, duplicate),
            events=(),
        )


def test_event_beyond_duration_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=20,
            seed=42,
            incidents=(INCIDENT,),
            events=(make_event(21, SimulationEventType.NEWS),),
        )


def test_event_referencing_unknown_incident_is_rejected():
    with pytest.raises(ValueError):
        SimulationScenario(
            duration_seconds=120,
            seed=42,
            incidents=(INCIDENT,),
            events=(make_event(20, SimulationEventType.SATELLITE, incident_id="incident-golan-01"),),
        )


def test_scenario_with_no_events_is_allowed():
    scenario = SimulationScenario(
        duration_seconds=30,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    assert scenario.events == ()
    assert scenario.incidents == (INCIDENT,)


def test_events_are_sorted_chronologically():
    weather = make_event(65, SimulationEventType.WEATHER, source_event_index=1)
    satellite = make_event(20, SimulationEventType.SATELLITE)
    news = make_event(40, SimulationEventType.NEWS)

    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(weather, satellite, news),
    )

    assert scenario.events == (satellite, news, weather)


def test_events_with_same_offset_keep_stable_order():
    first = make_event(20, SimulationEventType.WEATHER)
    second = make_event(20, SimulationEventType.SATELLITE)
    third = make_event(20, SimulationEventType.NEWS)

    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(first, second, third),
    )

    assert scenario.events == (first, second, third)


def test_get_incident_returns_matching_incident():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    assert scenario.get_incident(INCIDENT_ID) == INCIDENT


def test_get_incident_rejects_unknown_id():
    scenario = SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=(),
    )

    with pytest.raises(KeyError):
        scenario.get_incident("incident-golan-01")


def test_seed_is_preserved():
    scenario = build_active_fire_scenario(seed=999)

    assert scenario.seed == 999


def test_active_fire_default_timeline():
    scenario = build_active_fire_scenario(
        location=CARMEL_LOCATION,
        incident_id=INCIDENT_ID,
    )

    assert scenario.incidents == (INCIDENT,)
    assert scenario.duration_seconds == LEGACY_FIXED_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, INCIDENT_ID, 0),
        (20, SimulationEventType.SATELLITE, INCIDENT_ID, 0),
        (40, SimulationEventType.NEWS, INCIDENT_ID, 0),
        (65, SimulationEventType.WEATHER, INCIDENT_ID, 1),
        (80, SimulationEventType.SATELLITE, INCIDENT_ID, 1),
        (100, SimulationEventType.NEWS, INCIDENT_ID, 1),
    ]


def test_low_risk_no_fire_default_timeline_is_weather_only():
    scenario = build_low_risk_no_fire_scenario(
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-low",
    )

    assert scenario.incidents[0].scenario_type is ScenarioType.LOW_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION
    assert scenario.duration_seconds == LEGACY_FIXED_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-golan-low", 0),
        (60, SimulationEventType.WEATHER, "incident-golan-low", 1),
    ]


def test_high_risk_no_fire_default_timeline_is_weather_only():
    scenario = build_high_risk_no_fire_scenario(
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-high",
    )

    assert scenario.incidents[0].scenario_type is ScenarioType.HIGH_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION
    assert scenario.duration_seconds == LEGACY_FIXED_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-golan-high", 0),
        (60, SimulationEventType.WEATHER, "incident-golan-high", 1),
    ]


def test_moderate_risk_no_fire_default_timeline_is_weather_only():
    scenario = build_moderate_risk_no_fire_scenario(
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-moderate",
    )

    assert scenario.incidents[0].scenario_type is ScenarioType.MODERATE_RISK_NO_FIRE
    assert scenario.incidents[0].location == GOLAN_LOCATION
    assert scenario.duration_seconds == LEGACY_FIXED_SCENARIO_DURATION_SECONDS
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-golan-moderate", 0),
        (60, SimulationEventType.WEATHER, "incident-golan-moderate", 1),
    ]


@pytest.mark.parametrize(
    ("scenario_type", "builder"),
    [
        (ScenarioType.LOW_RISK_NO_FIRE, build_low_risk_no_fire_scenario),
        (ScenarioType.MODERATE_RISK_NO_FIRE, build_moderate_risk_no_fire_scenario),
        (ScenarioType.HIGH_RISK_NO_FIRE, build_high_risk_no_fire_scenario),
        (ScenarioType.ACTIVE_FIRE, build_active_fire_scenario),
    ],
)
def test_build_scenario_dispatches_to_matching_builder(scenario_type, builder):
    assert build_scenario(
        scenario_type=scenario_type,
        location=GOLAN_LOCATION,
        seed=99,
        incident_id="incident-golan-01",
    ) == builder(location=GOLAN_LOCATION, seed=99, incident_id="incident-golan-01")


def test_build_scenario_rejects_invalid_scenario_type():
    with pytest.raises(ValueError):
        build_scenario("active_fire")


def test_carmel_golan_active_fire_scenario_builds_interleaved_incidents():
    scenario = build_carmel_golan_active_fire_scenario(seed=123)

    assert scenario.seed == 123
    assert [incident.incident_id for incident in scenario.incidents] == [
        "incident-carmel-01",
        "incident-golan-01",
    ]
    assert [incident.location for incident in scenario.incidents] == [
        CARMEL_LOCATION,
        GOLAN_LOCATION,
    ]
    assert summarize_events(scenario) == [
        (0, SimulationEventType.WEATHER, "incident-carmel-01", 0),
        (10, SimulationEventType.WEATHER, "incident-golan-01", 0),
        (20, SimulationEventType.SATELLITE, "incident-carmel-01", 0),
        (35, SimulationEventType.SATELLITE, "incident-golan-01", 0),
        (40, SimulationEventType.NEWS, "incident-carmel-01", 0),
        (55, SimulationEventType.NEWS, "incident-golan-01", 0),
        (65, SimulationEventType.WEATHER, "incident-carmel-01", 1),
        (75, SimulationEventType.WEATHER, "incident-golan-01", 1),
        (80, SimulationEventType.SATELLITE, "incident-carmel-01", 1),
        (95, SimulationEventType.SATELLITE, "incident-golan-01", 1),
        (100, SimulationEventType.NEWS, "incident-carmel-01", 1),
        (115, SimulationEventType.NEWS, "incident-golan-01", 1),
    ]


def test_operations_demo_same_seed_produces_identical_signature():
    first = build_operations_demo_scenario_signature(build_operations_demo_scenario(seed=42))
    second = build_operations_demo_scenario_signature(build_operations_demo_scenario(seed=42))

    assert first == second


def test_operations_demo_explicit_seed_42_remains_reproducible_as_a_fixed_regression_point():
    """Not a claim that seed 42 is special - just a pinned regression check
    that the exact seeded algorithm has not silently changed."""
    signature = build_operations_demo_scenario_signature(build_operations_demo_scenario(seed=42))

    assert signature.active_fire_count == 1
    assert signature.active_location_keys == ("carmel",)
    assert signature.event_schedule[0][0] == 0


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_active_fire_count_is_always_in_the_valid_range(seed):
    scenario = build_operations_demo_scenario(seed=seed)
    active_fire_count = sum(
        1 for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE
    )

    assert MIN_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT <= active_fire_count <= MAX_OPERATIONS_DEMO_ACTIVE_FIRE_COUNT
    assert active_fire_count == len(scenario.incidents) - _risk_only_incident_count(scenario)


@pytest.mark.parametrize("active_fire_count", [1, 2, 3, 4])
def test_operations_demo_representative_seeds_cover_every_active_fire_count(active_fire_count):
    """Not every seed needs a different count (Part P) - but at least one
    representative seed must exist for each of 1/2/3/4."""
    seed = _REPRESENTATIVE_SEED_BY_ACTIVE_FIRE_COUNT[active_fire_count]
    scenario = build_operations_demo_scenario(seed=seed)

    assert sum(1 for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE) == active_fire_count


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_active_locations_are_unique_and_canonical(seed):
    scenario = build_operations_demo_scenario(seed=seed)
    active_incidents = [incident for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE]
    active_locations = [incident.location for incident in active_incidents]

    # Unique - no two active-fire incidents share a canonical location.
    assert len(active_locations) == len(set(active_locations))
    # Canonical - every active location is one of the registry's own values, never invented.
    for location in active_locations:
        assert location in SIMULATION_LOCATIONS.values()
    # No duplicate active incident at the same location (Part D).
    assert len(active_incidents) == len({incident.location for incident in active_incidents})


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_every_incident_location_is_canonical(seed):
    """Every incident (active-fire AND risk-only) uses the existing
    registry - this builder never duplicates/invents coordinates."""
    scenario = build_operations_demo_scenario(seed=seed)

    for incident in scenario.incidents:
        assert incident.location in SIMULATION_LOCATIONS.values()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_risk_only_incidents_never_schedule_satellite_or_news(seed):
    """Risk-only incidents must never schedule SATELLITE/NEWS evidence that
    could cause FireDetectionAgent to open a fake FireEvent."""
    scenario = build_operations_demo_scenario(seed=seed)
    risk_only_incident_ids = {
        incident.incident_id for incident in scenario.incidents if incident.scenario_type is not ScenarioType.ACTIVE_FIRE
    }

    for event in scenario.events:
        if event.incident_id in risk_only_incident_ids:
            assert event.event_type is SimulationEventType.WEATHER


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_every_active_fire_incident_has_full_evidence_sequence(seed):
    """Every ACTIVE_FIRE incident - however many the seed picked - gets at
    least one WEATHER + SATELLITE + NEWS event, so the real Fire Detection
    pipeline has valid evidence to act on."""
    scenario = build_operations_demo_scenario(seed=seed)
    active_fire_incident_ids = [
        incident.incident_id for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE
    ]
    assert active_fire_incident_ids  # every seed produces at least one

    for incident_id in active_fire_incident_ids:
        event_types = {event.event_type for event in scenario.events if event.incident_id == incident_id}
        assert SimulationEventType.WEATHER in event_types
        assert SimulationEventType.SATELLITE in event_types
        assert SimulationEventType.NEWS in event_types


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_active_fire_evidence_stays_within_fire_detection_correlation_window(seed):
    """SATELLITE/NEWS for the SAME incident must land close enough in
    simulated time to correlate under the existing (unmodified) Fire
    Detection time window, regardless of randomized ordering/timing."""
    from src.calculators.fire_detection.fire_detection_config import MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES

    scenario = build_operations_demo_scenario(seed=seed)
    active_fire_incident_ids = {
        incident.incident_id for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE
    }

    for incident_id in active_fire_incident_ids:
        evidence_offsets = [
            event.offset_seconds
            for event in scenario.events
            if event.incident_id == incident_id and event.event_type in (SimulationEventType.SATELLITE, SimulationEventType.NEWS)
        ]
        assert max(evidence_offsets) - min(evidence_offsets) <= MAX_EVIDENCE_TIME_DIFFERENCE_MINUTES * 60


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_source_event_gaps_after_the_first_are_in_the_requested_range(seed):
    scenario = build_operations_demo_scenario(seed=seed)
    offsets = sorted(event.offset_seconds for event in scenario.events)

    assert offsets[0] == 0  # the first event may land at T+0
    gaps = [offsets[index] - offsets[index - 1] for index in range(1, len(offsets))]
    for gap in gaps:
        assert _MIN_SOURCE_EVENT_GAP_SECONDS <= gap <= _MAX_SOURCE_EVENT_GAP_SECONDS
    # Gaps are not all identical - genuine seeded variation, not a fixed stagger.
    assert len(set(gaps)) > 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 9, 42, 100, 999])
def test_operations_demo_duration_covers_the_final_event_plus_completion_margin(seed):
    scenario = build_operations_demo_scenario(seed=seed)
    final_offset = max(event.offset_seconds for event in scenario.events)

    assert scenario.duration_seconds == final_offset + _OPERATIONS_DEMO_COMPLETION_MARGIN_SECONDS
    assert all(event.offset_seconds <= scenario.duration_seconds for event in scenario.events)
    assert scenario.duration_seconds <= MAX_SCENARIO_DURATION_SECONDS


def test_operations_demo_different_seeds_can_produce_meaningfully_different_scenarios():
    """Not every pair of seeds must differ (Part P) - but representative
    seeds together must demonstrate real variation in count/locations/order."""
    signature_a = build_operations_demo_scenario_signature(build_operations_demo_scenario(seed=1))
    signature_b = build_operations_demo_scenario_signature(build_operations_demo_scenario(seed=9))

    assert signature_a != signature_b
    assert signature_a.active_fire_count != signature_b.active_fire_count


def _risk_only_incident_count(scenario: SimulationScenario) -> int:
    return sum(1 for incident in scenario.incidents if incident.scenario_type is not ScenarioType.ACTIVE_FIRE)


def test_active_fire_resource_refresh_scenario_adds_resource_status_cycle():
    scenario = build_active_fire_resource_refresh_scenario(seed=42)

    resource_events = [event for event in scenario.events if event.event_type is SimulationEventType.RESOURCE_STATUS]

    assert [event.offset_seconds for event in resource_events] == [60, 90]
    assert resource_events[0].resource_status_change.new_status is ResourceStatus.UNAVAILABLE
    assert resource_events[1].resource_status_change.new_status is ResourceStatus.AVAILABLE
    assert resource_events[0].resource_status_change.selection_key == resource_events[1].resource_status_change.selection_key
