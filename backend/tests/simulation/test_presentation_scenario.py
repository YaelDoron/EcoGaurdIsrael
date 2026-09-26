"""The paced `presentation_demo` timeline (schedule shape, determinism, logical evidence order)."""
from __future__ import annotations

from collections import defaultdict

import pytest

from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.presentation_scenario import (
    GALILEE,
    JUDEAN_HILLS,
    PRESENTATION_DEFAULT_SEED,
    PRESENTATION_DURATION_SECONDS,
    build_presentation_demo_scenario,
)
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_event_executor import simulation_event_seed_key
from src.simulation.simulation_locations import SIMULATION_LOCATIONS
from src.simulation.simulation_presets import get_simulation_preset
from src.simulation.simulation_scenario import build_operations_demo_scenario

OLD_OPERATIONS_DEMO_WALL_CLOCK_SECONDS = 488  # the previous presentation run's total length
INITIAL_QUIET_SECONDS = 3


@pytest.fixture(scope="module")
def scenario():
    return build_presentation_demo_scenario()


def _active_fire_ids(scenario) -> set[str]:
    return {incident.incident_id for incident in scenario.incidents if incident.scenario_type is ScenarioType.ACTIVE_FIRE}


def _first(scenario, incident_id: str, event_type: SimulationEventType) -> int:
    return min(e.offset_seconds for e in scenario.events if e.incident_id == incident_id and e.event_type is event_type)


def test_nothing_happens_during_the_initial_quiet_period_then_one_update_around_t4(scenario):
    offsets = [event.offset_seconds for event in scenario.events]

    assert min(offsets) > INITIAL_QUIET_SECONDS
    assert min(offsets) == 4
    assert offsets.count(4) == 1  # one visible update, never a burst
    assert scenario.events[0].event_type is SimulationEventType.WEATHER  # environment first


def _satellite_passes(scenario, incident_id: str) -> list[int]:
    return sorted(
        e.offset_seconds for e in scenario.events if e.incident_id == incident_id and e.event_type is SimulationEventType.SATELLITE
    )


# Each confirmation runs severity/spread/targets/planning inside the event
# (~30-36 s observed live) before the next event is processed.
CONFIRMATION_PROCESSING_BUDGET_SECONDS = 36


def test_two_fires_have_both_passes_needed_for_confirmation_within_90_seconds(scenario):
    judean, galilee = _satellite_passes(scenario, JUDEAN_HILLS), _satellite_passes(scenario, GALILEE)

    # First pass (SUSPECTED) early, second pass (earliest possible CONFIRMED) soon after.
    assert judean[:2] == [19, 35]
    assert galilee[:2] == [24, 41]
    # Worst case: the second confirmation waits for the first one's downstream work,
    # so nothing else may be scheduled between the two confirming passes.
    assert judean[1] + CONFIRMATION_PROCESSING_BUDGET_SECONDS + 10 <= 90
    between = [e for e in scenario.events if judean[1] < e.offset_seconds < galilee[1]]
    assert between == []
    # Fire A: news corroborates between its passes (never together with the first);
    # fire B: its news follows its confirmation (a news event takes ~10 s live, so
    # it is kept out of the window before the second confirmation).
    judean_news = _first(scenario, JUDEAN_HILLS, SimulationEventType.NEWS)
    assert judean[0] < judean_news < judean[1]
    assert _first(scenario, GALILEE, SimulationEventType.NEWS) > galilee[1]


def test_every_fire_has_a_suspected_phase_before_its_second_pass(scenario):
    # One hotspot per pass and AI Hybrid V5 needs >= 2 current pixels to confirm,
    # so the second pass is the earliest possible confirmation.
    for incident_id in _active_fire_ids(scenario):
        passes = sorted(
            e.offset_seconds for e in scenario.events if e.incident_id == incident_id and e.event_type is SimulationEventType.SATELLITE
        )
        assert len(passes) >= 2, incident_id
        # A visible SUSPECTED phase: the confirming pass never lands on top of the first one.
        assert passes[1] - passes[0] >= 12, incident_id


def test_evidence_order_is_logical_per_incident(scenario):
    for incident_id in _active_fire_ids(scenario):
        first_satellite = _first(scenario, incident_id, SimulationEventType.SATELLITE)
        first_news = _first(scenario, incident_id, SimulationEventType.NEWS)
        first_weather = _first(scenario, incident_id, SimulationEventType.WEATHER)
        assert first_weather < first_satellite < first_news, incident_id
    for incident in scenario.incidents:
        if incident.scenario_type is not ScenarioType.ACTIVE_FIRE:
            kinds = {e.event_type for e in scenario.events if e.incident_id == incident.incident_id}
            assert kinds == {SimulationEventType.WEATHER}, incident.incident_id


def test_updates_are_paced_not_flooded(scenario):
    offsets = [event.offset_seconds for event in scenario.events]
    gaps = [later - earlier for earlier, later in zip(offsets, offsets[1:])]

    assert len(set(offsets)) == len(offsets)  # never two updates at the same moment
    assert min(gaps) >= 5  # the dense opening still leaves several seconds between updates
    assert max(gaps) <= 45


def test_run_keeps_producing_updates_far_beyond_the_old_run_length(scenario):
    assert scenario.duration_seconds == PRESENTATION_DURATION_SECONDS == 1800
    late = [event for event in scenario.events if event.offset_seconds > OLD_OPERATIONS_DEMO_WALL_CLOCK_SECONDS]
    assert len(late) >= 40
    # After minute 5 and after minute 8 every active fire still receives new evidence.
    for cutoff in (300, 480):
        for incident_id in _active_fire_ids(scenario):
            assert any(e.incident_id == incident_id and e.offset_seconds > cutoff for e in scenario.events)
    last_minute = [e for e in scenario.events if e.offset_seconds > PRESENTATION_DURATION_SECONDS - 60]
    assert last_minute


def test_new_incidents_are_rare_and_concurrent_fires_stay_within_four(scenario):
    first_evidence = sorted(_first(scenario, incident_id, SimulationEventType.SATELLITE) for incident_id in _active_fire_ids(scenario))

    assert len(first_evidence) == 4
    assert first_evidence[2] >= 150 and first_evidence[3] - first_evidence[2] >= 120  # new incidents minutes apart


def test_only_canonical_locations_are_used(scenario):
    canonical = set(SIMULATION_LOCATIONS.values())
    assert {incident.location for incident in scenario.incidents} <= canonical


def test_schedule_is_identical_for_every_seed_and_values_are_seed_driven():
    first = build_presentation_demo_scenario(seed=1)
    second = build_presentation_demo_scenario(seed=2)

    schedule = lambda s: [(e.offset_seconds, e.event_type, e.incident_id, e.source_event_index) for e in s.events]  # noqa: E731
    assert schedule(first) == schedule(second)
    event = next(e for e in first.events if e.event_type is SimulationEventType.WEATHER)
    incident = first.get_incident(event.incident_id)
    from datetime import datetime, timezone

    at = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
    key = simulation_event_seed_key(event)
    assert WeatherDataGenerator(1).generate(incident.scenario_type, at, incident.location, seed_key=key) != (
        WeatherDataGenerator(2).generate(incident.scenario_type, at, incident.location, seed_key=key)
    )


def test_same_seed_reproduces_every_generated_value_regardless_of_start_time():
    from datetime import datetime, timedelta, timezone

    def values(start):
        scenario = build_presentation_demo_scenario(PRESENTATION_DEFAULT_SEED)
        weather, satellite, news = (
            WeatherDataGenerator(scenario.seed),
            SatelliteDataGenerator(scenario.seed),
            NewsDataGenerator(scenario.seed),
        )
        out = []
        for event in scenario.events[:30]:
            incident = scenario.get_incident(event.incident_id)
            at = start + timedelta(seconds=event.offset_seconds)
            key = simulation_event_seed_key(event)
            if event.event_type is SimulationEventType.WEATHER:
                generated = weather.generate(incident.scenario_type, at, incident.location, seed_key=key)
                out.append([(m.observation.temperature, m.observation.relative_humidity, m.observation.wind_speed) for m in generated.measurements])
            elif event.event_type is SimulationEventType.SATELLITE:
                generated = satellite.generate(incident.scenario_type, at, incident.location, seed_key=key)
                out.append([(h.latitude, h.longitude, h.frp, h.brightness, h.confidence) for h in generated.hotspots])
            else:
                generated = news.generate(incident.scenario_type, at, incident.location, report_index=event.source_event_index, seed_key=key)
                out.append([(r.title, r.latitude, r.longitude) for r in generated.reports])
        return out

    assert values(datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)) == values(datetime(2026, 10, 3, 14, 37, tzinfo=timezone.utc))


def test_presentation_preset_is_registered_with_its_pinned_seed_and_operations_demo_is_unchanged():
    preset = get_simulation_preset("presentation_demo")

    assert preset is not None
    assert preset.default_seed == PRESENTATION_DEFAULT_SEED
    assert preset.simulation_duration_seconds == PRESENTATION_DURATION_SECONDS
    assert get_simulation_preset("operations_demo").default_seed is None
    # operations_demo keeps its own seeded composition (its seed-893 shape is unchanged).
    assert build_operations_demo_scenario(893).duration_seconds == 411


def test_each_incident_event_type_is_indexed_consecutively(scenario):
    seen: dict[tuple[str, SimulationEventType], list[int]] = defaultdict(list)
    for event in scenario.events:
        seen[(event.incident_id, event.event_type)].append(event.source_event_index)
    for indexes in seen.values():
        assert indexes == list(range(len(indexes)))
