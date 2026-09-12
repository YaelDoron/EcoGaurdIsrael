"""Tests for SimulationScenarioService."""

import pytest

from src.simulation import (
    CARMEL_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventType,
    SimulationMode,
    SimulationScenario,
    SimulationScenarioService,
    build_carmel_golan_active_fire_scenario,
)

INCIDENT = SimulatedIncident(
    incident_id="incident-carmel-01",
    scenario_type=ScenarioType.ACTIVE_FIRE,
    location=CARMEL_LOCATION,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_event(
    offset_seconds: int,
    event_type: SimulationEventType,
    incident_id: str = INCIDENT.incident_id,
    source_event_index: int = 0,
) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=offset_seconds,
        event_type=event_type,
        incident_id=incident_id,
        source_event_index=source_event_index,
    )


def make_scenario(events=None) -> SimulationScenario:
    return SimulationScenario(
        duration_seconds=120,
        seed=42,
        incidents=(INCIDENT,),
        events=events
        if events is not None
        else (
            make_event(0, SimulationEventType.WEATHER),
            make_event(20, SimulationEventType.SATELLITE),
            make_event(40, SimulationEventType.NEWS),
            make_event(65, SimulationEventType.WEATHER, source_event_index=1),
        ),
    )


def test_start_requires_scenario():
    service = SimulationScenarioService()

    with pytest.raises(ValueError):
        service.start(None)


def test_advance_before_start_fails():
    service = SimulationScenarioService()

    with pytest.raises(RuntimeError):
        service.advance()


def test_manual_advance_returns_events_in_order_and_finishes():
    handled = []
    service = SimulationScenarioService(
        handlers={
            SimulationEventType.WEATHER: handled.append,
            SimulationEventType.SATELLITE: handled.append,
            SimulationEventType.NEWS: handled.append,
        }
    )
    scenario = make_scenario()
    service.start(scenario, mode=SimulationMode.MANUAL)

    returned_events = [service.advance(), service.advance(), service.advance(), service.advance()]

    assert returned_events == list(scenario.events)
    assert handled == list(scenario.events)
    assert service.current_event_index == len(scenario.events)
    assert service.is_finished is True
    assert service.advance() is None


def test_no_event_scenario_is_immediately_finished_after_start():
    service = SimulationScenarioService()

    service.start(make_scenario(events=()), mode=SimulationMode.MANUAL)

    assert service.is_started is True
    assert service.is_finished is True
    assert service.advance() is None


def test_double_start_while_active_fails():
    service = SimulationScenarioService()
    service.start(make_scenario(), mode=SimulationMode.MANUAL)

    with pytest.raises(RuntimeError):
        service.start(make_scenario(), mode=SimulationMode.MANUAL)


def test_starting_second_scenario_after_completion_resets_state():
    first = make_scenario(events=(make_event(0, SimulationEventType.WEATHER),))
    second = make_scenario(events=(make_event(0, SimulationEventType.NEWS),))
    service = SimulationScenarioService()
    service.start(first, mode=SimulationMode.MANUAL)
    assert service.advance() == first.events[0]
    assert service.is_finished is True

    service.start(second, mode=SimulationMode.MANUAL)

    assert service.current_scenario == second
    assert service.current_event_index == 0
    assert service.advance() == second.events[0]


def test_get_due_events_before_start_fails():
    service = SimulationScenarioService()

    with pytest.raises(RuntimeError):
        service.get_due_events()


def test_get_due_events_requires_automatic_mode():
    service = SimulationScenarioService()
    service.start(make_scenario(), mode=SimulationMode.MANUAL)

    with pytest.raises(RuntimeError):
        service.get_due_events()


def test_automatic_mode_with_fake_time_returns_due_events_once():
    clock = FakeClock(now=100.0)
    service = SimulationScenarioService(clock=clock)
    scenario = make_scenario()
    service.start(scenario, mode=SimulationMode.AUTOMATIC)

    assert service.get_due_events() == [scenario.events[0]]
    assert service.get_due_events() == []

    clock.now = 110.0
    assert service.get_due_events() == []

    clock.now = 120.0
    assert service.get_due_events() == [scenario.events[1]]
    assert service.get_due_events() == []


def test_automatic_mode_returns_multiple_events_when_time_jumps():
    clock = FakeClock(now=0.0)
    service = SimulationScenarioService(clock=clock)
    scenario = make_scenario()
    service.start(scenario, mode=SimulationMode.AUTOMATIC)

    assert service.get_due_events() == [scenario.events[0]]

    clock.now = 70.0
    due_events = service.get_due_events()

    assert due_events == [scenario.events[1], scenario.events[2], scenario.events[3]]
    assert service.is_finished is True
    assert service.get_due_events() == []


def test_automatic_mode_dispatches_handlers_for_due_events():
    clock = FakeClock(now=0.0)
    handled = []
    service = SimulationScenarioService(
        clock=clock,
        handlers={
            SimulationEventType.WEATHER: handled.append,
            SimulationEventType.SATELLITE: handled.append,
        },
    )
    scenario = make_scenario(
        events=(
            make_event(0, SimulationEventType.WEATHER),
            make_event(20, SimulationEventType.SATELLITE),
        )
    )
    service.start(scenario, mode=SimulationMode.AUTOMATIC)

    service.get_due_events()
    clock.now = 20.0
    service.get_due_events()

    assert handled == list(scenario.events)


def test_manual_mode_returns_multi_incident_events_in_order():
    service = SimulationScenarioService()
    scenario = build_carmel_golan_active_fire_scenario()
    service.start(scenario, mode=SimulationMode.MANUAL)

    returned_events = [service.advance() for _ in scenario.events]

    assert returned_events == list(scenario.events)
    assert {event.incident_id for event in returned_events} == {
        "incident-carmel-01",
        "incident-golan-01",
    }


def test_automatic_mode_returns_due_multi_incident_events_in_timeline_order():
    clock = FakeClock(now=0.0)
    service = SimulationScenarioService(clock=clock)
    scenario = build_carmel_golan_active_fire_scenario()
    service.start(scenario, mode=SimulationMode.AUTOMATIC)

    assert service.get_due_events() == [scenario.events[0]]

    clock.now = 36.0

    assert service.get_due_events() == [scenario.events[1], scenario.events[2], scenario.events[3]]
