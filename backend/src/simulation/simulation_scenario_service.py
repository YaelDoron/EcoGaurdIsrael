"""In-memory execution service for simulation timelines."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from enum import Enum

from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_scenario import SimulationScenario

logger = logging.getLogger(__name__)

Clock = Callable[[], float]
SimulationEventHandler = Callable[[SimulationEvent], None]


class SimulationMode(Enum):
    """Simulation timeline execution mode."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


class SimulationScenarioService:
    """Coordinates simulation events without source-specific payload logic."""

    def __init__(
        self,
        clock: Clock | None = None,
        handlers: Mapping[SimulationEventType, SimulationEventHandler] | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._handlers = dict(handlers or {})
        self._scenario: SimulationScenario | None = None
        self._mode: SimulationMode | None = None
        self._start_time: float | None = None
        self._current_event_index = 0
        self._is_started = False

    @property
    def current_scenario(self) -> SimulationScenario | None:
        return self._scenario

    @property
    def mode(self) -> SimulationMode | None:
        return self._mode

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def is_finished(self) -> bool:
        return self._is_started and self._scenario is not None and self._current_event_index >= len(
            self._scenario.events
        )

    @property
    def current_event_index(self) -> int:
        return self._current_event_index

    def start(self, scenario: SimulationScenario, mode: SimulationMode = SimulationMode.AUTOMATIC) -> None:
        """Start a scenario in automatic or manual mode."""
        if not isinstance(scenario, SimulationScenario):
            raise ValueError(f"scenario must be a SimulationScenario, got {scenario!r}")
        if not isinstance(mode, SimulationMode):
            raise ValueError(f"mode must be a SimulationMode, got {mode!r}")
        if self._is_started and not self.is_finished:
            raise RuntimeError("Cannot start a new scenario while the current scenario is active.")

        self._scenario = scenario
        self._mode = mode
        self._start_time = self._clock()
        self._current_event_index = 0
        self._is_started = True
        logger.info("Simulation scenario started: %s incident(s) (%s)", len(scenario.incidents), mode.value)

        if self.is_finished:
            logger.info("Simulation scenario completed: no events")

    def advance(self) -> SimulationEvent | None:
        """Manually advance to the next event, returning None after completion."""
        self._ensure_started()
        if self.is_finished:
            return None

        event = self._scenario.events[self._current_event_index]
        self._current_event_index += 1
        self._dispatch(event)
        logger.info("Simulation event manually advanced: %s at +%ss", event.event_type.value, event.offset_seconds)
        self._log_if_finished()
        return event

    def get_due_events(self) -> list[SimulationEvent]:
        """Return all automatic-mode events due by the current elapsed time."""
        self._ensure_started()
        if self._mode is not SimulationMode.AUTOMATIC:
            raise RuntimeError("get_due_events() is only available in automatic mode.")
        if self.is_finished:
            return []

        elapsed_seconds = self._clock() - self._start_time
        due_events: list[SimulationEvent] = []

        while not self.is_finished:
            event = self._scenario.events[self._current_event_index]
            if event.offset_seconds > elapsed_seconds:
                break

            self._current_event_index += 1
            self._dispatch(event)
            due_events.append(event)
            logger.info("Simulation event became due: %s at +%ss", event.event_type.value, event.offset_seconds)

        self._log_if_finished()
        return due_events

    def _ensure_started(self) -> None:
        if not self._is_started or self._scenario is None:
            raise RuntimeError("Simulation scenario has not been started.")

    def _dispatch(self, event: SimulationEvent) -> None:
        handler = self._handlers.get(event.event_type)
        if handler is not None:
            handler(event)

    def _log_if_finished(self) -> None:
        if self.is_finished:
            logger.info("Simulation scenario completed")
