"""Simulation timeline event definitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.simulation.simulation_resource_status_change import SimulationResourceStatusChange


class SimulationEventType(Enum):
    """The simulated source category an event belongs to."""

    WEATHER = "weather"
    SATELLITE = "satellite"
    NEWS = "news"
    RESOURCE_STATUS = "resource_status"


@dataclass(frozen=True)
class SimulationEvent:
    """One event in a simulation timeline, relative to scenario start."""

    offset_seconds: int
    event_type: SimulationEventType
    incident_id: str
    source_event_index: int
    resource_status_change: SimulationResourceStatusChange | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.offset_seconds, bool)
            or not isinstance(self.offset_seconds, int)
            or self.offset_seconds < 0
        ):
            raise ValueError(
                f"offset_seconds must be a non-negative integer, got {self.offset_seconds!r}"
            )

        if not isinstance(self.event_type, SimulationEventType):
            raise ValueError(f"event_type must be a SimulationEventType, got {self.event_type!r}")

        if not isinstance(self.incident_id, str) or not self.incident_id.strip():
            raise ValueError(f"incident_id must be a non-empty string, got {self.incident_id!r}")

        if (
            isinstance(self.source_event_index, bool)
            or not isinstance(self.source_event_index, int)
            or self.source_event_index < 0
        ):
            raise ValueError(
                "source_event_index must be a non-negative integer, "
                f"got {self.source_event_index!r}"
            )

        if self.event_type is SimulationEventType.RESOURCE_STATUS:
            if not isinstance(self.resource_status_change, SimulationResourceStatusChange):
                raise ValueError("RESOURCE_STATUS events require a SimulationResourceStatusChange payload.")
        elif self.resource_status_change is not None:
            raise ValueError(f"{self.event_type.value} events must not include resource_status_change.")
