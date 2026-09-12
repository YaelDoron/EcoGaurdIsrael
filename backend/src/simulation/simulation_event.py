"""Simulation timeline event definitions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SimulationEventType(Enum):
    """The simulated source category an event belongs to."""

    WEATHER = "weather"
    SATELLITE = "satellite"
    NEWS = "news"


@dataclass(frozen=True)
class SimulationEvent:
    """One event in a simulation timeline, relative to scenario start."""

    offset_seconds: int
    event_type: SimulationEventType
    incident_id: str
    source_event_index: int

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
