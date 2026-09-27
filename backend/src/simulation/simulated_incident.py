"""Simulation incident definitions."""
from __future__ import annotations

from dataclasses import dataclass

from src.simulation.generators.weather_data_generator import WeatherScenarioProfile
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_location import SimulationLocation


@dataclass(frozen=True)
class SimulatedIncident:
    """One independent incident represented inside a simulation scenario."""

    incident_id: str
    scenario_type: ScenarioType
    location: SimulationLocation
    # Optional incident-specific weather ranges; None uses the scenario type's profile.
    weather_profile: WeatherScenarioProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.incident_id, str) or not self.incident_id.strip():
            raise ValueError(f"incident_id must be a non-empty string, got {self.incident_id!r}")
        if not isinstance(self.scenario_type, ScenarioType):
            raise ValueError(f"scenario_type must be a ScenarioType, got {self.scenario_type!r}")
        if not isinstance(self.location, SimulationLocation):
            raise ValueError(f"location must be a SimulationLocation, got {self.location!r}")
        if self.weather_profile is not None and not isinstance(self.weather_profile, WeatherScenarioProfile):
            raise ValueError(f"weather_profile must be a WeatherScenarioProfile or None, got {self.weather_profile!r}")
