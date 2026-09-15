"""Simulation timeline core."""

from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import (
    CARMEL_LOCATION,
    DEFAULT_CARMEL_LOCATION,
    GALILEE_LOCATION,
    GOLAN_LOCATION,
    JUDEAN_HILLS_LOCATION,
    JERUSALEM_FOREST_LOCATION,
    SIMULATION_LOCATIONS,
    get_simulation_location,
)
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import (
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    simulation_event_timestamp,
)
from src.simulation.simulated_incident import SimulatedIncident
from src.simulation.simulation_scenario import (
    MAX_SCENARIO_DURATION_SECONDS,
    SimulationScenario,
    build_active_fire_scenario,
    build_carmel_golan_active_fire_scenario,
    build_high_risk_no_fire_scenario,
    build_low_risk_no_fire_scenario,
    build_multi_incident_scenario,
    build_scenario,
)
from src.simulation.simulation_scenario_service import (
    SimulationMode,
    SimulationScenarioService,
)
from src.simulation.analysis import (
    ASSESSMENT_RADIUS_KM,
    SimulationFireDangerCoordinator,
    SimulationFireDangerResult,
    SimulationFireDetectionCoordinator,
    SimulationFireDetectionResult,
    SimulationOperationalCoordinator,
)

__all__ = [
    "ScenarioType",
    "SimulationLocation",
    "CARMEL_LOCATION",
    "DEFAULT_CARMEL_LOCATION",
    "JERUSALEM_FOREST_LOCATION",
    "GALILEE_LOCATION",
    "GOLAN_LOCATION",
    "JUDEAN_HILLS_LOCATION",
    "SIMULATION_LOCATIONS",
    "get_simulation_location",
    "SimulationEvent",
    "SimulationEventType",
    "SimulationEventExecutionResult",
    "SimulationEventExecutor",
    "simulation_event_timestamp",
    "SimulatedIncident",
    "SimulationScenario",
    "MAX_SCENARIO_DURATION_SECONDS",
    "build_low_risk_no_fire_scenario",
    "build_high_risk_no_fire_scenario",
    "build_active_fire_scenario",
    "build_multi_incident_scenario",
    "build_carmel_golan_active_fire_scenario",
    "build_scenario",
    "SimulationMode",
    "SimulationScenarioService",
    "ASSESSMENT_RADIUS_KM",
    "SimulationFireDangerCoordinator",
    "SimulationFireDangerResult",
    "SimulationFireDetectionCoordinator",
    "SimulationFireDetectionResult",
    "SimulationOperationalCoordinator",
]
