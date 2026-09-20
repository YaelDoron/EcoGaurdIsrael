"""Canonical registry of named, ready-to-run multi-incident scenario presets.

Single source of truth for "preset id" -> scenario builder. Both the CLI
(`scripts/run_demo_simulation.py`'s `SUPPORTED_PRESETS`/`build_scenario_from_args`)
and the Simulation Control API (Task A3, `src/services/simulation_control/`)
are free to use this registry, but neither is required to be rewritten onto
it in this task - it exists so a future consolidation has one place to look,
and so Task A3 does not invent a second, API-only preset list.

Only presets with a simple `(seed) -> SimulationScenario` signature are
listed here (single-incident scenarios selected via `--scenario`/`--location`
are a separate CLI-only concept and are out of scope for API selection -
see Task A3, Part 8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.simulation.simulation_scenario import (
    SimulationScenario,
    build_active_fire_resource_refresh_scenario,
    build_carmel_golan_active_fire_scenario,
    build_operations_demo_scenario,
)

_METADATA_SEED = 42


@dataclass(frozen=True)
class SimulationPreset:
    """One named, ready-to-run multi-incident scenario preset."""

    id: str
    display_name: str
    build: Callable[[int], SimulationScenario]

    @property
    def simulation_duration_seconds(self) -> int:
        """The scenario's simulated timeline length - NOT a wall-clock estimate.

        Built once with a fixed metadata seed to read the real
        `duration_seconds` the preset's builder actually uses, rather than
        duplicating that number here as a second, driftable constant.
        Scenario construction is pure in-memory object assembly (no DB/network
        I/O), so this is cheap enough to compute on every metadata read.
        """
        return self.build(_METADATA_SEED).duration_seconds


SIMULATION_PRESETS: tuple[SimulationPreset, ...] = (
    SimulationPreset(
        id="operations_demo",
        display_name="Operations Demo",
        build=lambda seed: build_operations_demo_scenario(seed=seed),
    ),
    SimulationPreset(
        id="carmel_golan_active_fire",
        display_name="Carmel & Golan Active Fire",
        build=lambda seed: build_carmel_golan_active_fire_scenario(seed=seed),
    ),
    SimulationPreset(
        id="active_fire_resource_refresh",
        display_name="Active Fire Resource Refresh",
        build=lambda seed: build_active_fire_resource_refresh_scenario(seed=seed),
    ),
)

_PRESETS_BY_ID = {preset.id: preset for preset in SIMULATION_PRESETS}


def get_simulation_preset(preset_id: str) -> SimulationPreset | None:
    """Return the named preset, or None if `preset_id` is not registered."""
    return _PRESETS_BY_ID.get(preset_id)
