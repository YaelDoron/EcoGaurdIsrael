"""Simulation Control endpoints (Task A3).

A thin HTTP transport over `SimulationRunManager`
(src/services/simulation_control/simulation_run_manager.py): no simulation
execution, coordinator construction, or event-loop logic happens in this
module - the manager already owns all of that, and it in turn only ever
calls the one authoritative `DemoSimulationRunner` (Task A2). No
routing/optimization/scoring/allocation logic happens here either.

Every endpoint below is gated by `settings.ENABLE_SIMULATION_CONTROL_API`
(default `false`) - checked fresh on every request, not once at import time,
so a test/deployment can flip it without restarting the process. This flag
is deliberately independent of `ENABLE_DEMO_DATA_RESET` (see
src/config/settings.py): a deployment can allow simulation control without
allowing destructive demo-state resets, and `POST .../runs` re-checks the
reset flag itself (via SimulationRunManager) whenever `reset_demo_state=True`
is requested.

All error responses use the shared Epic 6 API error envelope
(`{"error": {"code": ..., "message": ...}}`, see
src.api.routers.response_plans._response_plan_not_found_response) rather
than FastAPI's default `{"detail": ...}` shape, returned as a plain
`JSONResponse` bypassing each endpoint's declared `response_model`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_simulation_run_manager
from src.api.schemas.simulation_control import (
    SimulationPresetResponse,
    SimulationPresetsResponse,
    SimulationRunStatusResponse,
    StartSimulationRequest,
    to_simulation_run_status_response,
)
from src.services.simulation_control.simulation_run_manager import (
    SimulationAlreadyRunningError,
    SimulationPresetNotFoundError,
    SimulationResetDisabledError,
    SimulationResetRequiredError,
    SimulationRunManager,
)
from src.simulation.simulation_presets import SIMULATION_PRESETS

simulation_router = APIRouter(prefix="/simulation", tags=["simulation"])

SIMULATION_CONTROL_DISABLED_CODE = "SIMULATION_CONTROL_DISABLED"
SIMULATION_CONTROL_DISABLED_MESSAGE = "The Simulation Control API is not enabled on this deployment."
SIMULATION_ALREADY_RUNNING_CODE = "SIMULATION_ALREADY_RUNNING"
SIMULATION_PRESET_NOT_FOUND_CODE = "SIMULATION_PRESET_NOT_FOUND"
SIMULATION_RESET_DISABLED_CODE = "SIMULATION_RESET_DISABLED"
SIMULATION_RESET_REQUIRED_CODE = "SIMULATION_RESET_REQUIRED"
SIMULATION_START_FAILED_CODE = "SIMULATION_START_FAILED"


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def _disabled_response() -> JSONResponse:
    return _error_response(403, SIMULATION_CONTROL_DISABLED_CODE, SIMULATION_CONTROL_DISABLED_MESSAGE)


def _is_simulation_control_enabled() -> bool:
    # Imported lazily (not at module top) so a test/deployment can flip the
    # flag between requests without this module having cached a stale value
    # at import time - matches SimulationRunManager.start_run's own
    # lazy-settings-import rationale.
    from src.config.settings import settings

    return settings.ENABLE_SIMULATION_CONTROL_API


@simulation_router.get(
    "/presets",
    response_model=SimulationPresetsResponse,
    summary="List available simulation presets",
)
def list_simulation_presets() -> SimulationPresetsResponse | JSONResponse:
    """List every registered preset, including its simulated (not wall-clock) duration."""
    if not _is_simulation_control_enabled():
        return _disabled_response()
    return SimulationPresetsResponse(
        presets=[
            SimulationPresetResponse(
                id=preset.id,
                display_name=preset.display_name,
                simulation_duration_seconds=preset.simulation_duration_seconds,
            )
            for preset in SIMULATION_PRESETS
        ]
    )


@simulation_router.get(
    "/runs/current",
    response_model=SimulationRunStatusResponse,
    summary="Get the current or most recently finished simulation run",
)
def get_current_simulation_run(
    manager: SimulationRunManager = Depends(get_simulation_run_manager),
) -> SimulationRunStatusResponse | JSONResponse:
    """Return the manager's current snapshot - state=IDLE (never 404) if no run has ever started."""
    if not _is_simulation_control_enabled():
        return _disabled_response()
    return to_simulation_run_status_response(manager.get_current_snapshot())


@simulation_router.post(
    "/runs",
    response_model=SimulationRunStatusResponse,
    status_code=202,
    summary="Start a new simulation run in the background",
)
def start_simulation_run(
    request: StartSimulationRequest,
    manager: SimulationRunManager = Depends(get_simulation_run_manager),
) -> SimulationRunStatusResponse | JSONResponse:
    """Reserve and start a new run, returning immediately (202) without waiting for it to finish.

    Poll GET /runs/current for progress. Mode is always "automatic" - manual
    step-through mode is CLI-only (scripts/run_demo_simulation.py).
    """
    if not _is_simulation_control_enabled():
        return _disabled_response()

    if request.reset_demo_state is not True:
        return _error_response(
            422,
            SIMULATION_RESET_REQUIRED_CODE,
            "A demo simulation cannot start without the runtime reset; reset_demo_state must be true (or omitted).",
        )

    try:
        snapshot = manager.start_run(
            preset_id=request.preset,
            seed=request.seed,
            reset_demo_state=request.reset_demo_state,
        )
    except SimulationPresetNotFoundError as exc:
        return _error_response(404, SIMULATION_PRESET_NOT_FOUND_CODE, str(exc))
    except SimulationResetRequiredError as exc:
        return _error_response(422, SIMULATION_RESET_REQUIRED_CODE, str(exc))
    except SimulationResetDisabledError as exc:
        return _error_response(403, SIMULATION_RESET_DISABLED_CODE, str(exc))
    except SimulationAlreadyRunningError as exc:
        return _error_response(409, SIMULATION_ALREADY_RUNNING_CODE, str(exc))
    except Exception:  # noqa: BLE001 - never leak a raw traceback/credential to the API response.
        return _error_response(
            500,
            SIMULATION_START_FAILED_CODE,
            "Failed to start the simulation run.",
        )

    return to_simulation_run_status_response(snapshot)
