"""API transport contract for Task A3 (Simulation Control API).

These Pydantic models are the HTTP request/response shape only - distinct
from `SimulationRunManager`'s own `SimulationRunSnapshot`/`SimulationRunError`
dataclasses (src.services.simulation_control.simulation_run_manager), which
remain the manager's internal, transport-agnostic state.
`to_simulation_run_status_response`/`to_simulation_run_current_event_response`
below are the one place that convert between the two - a pure field-by-field
copy, never a recalculation; both `src.api.routers.simulation` (A3) and
`src.api.routers.operations_overview` (A6, which embeds this same run status
inside its dashboard snapshot) call these instead of each re-deriving the
mapping.

`Optional[X]` is used instead of `X | None` for the same reason as
`src.api.schemas.response_plans`: Pydantic resolves annotations at
class-definition time and `X | None` (PEP 604) is not evaluable on this
project's Python 3.9 runtime for arbitrary types.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from src.services.simulation_control.simulation_run_manager import (
    SimulationCurrentEventSnapshot,
    SimulationRunSnapshot,
    SimulationRunState,
)


class SimulationPresetResponse(BaseModel):
    """One selectable simulation preset, as sent over HTTP.

    `simulation_duration_seconds` describes the preset's SIMULATED timeline
    length only - it is not a wall-clock ETA for how long a run will take to
    execute (see SimulationRunStatusResponse.wall_clock_elapsed_seconds for
    the real-clock figure of an actual run).
    """

    id: str
    display_name: str
    simulation_duration_seconds: int


class SimulationPresetsResponse(BaseModel):
    """The full list of selectable presets, as sent over HTTP."""

    presets: list[SimulationPresetResponse]


class StartSimulationRequest(BaseModel):
    """Request body for POST /api/v1/simulation/runs.

    Mode is always "automatic" through this HTTP endpoint - manual
    step-through mode remains a CLI-only concept
    (scripts/run_demo_simulation.py) and is not exposed here.

    `seed` is OPTIONAL. The normal dashboard Start/Run Again action omits
    it entirely so every run varies meaningfully (SimulationRunManager
    generates and stores a fresh one - see its own `_generate_seed`);
    passing an explicit seed remains fully supported for reproducible
    testing/debugging and always wins over auto-generation.
    """

    preset: str
    seed: Optional[int] = None
    # Task 9A: the demo reset is MANDATORY. The field is kept for client compatibility (the dashboard sends true) and
    # now DEFAULTS to true; an explicit false is rejected with 422 SIMULATION_RESET_REQUIRED - a run can no longer
    # be started on top of a previous run's FireEvents/evidence/history.
    reset_demo_state: bool = True


class SimulationRunCurrentEventResponse(BaseModel):
    """The most recently started-or-completed event, as sent over HTTP."""

    event_index: int
    incident_id: str
    event_type: str
    timestamp_offset_sec: int


class SimulationRunErrorResponse(BaseModel):
    """A safe, sanitized failure summary - never a raw exception/traceback/credential."""

    code: str
    message: str


class SimulationRunStatusResponse(BaseModel):
    """The full current-or-most-recent run status, as sent over HTTP.

    Returned by GET /api/v1/simulation/runs/current even when no run has
    ever started (state=IDLE) - never a 404 - and remains queryable,
    unchanged, after a run finishes until a new run is started.
    """

    run_id: Optional[str]
    state: SimulationRunState
    preset_id: Optional[str]
    seed: Optional[int]
    mode: str
    simulation_duration_seconds: Optional[int]
    events_total: Optional[int]
    events_completed: int
    events_succeeded: int
    events_failed: int
    current_event: Optional[SimulationRunCurrentEventResponse]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    wall_clock_elapsed_seconds: Optional[float]
    last_message: Optional[str]
    error: Optional[SimulationRunErrorResponse]


def to_simulation_run_current_event_response(
    current_event: SimulationCurrentEventSnapshot | None,
) -> SimulationRunCurrentEventResponse | None:
    if current_event is None:
        return None
    return SimulationRunCurrentEventResponse(
        event_index=current_event.event_index,
        incident_id=current_event.incident_id,
        event_type=current_event.event_type,
        timestamp_offset_sec=current_event.timestamp_offset_sec,
    )


def to_simulation_run_status_response(snapshot: SimulationRunSnapshot) -> SimulationRunStatusResponse:
    return SimulationRunStatusResponse(
        run_id=snapshot.run_id,
        state=snapshot.state,
        preset_id=snapshot.preset_id,
        seed=snapshot.seed,
        mode=snapshot.mode,
        simulation_duration_seconds=snapshot.simulation_duration_seconds,
        events_total=snapshot.events_total,
        events_completed=snapshot.events_completed,
        events_succeeded=snapshot.events_succeeded,
        events_failed=snapshot.events_failed,
        current_event=to_simulation_run_current_event_response(snapshot.current_event),
        started_at=snapshot.started_at,
        completed_at=snapshot.completed_at,
        wall_clock_elapsed_seconds=snapshot.wall_clock_elapsed_seconds,
        last_message=snapshot.last_message,
        error=(
            SimulationRunErrorResponse(code=snapshot.error.code, message=snapshot.error.message)
            if snapshot.error is not None
            else None
        ),
    )
