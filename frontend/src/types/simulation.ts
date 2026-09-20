/**
 * Frontend mirror of the A3 Simulation Control API contract
 * (backend/src/api/schemas/simulation_control.py). Exact property names,
 * nullability, and enum values match that module - do not infer field
 * names from task descriptions, verify against the Pydantic source.
 *
 * `SimulationRunStatus` (the run's state) is also embedded unchanged inside
 * `GET /api/v1/operations/overview`'s `simulation.run` field (Task A6) -
 * this type is shared between both response contracts intentionally, since
 * the backend reuses the exact same `SimulationRunStatusResponse` schema
 * for both.
 */
export type SimulationRunState =
  | "idle"
  | "preparing"
  | "running"
  | "completed"
  | "completed_with_errors"
  | "failed";

export interface SimulationPreset {
  id: string;
  display_name: string;
  /** The preset's SIMULATED timeline length in seconds - not a wall-clock ETA. */
  simulation_duration_seconds: number;
}

export interface SimulationPresetsResponse {
  presets: SimulationPreset[];
}

/**
 * Request body for `POST /api/v1/simulation/runs`. Mode is always
 * "automatic" through this endpoint. `seed` is OPTIONAL - when omitted, the
 * backend chooses a new seed for that run (still fully deterministic given
 * that chosen seed) and reports it back via `SimulationRunStatus.seed`. The
 * normal dashboard Start/Run Again action always omits it; an explicit seed
 * remains supported for reproducible testing/debugging (e.g. the CLI).
 */
export interface StartSimulationRequest {
  preset: string;
  seed?: number;
  reset_demo_state: boolean;
}

export interface SimulationRunCurrentEvent {
  event_index: number;
  incident_id: string;
  event_type: string;
  timestamp_offset_sec: number;
}

/** A safe, sanitized failure summary - never a raw exception/traceback. */
export interface SimulationRunError {
  code: string;
  message: string;
}

/**
 * The full current-or-most-recent run status. Returned by
 * `GET /api/v1/simulation/runs/current` even when no run has ever started
 * (`state === "idle"`, `run_id === null`) - never a 404.
 */
export interface SimulationRunStatus {
  run_id: string | null;
  state: SimulationRunState;
  preset_id: string | null;
  seed: number | null;
  mode: string;
  simulation_duration_seconds: number | null;
  events_total: number | null;
  events_completed: number;
  events_succeeded: number;
  events_failed: number;
  current_event: SimulationRunCurrentEvent | null;
  started_at: string | null;
  completed_at: string | null;
  wall_clock_elapsed_seconds: number | null;
  last_message: string | null;
  error: SimulationRunError | null;
}
