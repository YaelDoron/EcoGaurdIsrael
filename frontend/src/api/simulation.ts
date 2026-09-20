import type { SimulationPresetsResponse, SimulationRunStatus, StartSimulationRequest } from "../types/simulation";
import { apiGet, apiPost } from "./client";

const SIMULATION_RUNS_PATH = "/api/v1/simulation/runs";

/**
 * A3 Simulation Control API client functions (Task A7). Thin wrappers over
 * the generic `apiGet`/`apiPost` client - no fetch call, base-URL handling,
 * retry, or caching logic lives here.
 *
 * The main dashboard must read simulation state from
 * `GET /api/v1/operations/overview`'s `simulation` field (Task A6), not
 * from `getCurrentSimulationRun` - this direct A3 status method remains
 * available for dedicated simulation-control flows/tests, not for the
 * dashboard's own polling.
 */

export function getSimulationPresets(signal?: AbortSignal): Promise<SimulationPresetsResponse> {
  return apiGet<SimulationPresetsResponse>("/api/v1/simulation/presets", { signal });
}

export function getCurrentSimulationRun(signal?: AbortSignal): Promise<SimulationRunStatus> {
  return apiGet<SimulationRunStatus>(`${SIMULATION_RUNS_PATH}/current`, { signal });
}

/**
 * Start a new simulation run. Resolves as soon as the backend responds
 * (HTTP 202) - it never waits for the simulation to finish; the returned
 * `SimulationRunStatus` reflects the just-reserved run (typically
 * `state: "preparing"`), not its eventual outcome. A concurrent run in
 * progress rejects with an `ApiError` whose `status === 409` and
 * `code === "SIMULATION_ALREADY_RUNNING"`; a disabled control API rejects
 * with `status === 403`/`code === "SIMULATION_CONTROL_DISABLED"` - both are
 * preserved on the thrown `ApiError`, not collapsed into a generic message.
 */
export function startSimulation(request: StartSimulationRequest, signal?: AbortSignal): Promise<SimulationRunStatus> {
  return apiPost<SimulationRunStatus>(SIMULATION_RUNS_PATH, request, { signal });
}
