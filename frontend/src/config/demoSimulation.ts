import type { StartSimulationRequest } from "../types/simulation";

/**
 * The single demo-workflow request body. The product's one operator-facing
 * action is "Start Simulation" / "Run Again" - there is no preset picker,
 * seed field, or reset checkbox in the UI, so this is the one place those
 * values are decided:
 * - `preset: "operations_demo"` is the project's one intended demo scenario
 *   (Task A1).
 * - `seed` is deliberately OMITTED - every normal Start/Run Again click asks
 *   the backend to choose a fresh seed for that run, so each run varies
 *   meaningfully (active-fire count, locations, event timing). The frontend
 *   never generates locations/schedules/seeds itself - all randomness is
 *   backend-owned. An explicit seed remains supported by the API for
 *   reproducible testing/debugging (e.g. the CLI script), just not from this
 *   normal dashboard action.
 * - `reset_demo_state: true` gives a clean runtime dataset every time,
 *   without exposing a reset control - the backend's DemoStateResetService
 *   (Task A1.6) preserves road-network cache/stations/resources on its own.
 *
 * Every Start/Run Again call must use this constant unchanged - never
 * reconstruct an equivalent object inline.
 */
export const DEMO_SIMULATION_REQUEST: StartSimulationRequest = {
  preset: "operations_demo",
  reset_demo_state: true,
};
