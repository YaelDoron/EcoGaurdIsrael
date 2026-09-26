import type { StartSimulationRequest } from "../types/simulation";

/**
 * The single demo-workflow request body behind the dashboard's one
 * "Start Simulation" action (there is no preset picker, seed field, or
 * reset checkbox in the UI, so this is the one place those values are
 * decided):
 * - `preset: "presentation_demo"` - the paced, stoppable 30-minute
 *   presentation timeline (quiet start, two fires confirmed within the
 *   first ~90 s, steady updates afterwards until Stop Simulation).
 * - `seed` is deliberately OMITTED - the backend pins the preset's approved
 *   seed (SimulationPreset.default_seed), so every run replays the same
 *   values, AI outcomes and story. The frontend never generates
 *   locations/schedules/seeds itself. The randomized `operations_demo`
 *   preset remains available through the API/CLI for exploratory runs.
 * - `reset_demo_state: true` gives a clean runtime dataset every time,
 *   without exposing a reset control - the backend's DemoStateResetService
 *   preserves road-network cache/stations/resources on its own.
 *
 * Every Start Simulation call must use this constant unchanged - never
 * reconstruct an equivalent object inline.
 */
export const DEMO_SIMULATION_REQUEST: StartSimulationRequest = {
  preset: "presentation_demo",
  reset_demo_state: true,
};
