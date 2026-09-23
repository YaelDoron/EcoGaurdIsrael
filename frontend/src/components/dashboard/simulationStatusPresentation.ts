import type { SimulationRunStatus } from "../../types/simulation";

export interface SimulationControlPresentation {
  label: string;
  disabled: boolean;
  busy: boolean;
}

/**
 * Task A9, Part 7/20: the Start/Run Again control's label + disabled/busy
 * state, derived only from the backend's own run state (+ the mutation's
 * local `isStarting` flag) - never a frontend-invented progress guess.
 * PREPARING/RUNNING (or an in-flight start POST) always disable the
 * control, so a second click cannot start a concurrent run; the backend's
 * own 409 remains the final authority regardless.
 */
export function describeSimulationControl(run: SimulationRunStatus | null, isStarting: boolean): SimulationControlPresentation {
  if (isStarting) {
    return { label: "Starting…", disabled: true, busy: true };
  }

  if (run === null || run.state === "idle") {
    return { label: "Start Simulation", disabled: false, busy: false };
  }

  switch (run.state) {
    case "preparing":
      return { label: "Preparing…", disabled: true, busy: true };
    case "running":
      return { label: "Running…", disabled: true, busy: true };
    case "completed":
    case "completed_with_errors":
    case "failed":
      return { label: "Run Again", disabled: false, busy: false };
  }
}
