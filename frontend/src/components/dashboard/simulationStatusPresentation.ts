import type { SimulationRunStatus } from "../../types/simulation";

export interface SimulationControlPresentation {
  label: string;
  disabled: boolean;
  busy: boolean;
}

/**
 * Task A9, Part 7/20: the Start Simulation control's label + disabled/busy
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
    // While a run is active the control shows Stop Simulation instead
    // (SimulationControl); this start state is never clickable then.
    case "preparing":
    case "running":
    case "stopping":
      return { label: "Start Simulation", disabled: true, busy: true };
    case "completed":
    case "completed_with_errors":
    case "failed":
    case "stopped":
      return { label: "Start Simulation", disabled: false, busy: false };
  }
}

/** Whether the backend run is PREPARING/RUNNING (Stop is offered only then). */
export function isSimulationActive(run: SimulationRunStatus | null): boolean {
  return run !== null && (run.state === "preparing" || run.state === "running" || run.state === "stopping");
}

/** A short, factual outcome line for a finished run (none while idle/active). */
export function describeSimulationOutcome(run: SimulationRunStatus | null): string | null {
  switch (run?.state) {
    case "stopping":
      return "Stopping simulation...";
    case "stopped":
      return "Simulation stopped";
    case "completed":
    case "completed_with_errors":
      return "Simulation completed";
    default:
      return null;
  }
}
