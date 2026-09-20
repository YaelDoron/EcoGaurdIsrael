import type { PresentationEntry } from "../status/presentation";
import type { OperationsSimulationSummary } from "../../types/operationsOverview";
import type { SimulationRunStatus } from "../../types/simulation";

const NO_SIMULATION: PresentationEntry = { label: "No Simulation Running", tone: "neutral" };

/**
 * Task A8, Part 22: a small header indicator built ONLY from A6's
 * `simulation` field - no Start Simulation button/control here (that is
 * A9), and no frontend timer/progress estimate. `events_completed` is the
 * backend's own persisted counter, never recomputed.
 */
export function describeSimulation(summary: OperationsSimulationSummary): PresentationEntry {
  if (!summary.enabled || summary.run === null) {
    return NO_SIMULATION;
  }

  const run = summary.run;

  switch (run.state) {
    case "idle":
      return NO_SIMULATION;
    case "preparing":
      return { label: "Preparing Simulation", tone: "warning" };
    case "running": {
      if (run.events_total !== null && run.events_total > 0) {
        const current = Math.min(run.events_completed + 1, run.events_total);
        return { label: `Running - Event ${current} of ${run.events_total}`, tone: "warning" };
      }
      return { label: "Running", tone: "warning" };
    }
    case "completed":
      return { label: "Completed", tone: "success" };
    case "completed_with_errors":
      return { label: "Completed with Errors", tone: "warning" };
    case "failed":
      return { label: "Failed", tone: "danger" };
  }
}

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

/**
 * Task A9, Part 8/9: a small factual progress line built only from fields
 * A6/A3 actually expose - never a fabricated location from `incident_id`,
 * never a percentage/ETA derived from `simulation_duration_seconds`. `null`
 * when there is nothing meaningful to show (not preparing/running, or the
 * backend hasn't reported a current event/elapsed time yet).
 */
export function describeSimulationProgress(run: SimulationRunStatus | null): string | null {
  if (run === null || (run.state !== "preparing" && run.state !== "running")) {
    return null;
  }

  const parts: string[] = [];
  if (run.current_event !== null) {
    parts.push(`Current event: ${run.current_event.event_type}`);
    parts.push(`Current incident: ${run.current_event.incident_id}`);
  }
  if (run.wall_clock_elapsed_seconds !== null) {
    parts.push(`Elapsed: ${Math.round(run.wall_clock_elapsed_seconds)}s`);
  }

  return parts.length > 0 ? parts.join(" · ") : null;
}
