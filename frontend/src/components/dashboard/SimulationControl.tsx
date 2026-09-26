import { useState } from "react";
import { DEMO_SIMULATION_REQUEST } from "../../config/demoSimulation";
import { rememberDemoRun } from "../../hooks/demoSession";
import { useStartSimulation } from "../../hooks/useStartSimulation";
import { stopSimulation } from "../../api/simulation";
import { ApiError } from "../../api/errors";
import {
  describeSimulationControl,
  describeSimulationOutcome,
  isSimulationActive,
} from "./simulationStatusPresentation";
import type { SimulationRunStatus } from "../../types/simulation";
import "./SimulationControl.css";

const STOP_ERROR_MESSAGE = "Unable to stop the simulation. Please try again.";

export interface SimulationControlProps {
  /** `null`/idle/finished -> Start Simulation; PREPARING/RUNNING -> Stop Simulation. */
  run: SimulationRunStatus | null;
  /** A6's `simulation.enabled` - when false, the demo action is hidden entirely (Task A9, Part 4). */
  enabled: boolean;
  /** Called after a successful 202 AND after a 409/403 rejection, so the dashboard picks up authoritative backend state (Task A9, Part 12/13/16). Never a second polling loop - this only nudges the existing `useOperationsOverview` refresh. */
  onRequestOverviewRefresh: () => void;
}

/**
 * The Operations Overview's one demo action (Task A9 + presentation pacing):
 * a single button that is
 * - Start Simulation when no run is active (never started, completed,
 *   failed or stopped): `DEMO_SIMULATION_REQUEST`, which always runs the
 *   mandatory demo-state reset first;
 * - Stop Simulation while a run is PREPARING/RUNNING: ends the run before
 *   its next scheduled event; generated data stays for inspection.
 * No preset picker, seed field, or reset checkbox. Owns only mutation-local
 * state; long-lived run state is always read from the `run` prop (A6
 * overview polling), never mirrored into a competing local state machine.
 */
export function SimulationControl({ run, enabled, onRequestOverviewRefresh }: SimulationControlProps) {
  const { start, isStarting, error } = useStartSimulation();
  const [stopState, setStopState] = useState<{ runId: string | null; error: string | null } | null>(null);

  if (!enabled) {
    return null;
  }

  const presentation = describeSimulationControl(run, isStarting);
  const active = isSimulationActive(run);
  const outcome = describeSimulationOutcome(run);
  // Stopping as soon as the click is sent (immediate feedback), and for as
  // long as the backend reports STOPPING (the event in progress finishing).
  const isStopping =
    run?.state === "stopping" ||
    (active && stopState !== null && stopState.runId === run?.run_id && stopState.error === null);
  const outcomeText = isStopping ? "Stopping simulation..." : outcome;

  const handleStart = () => {
    setStopState(null);
    start(DEMO_SIMULATION_REQUEST)
      .then((status) => {
        // This tab started the run: keep showing its results after it completes.
        rememberDemoRun(status.run_id);
        onRequestOverviewRefresh();
      })
      .catch((caught: unknown) => {
        // A synthetic "already in flight" rejection from a double click never
        // reaches here as a distinct case - the hook's own error state
        // already reflects the real in-flight request's outcome, and this
        // guard just decides whether THIS particular failure warrants an
        // overview refresh (409/403 mean backend state moved on without us).
        if (caught instanceof ApiError && (caught.status === 409 || caught.status === 403)) {
          onRequestOverviewRefresh();
        }
      });
  };

  const handleStop = () => {
    const runId = run?.run_id ?? null;
    setStopState({ runId, error: null });
    stopSimulation()
      .then(() => onRequestOverviewRefresh())
      .catch((caught: unknown) => {
        if (caught instanceof ApiError && caught.status === 409) {
          // Already finished on the backend - just pick up the real state.
          onRequestOverviewRefresh();
          return;
        }
        setStopState({ runId, error: STOP_ERROR_MESSAGE });
      });
  };

  return (
    <div className="simulation-control">
      {active && !isStarting ? (
        <button
          type="button"
          className="simulation-control__button simulation-control__button--stop"
          onClick={handleStop}
          disabled={isStopping}
          aria-busy={isStopping}
        >
          {isStopping ? "Stopping…" : "Stop Simulation"}
        </button>
      ) : (
        <button
          type="button"
          className="simulation-control__button"
          onClick={handleStart}
          disabled={presentation.disabled}
          aria-busy={presentation.busy}
        >
          {presentation.label}
        </button>
      )}
      {outcomeText ? (
        <p className="simulation-control__outcome" role="status">
          {outcomeText}
        </p>
      ) : null}
      {error ? <p className="simulation-control__error">{error.message}</p> : null}
      {stopState?.error ? <p className="simulation-control__error">{stopState.error}</p> : null}
    </div>
  );
}
