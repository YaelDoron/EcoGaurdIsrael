import { DEMO_SIMULATION_REQUEST } from "../../config/demoSimulation";
import { useStartSimulation } from "../../hooks/useStartSimulation";
import { ApiError } from "../../api/errors";
import { describeSimulationControl } from "./simulationStatusPresentation";
import type { SimulationRunStatus } from "../../types/simulation";
import "./SimulationControl.css";

export interface SimulationControlProps {
  /** `null`/`"idle"` -> Start Simulation; PREPARING/RUNNING -> disabled; a terminal state -> Run Again. */
  run: SimulationRunStatus | null;
  /** A6's `simulation.enabled` - when false, the demo action is hidden entirely (Task A9, Part 4). */
  enabled: boolean;
  /** Called after a successful 202 AND after a 409/403 rejection, so the dashboard picks up authoritative backend state (Task A9, Part 12/13/16). Never a second polling loop - this only nudges the existing `useOperationsOverview` refresh. */
  onRequestOverviewRefresh: () => void;
}

/**
 * The Operations Overview's one demo action (Task A9): Start Simulation /
 * Run Again, using the single fixed `DEMO_SIMULATION_REQUEST` (preset
 * operations_demo, seed 42, reset_demo_state=true) - never a preset picker,
 * seed field, or reset checkbox. Owns only mutation-local state
 * (`useStartSimulation`'s isStarting/error); long-lived run state is always
 * read from the `run` prop (A6 overview polling), never mirrored into a
 * competing local state machine.
 */
export function SimulationControl({ run, enabled, onRequestOverviewRefresh }: SimulationControlProps) {
  const { start, isStarting, error } = useStartSimulation();

  if (!enabled) {
    return null;
  }

  const presentation = describeSimulationControl(run, isStarting);

  const handleClick = () => {
    start(DEMO_SIMULATION_REQUEST)
      .then(() => {
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

  return (
    <div className="simulation-control">
      <button
        type="button"
        className="simulation-control__button"
        onClick={handleClick}
        disabled={presentation.disabled}
        aria-busy={presentation.busy}
      >
        {presentation.label}
      </button>
      {error ? <p className="simulation-control__error">{error.message}</p> : null}
    </div>
  );
}
