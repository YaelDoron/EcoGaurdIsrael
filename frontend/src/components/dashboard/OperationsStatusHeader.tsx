import { TimestampDisplay } from "../data/TimestampDisplay";
import { SimulationControl } from "./SimulationControl";
import type { OperationsSimulationSummary } from "../../types/operationsOverview";
import "./OperationsStatusHeader.css";

export interface OperationsStatusHeaderProps {
  simulation: OperationsSimulationSummary;
  /** The server's snapshot timestamp (A6 `generated_at`) - never `Date.now()`. */
  generatedAt: string;
  /** Shown when a background poll fails but a previous good snapshot is still displayed (Part 18). */
  refreshError?: string | null;
  /** Forwarded to `SimulationControl` (Task A9) - nudges one immediate `useOperationsOverview` refresh after a successful start or a 409/403 rejection. */
  onRequestOverviewRefresh: () => void;
}

/**
 * Task A8 Part 22-23 + Task A9: the dashboard's "last updated" timestamp and
 * the one Start Simulation/Run Again action - extended in place rather than
 * adding a second toolbar, so the map stays the visually dominant element.
 *
 * Production polish pass: the "No Simulation Running"/"Running" status
 * badge is deliberately gone - it read as development telemetry on an
 * operational dashboard. `SimulationControl`'s own button already reflects
 * PREPARING/RUNNING/terminal state via its label/disabled-ness, so no
 * separate indicator is needed to know whether a run is in progress.
 */
export function OperationsStatusHeader({
  simulation,
  generatedAt,
  refreshError,
  onRequestOverviewRefresh,
}: OperationsStatusHeaderProps) {
  const runFailureMessage =
    simulation.run?.state === "failed" && simulation.run.error !== null ? simulation.run.error.message : null;

  return (
    <div className="operations-status-header">
      <div className="operations-status-header__row">
        <span className="operations-status-header__updated">
          Last updated: <TimestampDisplay value={generatedAt} />
        </span>
        <SimulationControl
          run={simulation.run}
          enabled={simulation.enabled}
          onRequestOverviewRefresh={onRequestOverviewRefresh}
        />
      </div>

      {runFailureMessage ? <p className="operations-status-header__run-error">{runFailureMessage}</p> : null}
      {refreshError ? <span className="operations-status-header__refresh-warning">{refreshError}</span> : null}
    </div>
  );
}
