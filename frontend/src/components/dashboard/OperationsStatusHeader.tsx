import { TimestampDisplay } from "../data/TimestampDisplay";
import "../status/badges.css";
import { SimulationControl } from "./SimulationControl";
import { describeSimulation, describeSimulationProgress } from "./simulationStatusPresentation";
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
 * Task A8 Part 22-23 + Task A9: the dashboard's simulation-state indicator,
 * "last updated" timestamp, and (A9) the one Start Simulation/Run Again
 * action - extended in place rather than adding a second toolbar, so the
 * map stays the visually dominant element.
 */
export function OperationsStatusHeader({
  simulation,
  generatedAt,
  refreshError,
  onRequestOverviewRefresh,
}: OperationsStatusHeaderProps) {
  const presentation = describeSimulation(simulation);
  const progress = describeSimulationProgress(simulation.run);
  const runFailureMessage =
    simulation.run?.state === "failed" && simulation.run.error !== null ? simulation.run.error.message : null;

  return (
    <div className="operations-status-header">
      <div className="operations-status-header__row">
        <span className={`badge badge--${presentation.tone}`}>{presentation.label}</span>
        <span className="operations-status-header__updated">
          Last updated: <TimestampDisplay value={generatedAt} />
        </span>
        <SimulationControl
          run={simulation.run}
          enabled={simulation.enabled}
          onRequestOverviewRefresh={onRequestOverviewRefresh}
        />
      </div>

      {progress ? <p className="operations-status-header__progress">{progress}</p> : null}
      {runFailureMessage ? <p className="operations-status-header__run-error">{runFailureMessage}</p> : null}
      {refreshError ? <span className="operations-status-header__refresh-warning">{refreshError}</span> : null}
    </div>
  );
}
