import type { GlobalPlanningRunStatus } from "../../types/operationsActivity";

/**
 * Task A9, Part X: how long a newly-arrived Activity Feed item keeps its
 * "New" presentation after first being seen in an overview snapshot. This
 * is presentation-only timing, not persisted anywhere - see
 * `OperationsActivityFeed`'s snapshot-to-snapshot activity_id diff.
 */
export const NEW_ACTIVITY_HIGHLIGHT_MS = 8000;

export const GLOBAL_PLANNING_RUN_STATUS_LABEL: Record<GlobalPlanningRunStatus, string> = {
  running: "Running",
  completed: "Completed",
  partial: "Completed with issues",
  failed: "Failed",
  no_active_events: "No active events",
};

/**
 * NASA FIRMS' own well-known satellite-hotspot confidence categories
 * ("l"/"n"/"h" - low/nominal/high, see backend's
 * simulation/generators/satellite_data_generator.py) - a real persisted
 * categorical value, not a frontend-invented threshold. This is the one
 * canonical label lookup for it; an unrecognized code falls back to the
 * raw value rather than hiding it.
 */
export const SATELLITE_CONFIDENCE_LABEL: Record<string, string> = {
  l: "Low",
  n: "Nominal",
  h: "High",
};
