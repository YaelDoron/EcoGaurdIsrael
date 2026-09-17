import type { FireSeverityAssessmentStatus } from "../../types/fireEvent";

/**
 * Small caption shown alongside SeverityBadge when the latest persisted
 * severity assessment exists but is not "valid". SeverityBadge already
 * renders "Not available" for the (always-null, per backend contract)
 * level in this case; this caption adds honest extra context using the
 * real backend assessment status - it never fabricates a severity level.
 */
export const SEVERITY_STATUS_CAPTION: Record<Exclude<FireSeverityAssessmentStatus, "valid">, string> = {
  insufficient_data: "Insufficient data for a severity assessment.",
  inactive_event: "Event was not active at the time of the last assessment.",
};
