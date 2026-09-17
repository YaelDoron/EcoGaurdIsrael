/**
 * Presentation-only numeric formatting for the Response Plan summary
 * (Epic 6, US 6.3, Task 8). Every function here only reformats a value the
 * backend already computed and persisted - none of them calculate,
 * estimate, or derive a value from `plan.actions`/`plan.uncovered_targets`.
 */

export const NOT_AVAILABLE_LABEL = "Not available";

/**
 * `plan_score`/`baseline_score`/`score_difference` are already on a
 * bounded 0-100 point scale, not a 0-1 ratio (see
 * backend/src/models/plan_score_breakdown.py: `total_score must be within
 * [0, 100]`) - this only rounds for display.
 */
export function formatScore(value: number): string {
  return value.toFixed(1);
}

/** Same 0-100 point scale as `formatScore`, with an explicit sign so a
 * negative score_difference reads clearly as a regression vs. baseline. */
export function formatScoreDifference(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}`;
}

/**
 * `coverage_score`/`baseline_coverage_score` are already computed on the
 * backend as `100 * covered_priority / total_priority` (see
 * backend/src/calculators/response_optimization/response_plan_scorer.py)
 * - i.e. already a 0-100 percentage, not a 0-1 ratio. This only appends
 * "%"; it never multiplies/divides by 100 itself.
 */
export function formatCoveragePercentage(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * `improvement_percentage` is already computed as a percentage on the
 * backend (see
 * backend/src/calculators/baseline_plan/baseline_plan_comparison_calculator.py:
 * `((optimized_score - baseline_score) / baseline_score) * 100`) - this
 * only formats sign/decimal places, it never recomputes the ratio.
 */
export function formatImprovementPercentage(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

/**
 * `average_eta_seconds`/`baseline_average_eta_seconds`/a route's
 * `eta_seconds` are already the persisted seconds value - this only
 * reformats units for readability (e.g. `150` -> "2m 30s"). It never
 * estimates a duration from routes or actions.
 */
export function formatDurationSeconds(value: number): string {
  const totalSeconds = Math.round(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes === 0 ? `${seconds}s` : `${minutes}m ${seconds}s`;
}

/**
 * A route's `distance_meters` is already the persisted RouteResult
 * distance - this only chooses meters vs. kilometers units for
 * readability (below 1000m shows whole meters, at/above shows kilometers
 * to one decimal place). It never recalculates a route distance.
 */
export function formatDistanceMeters(value: number): string {
  if (value < 1000) {
    return `${Math.round(value)} m`;
  }
  return `${(value / 1000).toFixed(1)} km`;
}
