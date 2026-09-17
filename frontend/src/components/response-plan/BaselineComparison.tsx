import { MetricCard } from "../data/MetricCard";
import { EmptyState } from "../feedback/EmptyState";
import type { BaselineComparison as BaselineComparisonData } from "../../types/responsePlan";
import {
  formatCoveragePercentage,
  formatDurationSeconds,
  formatImprovementPercentage,
  formatScore,
  formatScoreDifference,
  NOT_AVAILABLE_LABEL,
} from "./formatting";
import "./ResponsePlanSummary.css";

export interface BaselineComparisonSectionProps {
  comparison: BaselineComparisonData | null;
}

const NOT_AVAILABLE_TITLE = "Baseline comparison not available";
const NOT_AVAILABLE_MESSAGE = "No baseline comparison has been computed for this response plan.";

/**
 * Displays the persisted optimized-vs-baseline comparison
 * (`plan.baseline_comparison`) exactly as returned by the backend - never
 * recalculates `score_difference`/`improvement_percentage`, and never
 * derives improvement from `plan_score`/`baseline_score` itself. A `null`
 * comparison is shown as an explicit unavailable state, never hidden.
 */
export function BaselineComparisonSection({ comparison }: BaselineComparisonSectionProps) {
  return (
    <section aria-labelledby="baseline-comparison-heading" className="response-plan-summary__section">
      <h2 id="baseline-comparison-heading" className="response-plan-summary__section-title">
        Baseline Comparison
      </h2>
      {comparison === null ? (
        <EmptyState title={NOT_AVAILABLE_TITLE} message={NOT_AVAILABLE_MESSAGE} />
      ) : (
        <div className="response-plan-summary__metrics-grid">
          <MetricCard label="Baseline Score" value={formatScore(comparison.baseline_score)} />
          <MetricCard
            label="Baseline Coverage"
            value={formatCoveragePercentage(comparison.baseline_coverage_score)}
          />
          <MetricCard
            label="Baseline Average ETA"
            value={
              comparison.baseline_average_eta_seconds !== null
                ? formatDurationSeconds(comparison.baseline_average_eta_seconds)
                : NOT_AVAILABLE_LABEL
            }
          />
          <MetricCard label="Score Difference" value={formatScoreDifference(comparison.score_difference)} />
          <MetricCard
            label="Improvement"
            value={
              comparison.improvement_percentage !== null
                ? formatImprovementPercentage(comparison.improvement_percentage)
                : NOT_AVAILABLE_LABEL
            }
          />
        </div>
      )}
    </section>
  );
}
