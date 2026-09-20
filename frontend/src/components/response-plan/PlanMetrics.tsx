import { MetricCard } from "../data/MetricCard";
import type { ResponsePlanMetrics } from "../../types/responsePlan";
import { formatDurationSeconds, NOT_AVAILABLE_LABEL } from "./formatting";
import "./ResponsePlanSummary.css";

export interface PlanMetricsProps {
  metrics: ResponsePlanMetrics;
}

/**
 * Displays the persisted plan-level metrics (`plan.metrics`) exactly as
 * returned by the backend, using the shared `MetricCard` tile. Formats
 * values for readability only (decimal places, percentage sign, duration
 * units) - never recomputes plan score, coverage, or ETA, and never
 * derives any of them from `plan.actions`.
 */
export function PlanMetrics({ metrics }: PlanMetricsProps) {
  return (
    <section aria-label="Plan metrics" className="response-plan-summary__section">
      <div className="response-plan-summary__metrics-grid">
        <MetricCard
          label="Average Travel Time"
          value={
            metrics.average_eta_seconds !== null
              ? formatDurationSeconds(metrics.average_eta_seconds)
              : NOT_AVAILABLE_LABEL
          }
        />
      </div>
    </section>
  );
}
