import { MetricCard } from "../data/MetricCard";
import { formatCoveragePercentage, formatDurationSeconds, NOT_AVAILABLE_LABEL } from "../response-plan/formatting";
import type { GlobalPlanMetrics, GlobalPlanShortage } from "../../types/globalResponsePlan";
import "./GlobalResponsePanels.css";

export interface GlobalMetricsPanelProps {
  metrics: GlobalPlanMetrics;
  shortage: GlobalPlanShortage;
}

/**
 * Displays the persisted generation-level metrics (`plan.metrics`) and
 * aggregate resource shortage (`plan.shortage`) exactly as returned by the
 * backend, reusing the same `MetricCard` tile and number-formatting helpers
 * (`../response-plan/formatting`) the single-event Response Plan page
 * already uses. Formats values for readability only (decimal places,
 * percentage sign, duration units) - never recomputes coverage, ETA, or any shortage total, and never derives one from
 * `plan.events`.
 */
export function GlobalMetricsPanel({ metrics, shortage }: GlobalMetricsPanelProps) {
  const { total_assigned: assigned, total_desired: desired } = shortage;
  // Display ratio only, capped at 100%: how much of the desired fleet is assigned.
  const fill = desired > 0 ? Math.min(100, Math.round((assigned / desired) * 100)) : assigned > 0 ? 100 : 0;

  return (
    <section aria-labelledby="global-metrics-heading" className="global-response-panel">
      <h2 id="global-metrics-heading" className="global-response-panel__title">
        Global Plan Metrics
      </h2>
      <div className="global-response-panel__grid global-response-panel__grid--wide">
        <MetricCard
          label="Coverage"
          value={metrics.coverage_score !== null ? formatCoveragePercentage(metrics.coverage_score) : NOT_AVAILABLE_LABEL}
        />
        <MetricCard
          label="Average ETA"
          value={
            metrics.average_eta_seconds !== null
              ? formatDurationSeconds(metrics.average_eta_seconds)
              : NOT_AVAILABLE_LABEL
          }
        />
      </div>

      <div className="resource-summary" aria-label="Resource shortage">
        <p className="resource-summary__line">
          <span className="resource-summary__label">Resources</span>
          <strong>Assigned: {assigned}</strong> / Desired: {desired}
          <span aria-hidden="true" className="resource-summary__sep">
            |
          </span>
          <span className={shortage.unmet_desired > 0 ? "resource-summary__unmet" : undefined}>
            Unmet: {shortage.unmet_desired}
          </span>
        </p>
        <div
          className="resource-summary__bar"
          role="progressbar"
          aria-label="Desired resources assigned"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={fill}
        >
          <div className="resource-summary__fill" style={{ width: `${fill}%` }} />
        </div>
        <p className="resource-summary__detail">
          Required: {shortage.total_required} | Unmet: {shortage.unmet_required}
        </p>
      </div>
    </section>
  );
}
