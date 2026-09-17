import type { ResponsePlan } from "../../types/responsePlan";
import "./ResponsePlanSummary.css";
import "./OptimizationDetails.css";

export interface OptimizationDetailsProps {
  plan: Pick<ResponsePlan, "methodology" | "methodology_version" | "random_seed" | "optimization_config">;
}

const NO_CONFIG_MESSAGE = "Detailed optimization configuration is not available for this response plan.";

/**
 * Persisted provenance only, collapsed by default (native `<details>`, no
 * new UI dependency). `methodology`/`methodology_version`/`random_seed`
 * always come straight from the plan; the GA configuration fields render
 * only when the backend actually persisted one (`plan.optimization_config`)
 * - for a legacy plan predating full GA-config capture this is `null` and
 * no default/fabricated GA values are substituted.
 */
export function OptimizationDetails({ plan }: OptimizationDetailsProps) {
  const config = plan.optimization_config;

  return (
    <section aria-labelledby="optimization-details-heading" className="response-plan-summary__section">
      <h2 id="optimization-details-heading" className="response-plan-summary__section-title">
        Optimization Details
      </h2>
      <details className="optimization-details">
        <summary className="optimization-details__summary">Technical details</summary>

        <dl className="optimization-details__list">
          <div className="optimization-details__row">
            <dt>Methodology</dt>
            <dd>{plan.methodology}</dd>
          </div>
          <div className="optimization-details__row">
            <dt>Methodology version</dt>
            <dd>{plan.methodology_version}</dd>
          </div>
          <div className="optimization-details__row">
            <dt>Random seed</dt>
            <dd>{plan.random_seed}</dd>
          </div>
        </dl>

        {config === null ? (
          <p className="optimization-details__unavailable">{NO_CONFIG_MESSAGE}</p>
        ) : (
          <dl className="optimization-details__list">
            <div className="optimization-details__row">
              <dt>Population size</dt>
              <dd>{config.population_size}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Generation count</dt>
              <dd>{config.generation_count}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Mutation rate</dt>
              <dd>{config.mutation_rate}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Crossover rate</dt>
              <dd>{config.crossover_rate}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>ETA reference (seconds)</dt>
              <dd>{config.eta_reference_seconds}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Initial assignment probability</dt>
              <dd>{config.initial_assignment_probability}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Tournament size</dt>
              <dd>{config.tournament_size}</dd>
            </div>
            <div className="optimization-details__row">
              <dt>Elitism count</dt>
              <dd>{config.elitism_count}</dd>
            </div>
          </dl>
        )}
      </details>
    </section>
  );
}
