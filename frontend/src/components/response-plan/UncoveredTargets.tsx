import type { ResponsePlanTarget } from "../../types/responsePlan";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import { EmptyState } from "../feedback/EmptyState";
import { formatPriority, NOT_AVAILABLE_LABEL } from "./formatting";
import { TARGET_TYPE_LABELS } from "./presentation";
import "./ResponsePlanSummary.css";
import "./UncoveredTargets.css";

export interface UncoveredTargetsProps {
  targets: ResponsePlanTarget[];
}

const ALL_COVERED_TITLE = "All response targets are covered by this plan.";

/**
 * Renders every persisted uncovered target (`plan.uncovered_targets`)
 * exactly as returned by the backend, in backend order. The backend alone
 * decides which targets are uncovered - this component never derives one
 * from `plan.actions`, target priority, or any other frontend calculation,
 * and never drops a target just because some of its fields are null.
 */
export function UncoveredTargets({ targets }: UncoveredTargetsProps) {
  return (
    <section aria-labelledby="uncovered-targets-heading" className="response-plan-summary__section">
      <h2 id="uncovered-targets-heading" className="response-plan-summary__section-title">
        Uncovered Targets
      </h2>
      {targets.length === 0 ? (
        <EmptyState title={ALL_COVERED_TITLE} />
      ) : (
        <ul className="uncovered-target-list">
          {targets.map((target) => (
            <li key={target.response_target_id} className="uncovered-target-list__item">
              <dl className="uncovered-target-list__details">
                <div className="uncovered-target-list__row">
                  <dt>Target</dt>
                  <dd>#{target.response_target_id}</dd>
                </div>
                <div className="uncovered-target-list__row">
                  <dt>Type</dt>
                  <dd>
                    {target.target_type !== null ? TARGET_TYPE_LABELS[target.target_type] : NOT_AVAILABLE_LABEL}
                  </dd>
                </div>
                <div className="uncovered-target-list__row">
                  <dt>Priority</dt>
                  <dd>{target.priority_score !== null ? formatPriority(target.priority_score) : NOT_AVAILABLE_LABEL}</dd>
                </div>
                <div className="uncovered-target-list__row">
                  <dt>Coordinates</dt>
                  <dd>
                    {target.latitude !== null && target.longitude !== null ? (
                      <CoordinateDisplay latitude={target.latitude} longitude={target.longitude} />
                    ) : (
                      NOT_AVAILABLE_LABEL
                    )}
                  </dd>
                </div>
              </dl>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
