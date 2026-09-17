import type { ResponsePlanAction } from "../../types/responsePlan";
import { formatDistanceMeters, formatDurationSeconds, NOT_AVAILABLE_LABEL } from "./formatting";
import { ROUTE_STATUS_LABELS, TARGET_TYPE_LABELS } from "./presentation";
import { getResponseActionKey } from "./ResponseRouteLayerModel";
import "./ResponseActionCard.css";

export interface ResponseActionCardProps {
  action: ResponsePlanAction;
  /** Whether this action is the currently selected/highlighted one. Purely
   * a UI concern - never affects the persisted assignment. */
  isSelected?: boolean;
  /** Called with this action's stable key (see `getResponseActionKey`) when
   * the user selects it. Omit to render the card without a select control
   * (e.g. existing Task 9 usages/tests). */
  onSelect?: (actionKey: string) => void;
}

/**
 * One persisted response assignment (resource/station -> target, with its
 * route), exactly as returned by the backend. Presentation formatting only
 * (labels, duration/distance units) - never recalculates ETA, distance, or
 * target priority, never reinterprets a missing route as reachable, and
 * never invents a station name or vehicle coordinate the backend did not
 * provide. Selecting a card (when `onSelect` is provided) only sets UI
 * highlight state - it never reorders actions or changes backend data.
 */
export function ResponseActionCard({ action, isSelected = false, onSelect }: ResponseActionCardProps) {
  const { resource, target, route } = action;
  const cardClassName = isSelected
    ? "response-action-card response-action-card--selected"
    : "response-action-card";

  return (
    <article className={cardClassName}>
      <div className="response-action-card__flow">
        <div className="response-action-card__party">
          <h3 className="response-action-card__party-title">Resource</h3>
          <dl className="response-action-card__details">
            <div className="response-action-card__row">
              <dt>Resource</dt>
              <dd>{resource.resource_id}</dd>
            </div>
            <div className="response-action-card__row">
              <dt>Station</dt>
              <dd>
                {resource.station_name ?? NOT_AVAILABLE_LABEL}{" "}
                <span className="response-action-card__muted">({resource.station_id})</span>
              </dd>
            </div>
          </dl>
        </div>

        <span className="response-action-card__arrow" aria-hidden="true">
          →
        </span>

        <div className="response-action-card__party">
          <h3 className="response-action-card__party-title">Target</h3>
          <dl className="response-action-card__details">
            <div className="response-action-card__row">
              <dt>Target</dt>
              <dd>#{target.response_target_id}</dd>
            </div>
            <div className="response-action-card__row">
              <dt>Type</dt>
              <dd>{target.target_type !== null ? TARGET_TYPE_LABELS[target.target_type] : NOT_AVAILABLE_LABEL}</dd>
            </div>
            <div className="response-action-card__row">
              <dt>Priority</dt>
              <dd>{target.priority_score !== null ? target.priority_score : NOT_AVAILABLE_LABEL}</dd>
            </div>
          </dl>
        </div>
      </div>

      <dl className="response-action-card__route">
        <div className="response-action-card__row">
          <dt>Route status</dt>
          <dd>{route.status !== null ? ROUTE_STATUS_LABELS[route.status] : NOT_AVAILABLE_LABEL}</dd>
        </div>
        <div className="response-action-card__row">
          <dt>ETA</dt>
          <dd>{route.eta_seconds !== null ? formatDurationSeconds(route.eta_seconds) : NOT_AVAILABLE_LABEL}</dd>
        </div>
        <div className="response-action-card__row">
          <dt>Distance</dt>
          <dd>
            {route.distance_meters !== null ? formatDistanceMeters(route.distance_meters) : NOT_AVAILABLE_LABEL}
          </dd>
        </div>
      </dl>

      {onSelect ? (
        <button
          type="button"
          className="response-action-card__select"
          aria-pressed={isSelected}
          onClick={() => onSelect(getResponseActionKey(action))}
        >
          {isSelected ? "Selected" : "Highlight on map"}
        </button>
      ) : null}
    </article>
  );
}
