import type { ResponsePlanAction } from "../../types/responsePlan";
import { translateStationLabel } from "../map/stationTranslations";
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
   * the user selects it. Omit to render the card without a select control. */
  onSelect?: (actionKey: string) => void;
}

/**
 * One persisted response assignment as a horizontal A-to-B journey:
 * `[Station: Nesher (TRUCK-82-2)] ➔ [Active fire]  8.5 km • ETA: 9m 1s`.
 * Presentation formatting only (labels, duration/distance units) - never
 * recalculates ETA or distance, never reinterprets a missing route as
 * reachable, and never invents a station name the backend did not provide.
 * Raw algorithm values (target id, priority score) are deliberately not shown.
 */
export function ResponseActionCard({ action, isSelected = false, onSelect }: ResponseActionCardProps) {
  const { resource, target, route } = action;
  const cardClassName = isSelected
    ? "response-action-card response-action-card--selected"
    : "response-action-card";
  const targetLabel = target.target_type !== null ? TARGET_TYPE_LABELS[target.target_type] : NOT_AVAILABLE_LABEL;

  return (
    <article className={cardClassName}>
      <div className="response-action-card__journey">
        <div className="response-action-card__origin">
          <span className="response-action-card__station">
            <span className="response-action-card__label">Station: </span>
            {translateStationLabel(resource.station_name ?? resource.station_id)}
          </span>
          <span className="response-action-card__truck">({resource.resource_id})</span>
        </div>

        <span className="response-action-card__arrow" aria-hidden="true">
          ➔
        </span>

        <div className="response-action-card__destination">
          <span
            className={`response-action-card__target response-action-card__target--${target.target_type ?? "unknown"}`}
          >
            {targetLabel}
          </span>
          <span className="response-action-card__stats">
            {route.distance_meters !== null ? formatDistanceMeters(route.distance_meters) : NOT_AVAILABLE_LABEL}
            <span aria-hidden="true"> • </span>
            ETA:{" "}
            <strong className="response-action-card__eta">
              {route.eta_seconds !== null ? formatDurationSeconds(route.eta_seconds) : NOT_AVAILABLE_LABEL}
            </strong>
          </span>
        </div>
      </div>

      {route.status !== "reachable" ? (
        <span className="response-action-card__route-status">
          Route: {route.status !== null ? ROUTE_STATUS_LABELS[route.status] : NOT_AVAILABLE_LABEL}
        </span>
      ) : null}

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
