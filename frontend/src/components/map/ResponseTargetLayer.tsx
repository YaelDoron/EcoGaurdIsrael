import { Marker, Popup } from "react-leaflet";
import type L from "leaflet";
import type { ResponseTarget } from "../../types/eventDetails";
import { createCircleIcon } from "./divIcon";

export interface ResponseTargetLayerProps {
  targets: ResponseTarget[];
  /** English place name for the parent FireEvent, shown in every target's popup for context. `null`/omitted when none is known yet. */
  eventLocationName?: string | null;
}

const TARGET_ICON_SIZE = 16;
const ACTIVE_FIRE_COLOR = "var(--color-danger)";
const PREDICTED_RISK_COLOR = "var(--color-warning)";

export const TARGET_TYPE_LABEL: Record<ResponseTarget["target_type"], string> = {
  active_fire: "Active fire",
  predicted_risk: "Predicted risk",
};

// V1 response-target priority methodology (see
// backend/src/calculators/response_target/response_target_config.py):
// an ACTIVE_FIRE target's priority_score is `100 + severity_score`
// (severity_score in [0, 100]), so this offset recovers the severity score
// for operational-term bucketing. A PREDICTED_RISK target's priority_score
// has no such offset (it is `risk_score * horizon_factor`, already 0-100).
const ACTIVE_FIRE_PRIORITY_BASE = 100;

// Same 0/25/50/75 boundaries as the backend's own fire-severity levels (see
// backend/src/calculators/fire_severity/fire_severity_config.py), so an
// ACTIVE_FIRE target's derived term lines up with FireEvent severity wording
// elsewhere in the app.
function operationalLevel(score: number): "Low" | "Moderate" | "High" | "Critical" {
  if (score >= 75) return "Critical";
  if (score >= 50) return "High";
  if (score >= 25) return "Moderate";
  return "Low";
}

/** A human-readable operational term for a target's raw `priority_score` - never the raw number itself. */
function operationalSummary(target: ResponseTarget): string {
  if (target.target_type === "active_fire") {
    return `Severity: ${operationalLevel(target.priority_score - ACTIVE_FIRE_PRIORITY_BASE)}`;
  }
  return `Risk: ${operationalLevel(target.priority_score)}`;
}

function iconFor(targetType: ResponseTarget["target_type"]): L.DivIcon {
  if (targetType === "active_fire") {
    return createCircleIcon(ACTIVE_FIRE_COLOR, { size: TARGET_ICON_SIZE, filled: true });
  }
  // Hollow, dashed ring - visually distinct from the solid ACTIVE_FIRE marker.
  return createCircleIcon(PREDICTED_RISK_COLOR, {
    size: TARGET_ICON_SIZE,
    filled: false,
    borderStyle: "dashed",
  });
}

/** Response targets from the FireEvent's latest persisted target set, if any. */
export function ResponseTargetLayer({ targets, eventLocationName = null }: ResponseTargetLayerProps) {
  if (targets.length === 0) {
    return null;
  }

  return (
    <>
      {targets.map((target) => (
        <Marker
          key={`${target.target_order}-${target.latitude}-${target.longitude}`}
          position={[target.latitude, target.longitude]}
          icon={iconFor(target.target_type)}
        >
          <Popup>
            <strong>{TARGET_TYPE_LABEL[target.target_type] ?? target.target_type}</strong>
            {eventLocationName ? <p>{eventLocationName}</p> : null}
            <p>{operationalSummary(target)}</p>
            {target.prediction_horizon_minutes !== null ? (
              <p>Prediction horizon: {target.prediction_horizon_minutes} min</p>
            ) : null}
          </Popup>
        </Marker>
      ))}
    </>
  );
}
