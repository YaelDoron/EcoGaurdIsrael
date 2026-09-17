import { Marker, Popup } from "react-leaflet";
import type L from "leaflet";
import type { ResponseTarget } from "../../types/eventDetails";
import { createCircleIcon } from "./divIcon";

export interface ResponseTargetLayerProps {
  targets: ResponseTarget[];
}

const TARGET_ICON_SIZE = 16;
const ACTIVE_FIRE_COLOR = "var(--color-danger)";
const PREDICTED_RISK_COLOR = "var(--color-warning)";

export const TARGET_TYPE_LABEL: Record<ResponseTarget["target_type"], string> = {
  active_fire: "Active fire",
  predicted_risk: "Predicted risk",
};

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
export function ResponseTargetLayer({ targets }: ResponseTargetLayerProps) {
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
            <p>Priority score: {target.priority_score.toFixed(2)}</p>
            {target.prediction_horizon_minutes !== null ? (
              <p>Prediction horizon: {target.prediction_horizon_minutes} min</p>
            ) : null}
          </Popup>
        </Marker>
      ))}
    </>
  );
}
