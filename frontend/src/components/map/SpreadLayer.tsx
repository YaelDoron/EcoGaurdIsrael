import { CircleMarker, Popup } from "react-leaflet";
import type { SpreadPrediction } from "../../types/eventDetails";
import { spreadRiskColor } from "./colors";

export interface SpreadLayerProps {
  predictions: SpreadPrediction[];
}

const CELL_RADIUS_PX = 7;
const MIN_FILL_OPACITY = 0.35;
const PROBABILITY_OPACITY_RANGE = 0.4;

/**
 * Renders every predicted spread cell across all supplied horizons. A
 * horizon whose latest prediction is not "valid" (insufficient_data /
 * inactive_event) always carries `cells: []` from the backend - this layer
 * renders nothing at all for that horizon rather than showing a stale or
 * fabricated spread, and the FireEventMarker above stays visible on its own.
 */
export function SpreadLayer({ predictions }: SpreadLayerProps) {
  const cells = predictions.flatMap((prediction) =>
    prediction.cells.map((cell) => ({ ...cell, horizonMinutes: prediction.horizon_minutes })),
  );

  if (cells.length === 0) {
    return null;
  }

  return (
    <>
      {cells.map((cell, index) => {
        const color = spreadRiskColor(cell.spread_risk_score);
        return (
          <CircleMarker
            key={`${cell.horizonMinutes}-${cell.latitude}-${cell.longitude}-${index}`}
            center={[cell.latitude, cell.longitude]}
            radius={CELL_RADIUS_PX}
            pathOptions={{
              color,
              fillColor: color,
              fillOpacity: MIN_FILL_OPACITY + cell.spread_probability * PROBABILITY_OPACITY_RANGE,
              weight: 1,
            }}
          >
            <Popup>
              <strong>Predicted spread ({cell.horizonMinutes} min horizon)</strong>
              <p>Risk score: {cell.spread_risk_score.toFixed(1)}</p>
              <p>Probability: {Math.round(cell.spread_probability * 100)}%</p>
              <p>Reached at: {cell.reached_minutes} min</p>
            </Popup>
          </CircleMarker>
        );
      })}
    </>
  );
}
