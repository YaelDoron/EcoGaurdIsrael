import { CircleMarker, Popup } from "react-leaflet";
import { isSpreadingCell, type SpreadPrediction } from "../../types/eventDetails";
import { spreadRiskColor } from "./colors";

export interface SpreadLayerProps {
  predictions: SpreadPrediction[];
}

const CELL_RADIUS_PX = 7;
const MIN_FILL_OPACITY = 0.35;
const PROBABILITY_OPACITY_RANGE = 0.4;
// Risk-only cells (below the propagation threshold): dashed outline and a
// slightly lower fill so they read as predicted risk, not predicted spread.
const RISK_ONLY_DASH_ARRAY = "3 3";
const RISK_ONLY_FILL_OPACITY_FACTOR = 0.6;

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
        const spreading = isSpreadingCell(cell);
        const fillOpacity = MIN_FILL_OPACITY + cell.spread_probability * PROBABILITY_OPACITY_RANGE;
        return (
          <CircleMarker
            key={`${cell.horizonMinutes}-${cell.latitude}-${cell.longitude}-${index}`}
            center={[cell.latitude, cell.longitude]}
            radius={CELL_RADIUS_PX}
            pathOptions={
              spreading
                ? { color, fillColor: color, fillOpacity, weight: 1 }
                : {
                    color,
                    fillColor: color,
                    fillOpacity: fillOpacity * RISK_ONLY_FILL_OPACITY_FACTOR,
                    weight: 1,
                    dashArray: RISK_ONLY_DASH_ARRAY,
                  }
            }
          >
            <Popup>
              <strong>
                {spreading ? "Predicted spread" : "Predicted spread risk"} ({cell.horizonMinutes} min horizon)
              </strong>
              <p>{spreading ? "Reached the propagation threshold" : "Did not reach the propagation threshold"}</p>
              <p>Risk score: {cell.spread_risk_score.toFixed(1)}</p>
              <p>Probability: {Math.round(cell.spread_probability * 100)}%</p>
              <p>
                {spreading ? "Model reach time" : "Risk assessed at"}: {cell.reached_minutes} min
              </p>
            </Popup>
          </CircleMarker>
        );
      })}
    </>
  );
}
