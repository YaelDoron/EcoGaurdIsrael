import "./DataDisplay.css";

export interface CoordinateDisplayProps {
  latitude: number;
  longitude: number;
  /** Decimal places shown for both values. Defaults to 4 (~11m precision). */
  precision?: number;
}

const DEFAULT_PRECISION = 4;

/**
 * Presentation formatting only: fixed decimal precision, comma-separated
 * "lat, lon" order. Never reverse-geocodes, infers an area name, or
 * mutates the coordinates it is given.
 */
export function CoordinateDisplay({ latitude, longitude, precision = DEFAULT_PRECISION }: CoordinateDisplayProps) {
  return (
    <span className="coordinate-display">
      {latitude.toFixed(precision)}, {longitude.toFixed(precision)}
    </span>
  );
}
