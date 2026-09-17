import L from "leaflet";

/**
 * Builds small colored DOM markers via `L.divIcon` instead of Leaflet's
 * default image-based marker (whose bundled icon paths break under Vite
 * without extra asset copying). Purely presentational - callers decide the
 * color/shape from data they already have; this module invents nothing.
 */
export interface CircleIconOptions {
  size?: number;
  borderColor?: string;
  borderStyle?: "solid" | "dashed";
  filled?: boolean;
}

const DEFAULT_SIZE = 18;
const DEFAULT_BORDER_COLOR = "#ffffff";

export function createCircleIcon(color: string, options: CircleIconOptions = {}): L.DivIcon {
  const size = options.size ?? DEFAULT_SIZE;
  const borderColor = options.borderColor ?? DEFAULT_BORDER_COLOR;
  const borderStyle = options.borderStyle ?? "solid";
  const filled = options.filled ?? true;
  const borderWidth = borderStyle === "dashed" ? 3 : 2;

  const style = [
    "display:block",
    `width:${size}px`,
    `height:${size}px`,
    "border-radius:50%",
    `background:${filled ? color : "transparent"}`,
    `border:${borderWidth}px ${borderStyle} ${filled ? borderColor : color}`,
    filled ? "box-shadow:0 0 0 1px rgba(0,0,0,0.25)" : "",
  ]
    .filter(Boolean)
    .join(";");

  return L.divIcon({
    className: "map-div-icon",
    html: `<span style="${style}"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

/** A small square marker, visually distinct from the circular markers above. */
export function createSquareIcon(color: string, options: CircleIconOptions = {}): L.DivIcon {
  const size = options.size ?? DEFAULT_SIZE;
  const borderColor = options.borderColor ?? DEFAULT_BORDER_COLOR;

  const style = [
    "display:block",
    `width:${size}px`,
    `height:${size}px`,
    "border-radius:3px",
    `background:${color}`,
    `border:2px solid ${borderColor}`,
    "box-shadow:0 0 0 1px rgba(0,0,0,0.25)",
  ].join(";");

  return L.divIcon({
    className: "map-div-icon",
    html: `<span style="${style}"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}
