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
  /** CSS opacity (0-1) baked into the marker's own div-icon style, so it
   * renders consistently in real Leaflet and in tests (which read the
   * icon's `html` directly rather than a Leaflet-managed DOM attribute).
   * Omitted entirely (not even `opacity:1`) when not given, so every
   * existing caller's rendered output is unchanged. */
  opacity?: number;
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
    options.opacity !== undefined ? `opacity:${options.opacity}` : "",
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
    options.opacity !== undefined ? `opacity:${options.opacity}` : "",
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

// Bootstrap Icons glyphs (MIT, https://icons.getbootstrap.com), inlined as SVG
// so markers render without adding the icon-font dependency/stylesheet.
// `fill="currentColor"` lets each marker's own `color` style drive the glyph.
const BI_GEO_ALT_FILL_PATH =
  "M8 16s6-5.686 6-10A6 6 0 0 0 2 6c0 4.314 6 10 6 10m0-7a3 3 0 1 1 0-6 3 3 0 0 1 0 6";
const BI_FIRE_PATH =
  "M8 16c3.314 0 6-2 6-5.5 0-1.5-.5-4-2.5-6 .25 1.5-1.25 2-1.25 2C11 4 9 .5 6 0c.357 2 .5 4-2 6-1.25 1-2 2.729-2 4.5C2 14 4.686 16 8 16m0-1c-1.657 0-3-1-3-2.75 0-.75.25-2 1.25-3C6.125 10 7 10.5 7 10.5c-.375-1.25.5-3.25 2-3.5-.179 1-.25 2 1 3 .625.5 1 1.364 1 2.25C11 14 9.657 15 8 15";

function glyphIcon(
  kind: "station" | "fire",
  path: string,
  color: string,
  size: number,
  anchorY: number,
  opacity: number | undefined,
  extraStyle: string,
): L.DivIcon {
  const style = [
    "display:block",
    `width:${size}px`,
    `height:${size}px`,
    `color:${color}`,
    extraStyle,
    opacity !== undefined ? `opacity:${opacity}` : "",
  ]
    .filter(Boolean)
    .join(";");

  return L.divIcon({
    className: "map-div-icon",
    html:
      `<span data-${kind}-icon="true" style="${style}">` +
      `<svg viewBox="0 0 16 16" width="100%" height="100%" fill="currentColor" aria-hidden="true">` +
      `<path d="${path}"/></svg></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, anchorY],
  });
}

/**
 * A fire-station marker: Bootstrap Icons' `geo-alt-fill` location pin, tinted
 * with `color` (callers pick it from data they already have, e.g. resource
 * availability). The pin's tip - not its center - sits on the coordinate.
 * `opacity` behaves as in `createCircleIcon`.
 */
export function createStationIcon(color: string, options: CircleIconOptions = {}): L.DivIcon {
  const size = options.size ?? 28;
  return glyphIcon(
    "station",
    BI_GEO_ALT_FILL_PATH,
    color,
    size,
    size,
    options.opacity,
    "filter:drop-shadow(0 1px 1px rgba(0,0,0,0.55))",
  );
}

/**
 * An active-wildfire marker: Bootstrap Icons' `fire` flame in `color`
 * (callers pass a vibrant orange/red), with a soft glow so it stands out on
 * the map. Centered on the coordinate.
 */
export function createFireIcon(color: string, options: CircleIconOptions = {}): L.DivIcon {
  const size = options.size ?? 30;
  return glyphIcon(
    "fire",
    BI_FIRE_PATH,
    color,
    size,
    size / 2,
    options.opacity,
    `filter:drop-shadow(0 0 3px ${color}) drop-shadow(0 1px 1px rgba(0,0,0,0.5))`,
  );
}
