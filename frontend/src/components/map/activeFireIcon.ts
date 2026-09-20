import L from "leaflet";

/**
 * Builds the Operations dashboard's active-fire marker icon (Task A8) - a
 * small, isolated builder independent of divIcon.ts's createCircleIcon/
 * createSquareIcon (used by FireEventMarker/StationLayer/etc, left
 * untouched here to avoid any risk to Event Details' own map). Purely
 * presentational: the caller decides color/emphasis from persisted data it
 * already has (FireEvent status + latest_severity.level) - this module
 * invents no business meaning of its own.
 */
export interface ActiveFireIconOptions {
  size?: number;
  /**
   * True for CONFIRMED + HIGH/CRITICAL severity - renders a larger dot with
   * a subtle pulsing halo (see activeFireIcon.css). This is presentation
   * only: it changes how an already-known fact is displayed, never a new
   * business score/rank.
   */
  emphasize?: boolean;
  /**
   * True for CONFIRMED, false for SUSPECTED - a real persisted FireEvent
   * status, never inferred from severity. Controls solid-fill (CONFIRMED)
   * vs hollow/lighter-center (SUSPECTED) presentation, so the two remain
   * visually distinguishable even before reading any text/popup - color
   * alone (amber vs red) was not enough of a distinction on its own.
   */
  confirmed: boolean;
}

const DEFAULT_SIZE = 16;
const EMPHASIZED_SIZE = 22;
const BORDER_COLOR = "#ffffff";
const HALO_SCALE = 2;
const HOLLOW_CENTER_COLOR = "#ffffff";
const HOLLOW_RING_WIDTH = 3;

export function createActiveFireIcon(color: string, options: ActiveFireIconOptions): L.DivIcon {
  const emphasize = options.emphasize ?? false;
  const confirmed = options.confirmed;
  const size = options.size ?? (emphasize ? EMPHASIZED_SIZE : DEFAULT_SIZE);
  const haloSize = size * HALO_SCALE;
  const haloOffset = (haloSize - size) / 2;

  // CONFIRMED: a solid filled dot. SUSPECTED: a hollow ring with a lighter
  // center - the same shape family, but unmistakably different up close.
  const dotStyle = confirmed
    ? [
        "display:block",
        `width:${size}px`,
        `height:${size}px`,
        "border-radius:50%",
        `background:${color}`,
        `border:2px solid ${BORDER_COLOR}`,
        "box-shadow:0 0 0 1px rgba(0,0,0,0.3)",
        "position:relative",
        "z-index:1",
      ].join(";")
    : [
        "display:block",
        `width:${size}px`,
        `height:${size}px`,
        "border-radius:50%",
        `background:${HOLLOW_CENTER_COLOR}`,
        `border:${HOLLOW_RING_WIDTH}px solid ${color}`,
        "box-shadow:0 0 0 1px rgba(0,0,0,0.25)",
        "position:relative",
        "z-index:1",
      ].join(";");

  const haloStyle = [
    "position:absolute",
    `top:${-haloOffset}px`,
    `left:${-haloOffset}px`,
    `width:${haloSize}px`,
    `height:${haloSize}px`,
    "border-radius:50%",
    `background:${color}`,
  ].join(";");

  const haloHtml = emphasize
    ? `<span class="active-fire-icon__halo" style="${haloStyle}"></span>`
    : "";

  const classNames = ["map-div-icon", "active-fire-icon", confirmed ? "active-fire-icon--confirmed" : "active-fire-icon--suspected"];
  if (emphasize) {
    classNames.push("active-fire-icon--emphasized");
  }

  return L.divIcon({
    className: classNames.join(" "),
    html: `<span style="position:relative;display:block;width:${size}px;height:${size}px;">${haloHtml}<span style="${dotStyle}"></span></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}
