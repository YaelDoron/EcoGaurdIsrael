import type { ReactNode } from "react";
import { MapContainer, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { FitBoundsToPoints } from "./FitBoundsToPoints";
import type { LatLngPoint } from "./mapTypes";
import "./MapView.css";

export interface MapViewProps {
  /** Points the map's viewport auto-fits to on load and whenever they change. */
  boundsPoints: LatLngPoint[];
  children?: ReactNode;
  ariaLabel?: string;
  /**
   * Whether the mouse wheel zooms the map. Defaults to `true` - the existing
   * behavior for Event Details/Response Plan, both unchanged by this prop's
   * addition. The Operations Overview map passes `false` explicitly (see
   * OperationsMap.tsx) since it sits inside a normally-scrolling page and a
   * wheel-zooming map under the cursor makes the page itself impossible to
   * scroll past. Click-drag panning and the +/- zoom controls are unaffected
   * either way.
   */
  scrollWheelZoom?: boolean;
}

const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
// Only used as MapContainer's required initial `center`/`zoom` before
// FitBoundsToPoints runs its effect - never rendered as a marker or shown
// to the user as a location, and immediately superseded once boundsPoints
// is non-empty (which it always is once real event data has loaded).
const PLACEHOLDER_CENTER: [number, number] = [0, 0];
const PLACEHOLDER_ZOOM = 2;

/**
 * Generic, reusable interactive map container: a base tile layer plus
 * automatic viewport fitting to whatever `boundsPoints` it is given. Holds
 * no FireEvent/wildfire-specific knowledge - feature layers are passed in
 * as `children` (see FireEventMarker, SpreadLayer, etc.) so this same
 * container can be reused by US 6.3's routing map.
 */
export function MapView({
  boundsPoints,
  children,
  ariaLabel = "Interactive map",
  scrollWheelZoom = true,
}: MapViewProps) {
  return (
    <div className="map-view" role="region" aria-label={ariaLabel}>
      <MapContainer
        center={PLACEHOLDER_CENTER}
        zoom={PLACEHOLDER_ZOOM}
        className="map-view__container"
        scrollWheelZoom={scrollWheelZoom}
      >
        <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
        <FitBoundsToPoints points={boundsPoints} />
        {children}
      </MapContainer>
    </div>
  );
}
