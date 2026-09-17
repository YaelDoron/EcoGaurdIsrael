/**
 * Small, domain-agnostic map vocabulary shared by the generic map
 * infrastructure (MapView, LayerControls, FitBoundsToPoints) and every
 * feature-specific layer. Kept separate from any FireEvent-shaped type so
 * this infrastructure can be reused as-is by US 6.3's routing map.
 */
export interface LatLngPoint {
  lat: number;
  lng: number;
}

/** One toggleable entry in a LayerControls panel. */
export interface LayerToggle {
  id: string;
  label: string;
  /** Optional item count shown next to the label (e.g. "Stations (4)"). */
  count?: number;
}

/** Visibility flags keyed by LayerToggle.id. A missing key is treated as visible. */
export type LayerVisibility = Record<string, boolean>;
