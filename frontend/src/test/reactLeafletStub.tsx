import { createContext, useContext, type ReactNode } from "react";
import type L from "leaflet";

/**
 * A minimal `react-leaflet` stand-in used only by tests (via
 * `vi.mock("react-leaflet", ...)`). Real Leaflet DOM/canvas rendering is
 * notoriously unreliable under jsdom (no real layout, no ResizeObserver
 * guarantees); this stub renders plain, queryable DOM instead, so tests
 * exercise this project's own component logic (which marker got which
 * icon/color, which cells got rendered, bounds-fitting calls) without
 * depending on Leaflet's internal rendering pipeline.
 */
export interface FakeLeafletMap {
  setView: (center: [number, number], zoom: number) => FakeLeafletMap;
  fitBounds: (bounds: L.LatLngBounds, options?: L.FitBoundsOptions) => FakeLeafletMap;
  getZoom: () => number;
}

export function createFakeMap(overrides: Partial<FakeLeafletMap> = {}): FakeLeafletMap {
  const fakeMap: FakeLeafletMap = {
    setView: () => fakeMap,
    fitBounds: () => fakeMap,
    getZoom: () => 2,
    ...overrides,
  };
  return fakeMap;
}

export const MapInstanceContext = createContext<FakeLeafletMap>(createFakeMap());

export function useMap(): FakeLeafletMap {
  return useContext(MapInstanceContext);
}

interface ChildrenProps {
  children?: ReactNode;
}

export function MapContainer({ children }: ChildrenProps & Record<string, unknown>) {
  return (
    <div data-testid="map-container">
      <MapInstanceContext.Provider value={createFakeMap()}>{children}</MapInstanceContext.Provider>
    </div>
  );
}

export function TileLayer() {
  return <div data-testid="tile-layer" />;
}

function iconHtml(icon: unknown): string | undefined {
  if (icon && typeof icon === "object" && "options" in icon) {
    const options = (icon as L.DivIcon).options as { html?: string };
    return options.html;
  }
  return undefined;
}

export interface MarkerProps extends ChildrenProps {
  position: [number, number];
  icon?: unknown;
}

export function Marker({ position, icon, children }: MarkerProps) {
  const html = iconHtml(icon);
  return (
    <div data-testid="marker" data-lat={position[0]} data-lng={position[1]}>
      {html ? <span data-testid="marker-icon" dangerouslySetInnerHTML={{ __html: html }} /> : null}
      {children}
    </div>
  );
}

export interface CircleMarkerProps extends ChildrenProps {
  center: [number, number];
  radius?: number;
  pathOptions?: { color?: string; fillColor?: string; fillOpacity?: number; weight?: number };
}

export function CircleMarker({ center, pathOptions, children }: CircleMarkerProps) {
  return (
    <div
      data-testid="circle-marker"
      data-lat={center[0]}
      data-lng={center[1]}
      data-fill-color={pathOptions?.fillColor}
      data-fill-opacity={pathOptions?.fillOpacity}
    >
      {children}
    </div>
  );
}

export function Popup({ children }: ChildrenProps) {
  return <div data-testid="popup">{children}</div>;
}

export interface PolylineProps extends ChildrenProps {
  positions: [number, number][];
  pathOptions?: { color?: string; weight?: number };
}

export function Polyline({ positions, pathOptions, children }: PolylineProps) {
  return (
    <div
      data-testid="polyline"
      data-positions={JSON.stringify(positions)}
      data-color={pathOptions?.color}
      data-weight={pathOptions?.weight}
    >
      {children}
    </div>
  );
}
