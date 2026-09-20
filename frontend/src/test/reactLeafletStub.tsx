import { createContext, useContext, useImperativeHandle, useRef, type ReactNode, type Ref } from "react";
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
export interface FakeLeafletHandler {
  enable: () => void;
  disable: () => void;
}

export interface FakeLeafletMap {
  setView: (center: [number, number], zoom: number) => FakeLeafletMap;
  fitBounds: (bounds: L.LatLngBounds, options?: L.FitBoundsOptions) => FakeLeafletMap;
  getZoom: () => number;
  /** Mirrors Leaflet's own `Map.scrollWheelZoom` Handler (enable/disable pair) - see DisableScrollWheelZoom.tsx. */
  scrollWheelZoom: FakeLeafletHandler;
}

export function createFakeMap(overrides: Partial<FakeLeafletMap> = {}): FakeLeafletMap {
  const fakeMap: FakeLeafletMap = {
    setView: () => fakeMap,
    fitBounds: () => fakeMap,
    getZoom: () => 2,
    scrollWheelZoom: { enable: () => {}, disable: () => {} },
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

export interface MapContainerProps extends ChildrenProps {
  scrollWheelZoom?: boolean;
}

export function MapContainer({ children, scrollWheelZoom }: MapContainerProps & Record<string, unknown>) {
  return (
    <div data-testid="map-container" data-scroll-wheel-zoom={String(scrollWheelZoom)}>
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

/**
 * A true geographic circle (radius in meters) - distinct from CircleMarker's
 * fixed-pixel radius. Added for Task A8's Fire Danger region layer.
 */
export interface CircleProps extends ChildrenProps {
  center: [number, number];
  radius: number;
  pathOptions?: { color?: string; fillColor?: string; fillOpacity?: number; weight?: number };
  eventHandlers?: { click?: () => void };
}

export function Circle({ center, radius, pathOptions, eventHandlers, children }: CircleProps) {
  return (
    <div
      data-testid="circle"
      data-lat={center[0]}
      data-lng={center[1]}
      data-radius={radius}
      data-color={pathOptions?.color}
      data-fill-color={pathOptions?.fillColor}
      data-fill-opacity={pathOptions?.fillOpacity}
      onClick={eventHandlers?.click}
    >
      {children}
    </div>
  );
}

export function Popup({ children }: ChildrenProps) {
  return <div data-testid="popup">{children}</div>;
}

export interface PolylineProps extends ChildrenProps {
  ref?: Ref<{ bringToFront: () => void }>;
  positions: [number, number][];
  pathOptions?: { color?: string; weight?: number; opacity?: number; dashArray?: string };
}

export function Polyline({ positions, pathOptions, children, ref }: PolylineProps) {
  const element = useRef<HTMLDivElement>(null);
  useImperativeHandle(ref, () => ({
    bringToFront: () => element.current?.setAttribute("data-front", "true"),
  }));
  return (
    <div
      ref={element}
      data-testid="polyline"
      data-positions={JSON.stringify(positions)}
      data-color={pathOptions?.color}
      data-weight={pathOptions?.weight}
      data-opacity={pathOptions?.opacity}
      data-dash={pathOptions?.dashArray}
    >
      {children}
    </div>
  );
}
