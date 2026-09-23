import { useEffect, useRef } from "react";
import type L from "leaflet";
import { Marker, Polyline, Popup } from "react-leaflet";
import { STATION_ALLOCATED_COLOR } from "../map/colors";
import { translateStationLabel } from "../map/stationTranslations";
import { createCircleIcon, createFireIcon, createStationIcon } from "../map/divIcon";
import type { GlobalResponseMapLayerData } from "./GlobalResponseMapLayerModel";

export interface GlobalResponseMapLayerProps {
  layer: GlobalResponseMapLayerData;
  /** Display name per fire event id (English location name); falls back to "Event #id". */
  eventLabels?: Record<number, string>;
}

// With no focus set, each fire event's routes get their own hue so multiple
// events read as distinct groups instead of one uniform blue tangle.
const EVENT_ROUTE_PALETTE = ["#1d4ed8", "#0f766e", "#7c3aed", "#c2410c", "#be185d", "#4d7c0f"];
const ROUTE_WEIGHT_DEFAULT = 4;
const ROUTE_COLOR_FOCUSED = "var(--color-focus-ring)";
const ROUTE_COLOR_DIMMED = "var(--color-text-muted)";
const ROUTE_WEIGHT_FOCUSED = 6;
const ROUTE_WEIGHT_DIMMED = 3;
const ROUTE_OPACITY_FOCUSED = 1;

const ORIGIN_MARKER_COLOR = "var(--color-info)";
const FIRE_MARKER_COLOR = "#ff2200";
const PREDICTED_RISK_COLOR = "var(--color-warning)";

// "Last mile" connector: bridges the routed path's snapped-to-road end point
// to the target's own (off-road) coordinate - always dashed and thin so it
// is never mistaken for part of the actual computed route.
const LAST_MILE_GAP_DASH = "5, 10";
const LAST_MILE_GAP_WEIGHT = 2;

const MARKER_SIZE = 16;
const MARKER_SIZE_FOCUSED = 22;
const STATION_SIZE = 24;
const STATION_SIZE_FOCUSED = 30;
const FIRE_MARKER_SIZE = 28;
const FIRE_MARKER_SIZE_FOCUSED = 38;

const DIMMED_OPACITY = 0.25;

type PolylineProps = Parameters<typeof Polyline>[0];

/** Brings the focused event's route to the front so dimmed routes never paint over it. */
function FocusPolyline({ isFocused, ...props }: PolylineProps & { isFocused: boolean }) {
  const ref = useRef<L.Polyline | null>(null);
  useEffect(() => {
    if (isFocused) {
      ref.current?.bringToFront();
    }
  }, [isFocused]);
  return <Polyline ref={ref} {...props} />;
}

/**
 * Renders `GlobalResponseMapLayerModel.buildGlobalResponseMapLayer`'s
 * precomputed data onto the shared `MapView` surface (Task B-FE-7), using
 * the same react-leaflet primitives and `createCircleIcon`/`createSquareIcon`
 * helpers every other map layer in this app already uses - including the
 * exact Polyline-drawing approach `ResponsePlanMapLayer` uses for one
 * child plan's routes, applied here across every materialized event's
 * actions at once. `layer` has already excluded anything that isn't a
 * persisted, reachable, >=2-point path (see `isDrawableRoute`) and anything
 * missing its own coordinate - this component draws exactly what it is
 * given, and nothing more (no coordinate derivation, no route calculation,
 * no network call).
 *
 * Focus styling (dim everything except the focused event) is opacity-only
 * and baked into each marker's own div-icon (`createCircleIcon`/
 * `createSquareIcon`'s `opacity` option) rather than a Leaflet-level marker
 * `opacity` prop, so it renders identically under real Leaflet and under
 * this project's test stub (which reads the icon's own `html`).
 */
export function GlobalResponseMapLayer({ layer, eventLabels = {} }: GlobalResponseMapLayerProps) {
  const eventLabel = (fireEventId: number) => eventLabels[fireEventId] ?? `Event #${fireEventId}`;
  const hasFocus = layer.focusEventId !== null;
  // `undefined` (not 1) when not dimmed, so an unfocused layer's markers
  // render with no `opacity` clause at all - identical output to before
  // focus support existed.
  const opacityFor = (isFocused: boolean): number | undefined =>
    hasFocus && !isFocused ? DIMMED_OPACITY : undefined;

  // Focused route: primary color, thick, fully opaque. Other routes while a
  // focus is active: neutral grey, thin, heavily dimmed. No focus: default.
  const routePathOptions = (isFocused: boolean, fireEventId: number) => {
    if (hasFocus && isFocused) {
      return { color: ROUTE_COLOR_FOCUSED, weight: ROUTE_WEIGHT_FOCUSED, opacity: ROUTE_OPACITY_FOCUSED };
    }
    if (hasFocus) {
      return { color: ROUTE_COLOR_DIMMED, weight: ROUTE_WEIGHT_DIMMED, opacity: DIMMED_OPACITY };
    }
    // Explicit opacity: Leaflet's setStyle keeps a stale dimmed value otherwise.
    return {
      color: EVENT_ROUTE_PALETTE[fireEventId % EVENT_ROUTE_PALETTE.length],
      weight: ROUTE_WEIGHT_DEFAULT,
      opacity: 1,
    };
  };

  return (
    <>
      {layer.routes.map((route) => (
        <FocusPolyline
          key={route.actionKey}
          isFocused={route.isFocused}
          positions={route.path.map((point): [number, number] => [point.latitude, point.longitude])}
          pathOptions={routePathOptions(route.isFocused, route.fireEventId)}
        />
      ))}

      {layer.routes.map((route) =>
        route.lastMileGap ? (
          <Polyline
            key={`last-mile-${route.actionKey}`}
            positions={route.lastMileGap.map((point): [number, number] => [point.latitude, point.longitude])}
            pathOptions={{
              color: hasFocus && route.isFocused ? ROUTE_COLOR_FOCUSED : ROUTE_COLOR_DIMMED,
              weight: LAST_MILE_GAP_WEIGHT,
              dashArray: LAST_MILE_GAP_DASH,
              opacity: opacityFor(route.isFocused) ?? (hasFocus && route.isFocused ? 1 : 0.85),
            }}
          />
        ) : null,
      )}

      {layer.fireMarkers.map((fire) => (
        <Marker
          key={`fire-${fire.key}`}
          position={[fire.coordinate.latitude, fire.coordinate.longitude]}
          icon={createFireIcon(FIRE_MARKER_COLOR, {
            size: fire.isFocused ? FIRE_MARKER_SIZE_FOCUSED : FIRE_MARKER_SIZE,
            opacity: opacityFor(fire.isFocused),
          })}
        >
          <Popup>
            <strong>Active fire</strong>
            <p>{eventLabel(fire.fireEventId)}</p>
          </Popup>
        </Marker>
      ))}

      {layer.targetMarkers.map((target) => (
        <Marker
          key={`target-${target.key}`}
          position={[target.coordinate.latitude, target.coordinate.longitude]}
          icon={createCircleIcon(PREDICTED_RISK_COLOR, {
            size: target.isFocused ? MARKER_SIZE_FOCUSED : MARKER_SIZE,
            filled: false,
            borderStyle: "dashed",
            opacity: opacityFor(target.isFocused),
          })}
        >
          <Popup>
            <strong>Response target #{target.targetId}</strong>
            <p>{eventLabel(target.fireEventId)}</p>
          </Popup>
        </Marker>
      ))}

      {layer.originMarkers.map((origin) => (
        <Marker
          key={`origin-${origin.actionKey}`}
          position={[origin.coordinate.latitude, origin.coordinate.longitude]}
          icon={createCircleIcon(ORIGIN_MARKER_COLOR, {
            size: origin.isFocused ? MARKER_SIZE_FOCUSED : MARKER_SIZE,
            opacity: opacityFor(origin.isFocused),
          })}
        >
          <Popup>
            <strong>Resource: {origin.resourceId}</strong>
            <p>{eventLabel(origin.fireEventId)}</p>
          </Popup>
        </Marker>
      ))}

      {layer.stationMarkers.map((station) => (
        <Marker
          key={`station-${station.stationId}`}
          position={[station.coordinate.latitude, station.coordinate.longitude]}
          icon={createStationIcon(STATION_ALLOCATED_COLOR, {
            size: station.isFocused ? STATION_SIZE_FOCUSED : STATION_SIZE,
            opacity: opacityFor(station.isFocused),
          })}
        >
          <Popup>
            <strong>{translateStationLabel(station.stationName ?? station.stationId)}</strong>
          </Popup>
        </Marker>
      ))}
    </>
  );
}
