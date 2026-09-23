import { useEffect, useRef } from "react";
import type L from "leaflet";
import { Marker, Polyline, Popup } from "react-leaflet";
import { createCircleIcon, createFireIcon } from "../map/divIcon";
import { translateStationLabel } from "../map/stationTranslations";
import { TARGET_TYPE_LABELS } from "./presentation";
import type { ResponseRouteLayerData, ResponseTargetMarker } from "./ResponseRouteLayerModel";
import "./ResponsePlanMapLayer.css";

export interface ResponsePlanMapLayerProps {
  layer: ResponseRouteLayerData;
  /** The parent FireEvent's location name, matching the list's target titles. */
  locationName?: string | null;
  /** Per-target place names from reverse geocoding (target id -> name); preferred over `locationName`. */
  targetLocations?: Record<number, string | null>;
}

// Route legend (see RouteLegend) mirrors these exact styles.
export const SELECTED_ROUTE_COLOR = "var(--color-focus-ring)";
export const OTHER_ROUTE_COLOR = "var(--color-text-muted)";
const SELECTED_ROUTE_WEIGHT = 6;
const OTHER_ROUTE_WEIGHT = 3;
const OTHER_ROUTE_DASH = "6 8";
// No route selected: every route is drawn the same, solid and mid-weight.
const NEUTRAL_ROUTE_WEIGHT = 4;

const TARGET_FIRE_COLOR = "#ff4500";
const ORIGIN_MARKER_COLOR = "var(--color-info)";
// "Last mile" connector: bridges the routed path's snapped-to-road end
// point to the target's own (off-road) coordinate. Always dashed and thin,
// regardless of selection state - it is a visual bridge, never mistaken for
// part of the actual computed route.
const LAST_MILE_GAP_DASH = "5, 10";
const LAST_MILE_GAP_WEIGHT = 2;

const MARKER_SIZE = 16;
const MARKER_SIZE_SELECTED = 22;
const TARGET_SIZE = 28;
const TARGET_SIZE_SELECTED = 38;
const DIMMED_OPACITY = 0.45;

/** Same wording as the Response Actions list: "Active fire - Modiin", or the id when no place is known. */
function targetPopupTitle(target: ResponseTargetMarker, place: string | null): string {
  const type = target.targetType !== null ? TARGET_TYPE_LABELS[target.targetType] : null;
  if (place) {
    return type !== null ? `${type} - ${place}` : place;
  }
  return type !== null ? `Target #${target.targetId} - ${type}` : `Target #${target.targetId}`;
}

/**
 * Leaflet's `setStyle` only overwrites the keys it is given, so a route that
 * changes from dashed to solid (or dimmed back to full) keeps its old
 * `dashArray`/`opacity` unless they are passed explicitly. Every state
 * therefore sets all four keys - `dashArray: undefined` is what clears a
 * previously applied dash.
 */
export function routePathOptions(isSelected: boolean, hasSelection: boolean) {
  if (isSelected) {
    return { color: SELECTED_ROUTE_COLOR, weight: SELECTED_ROUTE_WEIGHT, dashArray: undefined, opacity: 1 };
  }
  if (hasSelection) {
    return { color: OTHER_ROUTE_COLOR, weight: OTHER_ROUTE_WEIGHT, dashArray: OTHER_ROUTE_DASH, opacity: 0.85 };
  }
  return { color: SELECTED_ROUTE_COLOR, weight: NEUTRAL_ROUTE_WEIGHT, dashArray: undefined, opacity: 1 };
}

type PolylineProps = Parameters<typeof Polyline>[0];

/**
 * A route line that brings itself to the front of the map's SVG layer when
 * it becomes the selected route. Leaflet draws paths in the order they were
 * added, so without this the dashed alternatives (added later) would paint
 * over the selected solid line.
 */
function RoutePolyline({ isSelected, ...props }: PolylineProps & { isSelected: boolean }) {
  const ref = useRef<L.Polyline | null>(null);
  useEffect(() => {
    if (isSelected) {
      ref.current?.bringToFront();
    }
  }, [isSelected]);
  return <Polyline ref={ref} {...props} />;
}

/**
 * Renders US 6.3's precomputed route/marker data (see
 * `ResponseRouteLayerModel.buildResponseRouteLayer`) onto US 6.2's shared
 * `MapView` surface. `layer` has already excluded anything that isn't a
 * persisted, reachable, >=2-point path (see `isDrawableRoute`) and anything
 * missing its own coordinate - this component draws exactly what it is
 * given (no coordinate derivation, route calculation, or network call).
 *
 * Styling is one consistent vocabulary: the selected route is thick, solid,
 * and primary blue; when a route is selected every other route is thin,
 * dashed, and grey. Targets are fire glyphs; resources are blue dots.
 */
export function ResponsePlanMapLayer({
  layer,
  locationName = null,
  targetLocations = {},
}: ResponsePlanMapLayerProps) {
  const hasSelection =
    layer.routes.some((route) => route.isSelected) ||
    layer.originMarkers.some((origin) => origin.isSelected) ||
    layer.targetMarkers.some((target) => target.isSelected);
  const dimmedOpacity = (isSelected: boolean): number | undefined =>
    hasSelection && !isSelected ? DIMMED_OPACITY : undefined;

  return (
    <>
      {layer.routes.map((route) => (
        <RoutePolyline
          key={route.actionKey}
          isSelected={route.isSelected}
          positions={route.path.map((point): [number, number] => [point.latitude, point.longitude])}
          pathOptions={routePathOptions(route.isSelected, hasSelection)}
        />
      ))}

      {layer.routes.map((route) =>
        route.lastMileGap ? (
          <Polyline
            key={`last-mile-${route.actionKey}`}
            positions={route.lastMileGap.map((point): [number, number] => [point.latitude, point.longitude])}
            pathOptions={{
              color: route.isSelected ? SELECTED_ROUTE_COLOR : OTHER_ROUTE_COLOR,
              weight: LAST_MILE_GAP_WEIGHT,
              dashArray: LAST_MILE_GAP_DASH,
              opacity: dimmedOpacity(route.isSelected) ?? (route.isSelected ? 1 : 0.85),
            }}
          />
        ) : null,
      )}

      {layer.originMarkers.map((origin) => (
        <Marker
          key={`origin-${origin.actionKey}`}
          position={[origin.coordinate.latitude, origin.coordinate.longitude]}
          icon={createCircleIcon(ORIGIN_MARKER_COLOR, {
            size: origin.isSelected ? MARKER_SIZE_SELECTED : MARKER_SIZE,
            opacity: dimmedOpacity(origin.isSelected),
          })}
        >
          <Popup>
            <strong className="origin-popup__station">
              {origin.stationName !== null ? translateStationLabel(origin.stationName) : "Station not available"}
            </strong>
            <p className="origin-popup__heading">Allocated in current plan</p>
            <ul className="origin-popup__trucks">
              {origin.resourceIds.map((resourceId) => (
                <li key={resourceId} className="origin-popup__truck">
                  {resourceId}
                </li>
              ))}
            </ul>
          </Popup>
        </Marker>
      ))}

      {layer.targetMarkers.map((target) => (
        <Marker
          key={`target-${target.actionKey}`}
          position={[target.coordinate.latitude, target.coordinate.longitude]}
          icon={createFireIcon(TARGET_FIRE_COLOR, {
            size: target.isSelected ? TARGET_SIZE_SELECTED : TARGET_SIZE,
            opacity: dimmedOpacity(target.isSelected),
          })}
        >
          <Popup>
            <strong>{targetPopupTitle(target, targetLocations[target.targetId] ?? locationName)}</strong>
          </Popup>
        </Marker>
      ))}
    </>
  );
}
