import { Marker, Polyline, Popup } from "react-leaflet";
import { createCircleIcon, createSquareIcon } from "../map/divIcon";
import type { ResponseRouteLayerData } from "./ResponseRouteLayerModel";

export interface ResponsePlanMapLayerProps {
  layer: ResponseRouteLayerData;
}

const ROUTE_COLOR = "var(--color-focus-ring)";
const ROUTE_COLOR_SELECTED = "var(--color-danger)";
const ROUTE_WEIGHT = 3;
const ROUTE_WEIGHT_SELECTED = 5;

const ORIGIN_MARKER_COLOR = "var(--color-focus-ring)";
const ORIGIN_MARKER_COLOR_SELECTED = "var(--color-danger)";
const TARGET_MARKER_COLOR = "var(--color-warning)";
const TARGET_MARKER_COLOR_SELECTED = "var(--color-danger)";

const MARKER_SIZE = 16;
const MARKER_SIZE_SELECTED = 22;

/**
 * Renders US 6.3's precomputed route/marker data (see
 * `ResponseRouteLayerModel.buildResponseRouteLayer`) onto US 6.2's shared
 * `MapView` surface, using the same react-leaflet primitives and
 * `createCircleIcon`/`createSquareIcon` helpers StationLayer/
 * OperationalLayer/ResponseTargetLayer already use for their markers - this
 * is the `children` render-prop `ResponseRouteLayer` was built for. `layer`
 * has already excluded anything that isn't a persisted, reachable,
 * >=2-point path (see `isDrawableRoute`) and anything missing its own
 * coordinate - this component draws exactly what it is given, and nothing
 * more (no coordinate derivation, no route calculation, no network call).
 */
export function ResponsePlanMapLayer({ layer }: ResponsePlanMapLayerProps) {
  return (
    <>
      {layer.routes.map((route) => (
        <Polyline
          key={route.actionKey}
          positions={route.path.map((point): [number, number] => [point.latitude, point.longitude])}
          pathOptions={{
            color: route.isSelected ? ROUTE_COLOR_SELECTED : ROUTE_COLOR,
            weight: route.isSelected ? ROUTE_WEIGHT_SELECTED : ROUTE_WEIGHT,
          }}
        />
      ))}

      {layer.originMarkers.map((origin) => (
        <Marker
          key={`origin-${origin.actionKey}`}
          position={[origin.coordinate.latitude, origin.coordinate.longitude]}
          icon={createSquareIcon(origin.isSelected ? ORIGIN_MARKER_COLOR_SELECTED : ORIGIN_MARKER_COLOR, {
            size: origin.isSelected ? MARKER_SIZE_SELECTED : MARKER_SIZE,
          })}
        >
          <Popup>Resource: {origin.resourceId}</Popup>
        </Marker>
      ))}

      {layer.targetMarkers.map((target) => (
        <Marker
          key={`target-${target.actionKey}`}
          position={[target.coordinate.latitude, target.coordinate.longitude]}
          icon={createCircleIcon(target.isSelected ? TARGET_MARKER_COLOR_SELECTED : TARGET_MARKER_COLOR, {
            size: target.isSelected ? MARKER_SIZE_SELECTED : MARKER_SIZE,
          })}
        >
          <Popup>Target #{target.targetId}</Popup>
        </Marker>
      ))}
    </>
  );
}
