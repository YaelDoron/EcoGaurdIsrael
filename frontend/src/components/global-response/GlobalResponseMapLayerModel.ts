import type { Coordinate, ResponsePlanTarget } from "../../types/responsePlan";
import type { GlobalEventPlan } from "../../types/globalResponsePlan";
import { coordinatesEqual, isDrawableRoute } from "../response-plan/ResponseRouteLayerModel";

/**
 * Pure, map-framework-agnostic data model for the Global Response Map layer
 * (Task B-FE-7). Mirrors `../response-plan/ResponseRouteLayerModel` in
 * spirit (reuses its `isDrawableRoute` check rather than re-implementing
 * it) but operates across EVERY materialized event's actions/uncovered
 * targets in one generation, tagging each feature with the `fire_event_id`
 * it belongs to so the map layer component can apply focus styling.
 *
 * No pathfinding, coordinate derivation, demand, or shortage calculation
 * happens here - every coordinate is copied unchanged from
 * `action.resource.origin` / `action.target.latitude`+`longitude` /
 * `action.route.path_coordinates` / `uncovered target` fields already on
 * `GlobalEventPlan`. Grouping resources into distinct station markers is a
 * presentation-only dedup of already-provided data (the same kind of join
 * `OperationalLayer.groupResourcesByStation` already performs for the
 * single-event map), not a calculation.
 */

export interface GlobalRouteFeature {
  actionKey: string;
  fireEventId: number;
  resourceId: string;
  targetId: number;
  /** The exact persisted `route.path_coordinates`, in backend order. */
  path: Coordinate[];
  /** Visual-only connector bridging `path`'s last (snapped-to-road) point to
   * the target's own coordinate - see ResponseRouteLayerModel's
   * `ResponseRouteFeature.lastMileGap` for the full rationale, mirrored here
   * for the Global Response Map's own route feature type. */
  lastMileGap: [Coordinate, Coordinate] | null;
  isFocused: boolean;
}

export interface GlobalOriginMarker {
  actionKey: string;
  fireEventId: number;
  resourceId: string;
  coordinate: Coordinate;
  isFocused: boolean;
}

/** A response target whose `target_type` is `"active_fire"` - the fire's
 * own location within one FireEvent's persisted target set. */
export interface GlobalFireMarker {
  key: string;
  fireEventId: number;
  targetId: number;
  coordinate: Coordinate;
  isFocused: boolean;
}

/** Every other located response target (`"predicted_risk"`, or a type the
 * backend could not resolve). */
export interface GlobalTargetMarker {
  key: string;
  fireEventId: number;
  targetId: number;
  isPredictedRisk: boolean;
  coordinate: Coordinate;
  isFocused: boolean;
}

export interface GlobalStationMarker {
  stationId: string;
  stationName: string | null;
  coordinate: Coordinate;
  isFocused: boolean;
}

export interface GlobalResponseMapLayerData {
  /** Echoed back so the rendering component can distinguish "no focus set
   * at all" (nothing should be dimmed) from "focus set on another event"
   * (this feature's own `isFocused: false` should be dimmed). */
  focusEventId: number | null;
  routes: GlobalRouteFeature[];
  originMarkers: GlobalOriginMarker[];
  fireMarkers: GlobalFireMarker[];
  targetMarkers: GlobalTargetMarker[];
  stationMarkers: GlobalStationMarker[];
}

function hasCoordinate(target: ResponsePlanTarget): target is ResponsePlanTarget & {
  latitude: number;
  longitude: number;
} {
  return target.latitude !== null && target.longitude !== null;
}

function pushTargetMarker(
  fireEventId: number,
  target: ResponsePlanTarget,
  isFocused: boolean,
  fireMarkers: GlobalFireMarker[],
  targetMarkers: GlobalTargetMarker[],
): void {
  if (!hasCoordinate(target)) {
    return;
  }
  const key = `${fireEventId}-${target.response_target_id}`;
  const coordinate: Coordinate = { latitude: target.latitude, longitude: target.longitude };

  if (target.target_type === "active_fire") {
    fireMarkers.push({ key, fireEventId, targetId: target.response_target_id, coordinate, isFocused });
    return;
  }
  targetMarkers.push({
    key,
    fireEventId,
    targetId: target.response_target_id,
    isPredictedRisk: target.target_type === "predicted_risk",
    coordinate,
    isFocused,
  });
}

/**
 * Build the map layer's feature set from every materialized event in the
 * current generation. `focusEventId` only affects each feature's
 * `isFocused` flag (and, for a station shared across events, whether ANY
 * event serving it is focused) - it never removes/filters a feature from
 * the returned lists, so toggling focus never changes what data was
 * fetched, only how it is styled.
 */
export function buildGlobalResponseMapLayer(
  events: GlobalEventPlan[],
  focusEventId: number | null,
): GlobalResponseMapLayerData {
  const routes: GlobalRouteFeature[] = [];
  const originMarkers: GlobalOriginMarker[] = [];
  const fireMarkers: GlobalFireMarker[] = [];
  const targetMarkers: GlobalTargetMarker[] = [];
  const stationsById = new Map<string, GlobalStationMarker>();

  for (const event of events) {
    const isFocused = focusEventId !== null && event.fire_event_id === focusEventId;

    for (const action of event.actions) {
      const actionKey = `${event.fire_event_id}:${action.resource.resource_id}`;

      if (action.resource.origin !== null) {
        originMarkers.push({
          actionKey,
          fireEventId: event.fire_event_id,
          resourceId: action.resource.resource_id,
          coordinate: action.resource.origin,
          isFocused,
        });

        const existingStation = stationsById.get(action.resource.station_id);
        if (!existingStation || (!existingStation.isFocused && isFocused)) {
          stationsById.set(action.resource.station_id, {
            stationId: action.resource.station_id,
            stationName: action.resource.station_name,
            coordinate: action.resource.origin,
            isFocused,
          });
        }
      }

      pushTargetMarker(event.fire_event_id, action.target, isFocused, fireMarkers, targetMarkers);

      if (isDrawableRoute(action.route)) {
        // isDrawableRoute already confirmed path_coordinates is non-null and has >=2 points.
        const path = action.route.path_coordinates as Coordinate[];
        const lastPoint = path[path.length - 1];
        const targetCoordinate = hasCoordinate(action.target)
          ? { latitude: action.target.latitude, longitude: action.target.longitude }
          : null;
        const lastMileGap: [Coordinate, Coordinate] | null =
          targetCoordinate !== null && !coordinatesEqual(lastPoint, targetCoordinate)
            ? [lastPoint, targetCoordinate]
            : null;

        routes.push({
          actionKey,
          fireEventId: event.fire_event_id,
          resourceId: action.resource.resource_id,
          targetId: action.target.response_target_id,
          path,
          lastMileGap,
          isFocused,
        });
      }
    }

    for (const target of event.uncovered_targets) {
      pushTargetMarker(event.fire_event_id, target, isFocused, fireMarkers, targetMarkers);
    }
  }

  return {
    focusEventId,
    routes,
    originMarkers,
    fireMarkers,
    targetMarkers,
    stationMarkers: Array.from(stationsById.values()),
  };
}
