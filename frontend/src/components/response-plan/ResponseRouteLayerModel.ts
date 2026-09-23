import type { Coordinate, ResponsePlanAction, ResponsePlanRoute, ResponseTargetType } from "../../types/responsePlan";

/**
 * Pure, map-framework-agnostic data model for the Response Plan map/route
 * presentation layer (Epic 6, US 6.3, Task 11).
 *
 * US 6.2 owns the shared map surface - as of this task, no such shared
 * infrastructure (`MapView`, Leaflet, or otherwise) exists yet in this
 * repository (no map library dependency, no `MapView` component). Rather
 * than invent a competing map framework, this module computes only the
 * *data* a future map layer needs (drawable routes, origin markers, target
 * markers) from persisted `ResponsePlanAction` fields. It performs no
 * pathfinding, no interpolation, no coordinate derivation from `node_path`,
 * and no network calls - every coordinate here is copied unchanged from
 * `resource.origin`/`target.latitude`+`target.longitude`/
 * `route.path_coordinates`. See `ResponseRouteLayer.tsx` for the thin React
 * wrapper around this module.
 */

/** Stable per-action identity for selection/highlighting.
 *
 * A `resource_id` appears in at most one action per plan (enforced by the
 * backend's `ResponsePlan` invariants - see
 * backend/src/models/response_plan.py: "a resource may appear in at most
 * one response action"), so it alone is a safe, stable key - no index or
 * synthetic id is invented here.
 */
export function getResponseActionKey(action: ResponsePlanAction): string {
  return action.resource.resource_id;
}

export interface ResponseRouteFeature {
  actionKey: string;
  resourceId: string;
  targetId: number;
  /** The exact persisted `route.path_coordinates`, in backend order - never
   * modified or extended (see this module's docstring: every coordinate is
   * copied unchanged). */
  path: Coordinate[];
  /**
   * A visual-only 2-point connector from `path`'s last (snapped-to-road)
   * point to the target's own `[latitude, longitude]`, when the two differ -
   * bridges the "last mile" gap between where the routed path physically
   * ends (the nearest road node) and the fire icon itself, which is
   * otherwise off-road and can sit a visible distance from the line's end.
   * `null` when the route already ends exactly at the target (nothing to
   * bridge) or the target's own coordinate isn't known. This is presentation
   * only - not a routed/computed path segment, and not part of `path`
   * itself - the renderer is expected to draw it visually distinct (e.g.
   * dashed) from the real route.
   */
  lastMileGap: [Coordinate, Coordinate] | null;
  isSelected: boolean;
}

/** Exported for reuse by GlobalResponseMapLayerModel, which computes the
 * same "last mile" gap for its own, separately-typed route features. */
export function coordinatesEqual(a: Coordinate, b: Coordinate): boolean {
  return a.latitude === b.latitude && a.longitude === b.longitude;
}

export interface ResponseOriginMarker {
  /** Key of the first action dispatched from this origin (used as the React key). */
  actionKey: string;
  /** The first resource dispatched from this origin. */
  resourceId: string;
  /** Every resource dispatched from this origin, in backend order (one marker per origin, not per truck). */
  resourceIds: string[];
  /** The action's persisted `resource.station_name`; `null` when the station no longer resolves. */
  stationName: string | null;
  coordinate: Coordinate;
  /** True when any action dispatched from this origin is selected. */
  isSelected: boolean;
}

export interface ResponseTargetMarker {
  actionKey: string;
  targetId: number;
  /** The target's persisted `target_type`, or `null` when it no longer resolves. */
  targetType: ResponseTargetType | null;
  coordinate: Coordinate;
  isSelected: boolean;
}

export interface ResponseRouteLayerData {
  routes: ResponseRouteFeature[];
  originMarkers: ResponseOriginMarker[];
  targetMarkers: ResponseTargetMarker[];
}

/**
 * A route is drawable only when the backend marked it `reachable` AND
 * persisted a `path_coordinates` list with at least two points (a single
 * point cannot form a line). `unreachable`/`unmappable`/missing
 * `path_coordinates` are never drawn as a fake or straight-line route.
 */
export function isDrawableRoute(route: ResponsePlanRoute): boolean {
  return route.status === "reachable" && route.path_coordinates !== null && route.path_coordinates.length >= 2;
}

/**
 * Build the map layer's feature set from persisted plan actions. Every
 * action is inspected independently - an action with no drawable route
 * (missing/incomplete `path_coordinates`, or a non-reachable status) is
 * simply omitted from `routes`, never replaced with a fabricated line; its
 * origin/target markers are still included when their own coordinates are
 * available.
 */
export function buildResponseRouteLayer(
  actions: ResponsePlanAction[],
  selectedActionKey: string | null,
): ResponseRouteLayerData {
  const routes: ResponseRouteFeature[] = [];
  const originMarkers: ResponseOriginMarker[] = [];
  const targetMarkers: ResponseTargetMarker[] = [];
  const originIndex = new Map<string, ResponseOriginMarker>();

  for (const action of actions) {
    const actionKey = getResponseActionKey(action);
    const isSelected = actionKey === selectedActionKey;

    if (action.resource.origin !== null) {
      // All trucks leaving the same station share one origin marker.
      const { latitude, longitude } = action.resource.origin;
      const originKey = `${action.resource.station_name ?? ""}|${latitude},${longitude}`;
      const existing = originIndex.get(originKey);
      if (existing) {
        existing.resourceIds.push(action.resource.resource_id);
        existing.isSelected = existing.isSelected || isSelected;
      } else {
        const marker: ResponseOriginMarker = {
          actionKey,
          resourceId: action.resource.resource_id,
          resourceIds: [action.resource.resource_id],
          stationName: action.resource.station_name,
          coordinate: action.resource.origin,
          isSelected,
        };
        originIndex.set(originKey, marker);
        originMarkers.push(marker);
      }
    }

    if (action.target.latitude !== null && action.target.longitude !== null) {
      targetMarkers.push({
        actionKey,
        targetId: action.target.response_target_id,
        targetType: action.target.target_type,
        coordinate: { latitude: action.target.latitude, longitude: action.target.longitude },
        isSelected,
      });
    }

    if (isDrawableRoute(action.route)) {
      // isDrawableRoute already confirmed path_coordinates is non-null and has >=2 points.
      const path = action.route.path_coordinates as Coordinate[];
      const lastPoint = path[path.length - 1];
      const targetCoordinate =
        action.target.latitude !== null && action.target.longitude !== null
          ? { latitude: action.target.latitude, longitude: action.target.longitude }
          : null;
      const lastMileGap: [Coordinate, Coordinate] | null =
        targetCoordinate !== null && !coordinatesEqual(lastPoint, targetCoordinate)
          ? [lastPoint, targetCoordinate]
          : null;

      routes.push({
        actionKey,
        resourceId: action.resource.resource_id,
        targetId: action.target.response_target_id,
        path,
        lastMileGap,
        isSelected,
      });
    }
  }

  return { routes, originMarkers, targetMarkers };
}
