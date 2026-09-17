import type { ReactNode } from "react";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { buildResponseRouteLayer, type ResponseRouteLayerData } from "./ResponseRouteLayerModel";

export interface ResponseRouteLayerProps {
  actions: ResponsePlanAction[];
  /** The currently selected action's key (see `getResponseActionKey`), or
   * `null` when nothing is selected. Owned by `ResponsePlanPage`. */
  selectedActionKey: string | null;
  /**
   * Renders the computed, map-framework-agnostic layer data. This is the
   * plug point for the shared US 6.2 map surface once it exists (e.g. a
   * future `MapView` would pass a render function here that mounts its own
   * `<Polyline>`/`<Marker>` primitives from `layer.routes`/`originMarkers`/
   * `targetMarkers`). Omitting `children` renders nothing - safe to mount
   * before that integration exists.
   */
  children?: (layer: ResponseRouteLayerData) => ReactNode;
}

/**
 * US 6.3-owned Response Plan route/marker layer (Epic 6, Task 11).
 *
 * This is deliberately NOT a map component - US 6.2 owns the shared map
 * surface, and as of this task no such shared infrastructure (`MapView`,
 * Leaflet, or otherwise) exists yet in this repository. Rather than invent
 * a competing map framework, this component only computes the persisted,
 * ready-to-draw feature set (see `ResponseRouteLayerModel.buildResponseRouteLayer`)
 * and hands it to `children` to render however the eventual shared map
 * surface requires. No pathfinding, coordinate derivation, or network call
 * happens here.
 */
export function ResponseRouteLayer({ actions, selectedActionKey, children }: ResponseRouteLayerProps) {
  const layer = buildResponseRouteLayer(actions, selectedActionKey);

  if (!children) {
    return null;
  }

  return <>{children(layer)}</>;
}
