import { describe, expect, it } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { buildResponseRouteLayer, getResponseActionKey, isDrawableRoute } from "./ResponseRouteLayerModel";

function makeAction(overrides: Partial<ResponsePlanAction> = {}): ResponsePlanAction {
  return {
    resource: {
      resource_id: "engine-1",
      station_id: "station-1",
      station_name: "Central Station",
      origin: { latitude: 32.0, longitude: 35.0 },
    },
    target: {
      response_target_id: 1,
      target_type: "active_fire",
      priority_score: 0.75,
      latitude: 32.5,
      longitude: 35.5,
    },
    route: {
      status: "reachable",
      eta_seconds: 120,
      distance_meters: 800,
      node_path: [1, 2, 3],
      path_coordinates: [
        { latitude: 32.0, longitude: 35.0 },
        { latitude: 32.25, longitude: 35.25 },
        { latitude: 32.5, longitude: 35.5 },
      ],
    },
    ...overrides,
  };
}

describe("getResponseActionKey", () => {
  it("uses the resource_id as the stable key", () => {
    expect(getResponseActionKey(makeAction({ resource: { ...makeAction().resource, resource_id: "engine-9", station_id: "station-engine-9", station_name: "engine-9" } }))).toBe(
      "engine-9",
    );
  });
});

describe("isDrawableRoute", () => {
  it("is true for a reachable route with a multi-point path", () => {
    expect(isDrawableRoute(makeAction().route)).toBe(true);
  });

  it("is false for an unreachable route", () => {
    expect(
      isDrawableRoute({ status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null }),
    ).toBe(false);
  });

  it("is false for an unmappable route", () => {
    expect(
      isDrawableRoute({ status: "unmappable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null }),
    ).toBe(false);
  });

  it("is false for a reachable route with missing path_coordinates", () => {
    expect(
      isDrawableRoute({ status: "reachable", eta_seconds: 10, distance_meters: 10, node_path: [1, 2], path_coordinates: null }),
    ).toBe(false);
  });

  it("is false for a reachable route with only a single path coordinate", () => {
    expect(
      isDrawableRoute({
        status: "reachable",
        eta_seconds: 10,
        distance_meters: 10,
        node_path: [1],
        path_coordinates: [{ latitude: 1, longitude: 1 }],
      }),
    ).toBe(false);
  });

  it("is false when route status is null", () => {
    expect(
      isDrawableRoute({ status: null, eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null }),
    ).toBe(false);
  });
});

describe("buildResponseRouteLayer", () => {
  it("produces a drawable route for a reachable action with persisted path_coordinates", () => {
    const layer = buildResponseRouteLayer([makeAction()], null);

    expect(layer.routes).toHaveLength(1);
    expect(layer.routes[0].resourceId).toBe("engine-1");
    expect(layer.routes[0].targetId).toBe(1);
  });

  it("preserves path-coordinate order exactly", () => {
    const path = [
      { latitude: 1, longitude: 1 },
      { latitude: 2, longitude: 2 },
      { latitude: 3, longitude: 3 },
    ];
    const layer = buildResponseRouteLayer(
      [makeAction({ route: { ...makeAction().route, path_coordinates: path } })],
      null,
    );

    expect(layer.routes[0].path).toEqual(path);
  });

  it("never derives a route from node_path when path_coordinates is missing", () => {
    const action = makeAction({
      route: { status: "reachable", eta_seconds: 10, distance_meters: 10, node_path: [1, 2, 3], path_coordinates: null },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes).toHaveLength(0);
  });

  it("does not draw a route for a reachable status with incomplete path_coordinates", () => {
    const action = makeAction({
      route: {
        status: "reachable",
        eta_seconds: 10,
        distance_meters: 10,
        node_path: [1],
        path_coordinates: [{ latitude: 1, longitude: 1 }],
      },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes).toHaveLength(0);
  });

  it("does not draw a route for an unreachable action", () => {
    const action = makeAction({
      route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes).toHaveLength(0);
  });

  it("does not draw a route for an unmappable action", () => {
    const action = makeAction({
      route: { status: "unmappable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes).toHaveLength(0);
  });

  it("uses the persisted resource.origin for the origin marker", () => {
    const origin = { latitude: 33.1, longitude: 36.1 };
    const layer = buildResponseRouteLayer([makeAction({ resource: { ...makeAction().resource, origin } })], null);

    expect(layer.originMarkers).toHaveLength(1);
    expect(layer.originMarkers[0].coordinate).toEqual(origin);
  });

  it("uses the persisted target latitude/longitude for the target marker", () => {
    const layer = buildResponseRouteLayer(
      [makeAction({ target: { ...makeAction().target, latitude: 40.0, longitude: 41.0 } })],
      null,
    );

    expect(layer.targetMarkers).toHaveLength(1);
    expect(layer.targetMarkers[0].coordinate).toEqual({ latitude: 40.0, longitude: 41.0 });
  });

  it("omits the origin marker (without crashing or dropping other data) when origin is missing", () => {
    const action = makeAction({ resource: { ...makeAction().resource, origin: null } });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.originMarkers).toHaveLength(0);
    expect(layer.targetMarkers).toHaveLength(1);
    expect(layer.routes).toHaveLength(1);
  });

  it("omits the target marker when target coordinates are missing", () => {
    const action = makeAction({ target: { ...makeAction().target, latitude: null, longitude: null } });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.targetMarkers).toHaveLength(0);
    expect(layer.originMarkers).toHaveLength(1);
  });

  it("marks only the selected action's route/markers as selected", () => {
    const first = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1", station_name: "engine-1" } });
    const second = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2", station_name: "engine-2" } });

    const layer = buildResponseRouteLayer([first, second], "engine-2");

    expect(layer.routes.find((r) => r.resourceId === "engine-1")?.isSelected).toBe(false);
    expect(layer.routes.find((r) => r.resourceId === "engine-2")?.isSelected).toBe(true);
    expect(layer.originMarkers.find((m) => m.resourceId === "engine-1")?.isSelected).toBe(false);
    expect(layer.originMarkers.find((m) => m.resourceId === "engine-2")?.isSelected).toBe(true);
  });

  it("merges trucks from the same origin into one marker listing every resource", () => {
    const first = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } });
    const second = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } });

    const layer = buildResponseRouteLayer([first, second], "engine-2");

    expect(layer.originMarkers).toHaveLength(1);
    expect(layer.originMarkers[0].resourceIds).toEqual(["engine-1", "engine-2"]);
    expect(layer.originMarkers[0].isSelected).toBe(true);
    expect(layer.routes).toHaveLength(2);
  });

  it("does not reorder actions based on selection", () => {
    const first = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1", station_name: "engine-1" } });
    const second = makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2", station_name: "engine-2" } });

    const layer = buildResponseRouteLayer([first, second], "engine-1");

    expect(layer.routes.map((r) => r.resourceId)).toEqual(["engine-1", "engine-2"]);
  });

  it("still includes markers for an action with no drawable route", () => {
    const action = makeAction({
      route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes).toHaveLength(0);
    expect(layer.originMarkers).toHaveLength(1);
    expect(layer.targetMarkers).toHaveLength(1);
  });

  it("returns empty feature lists for an empty actions array", () => {
    const layer = buildResponseRouteLayer([], null);

    expect(layer).toEqual({ routes: [], originMarkers: [], targetMarkers: [] });
  });
});

describe("buildResponseRouteLayer lastMileGap ('last mile' visual bridge)", () => {
  it("bridges the gap between the route's last snapped point and the target's own coordinate", () => {
    const action = makeAction({
      target: { ...makeAction().target, latitude: 32.6, longitude: 35.6 },
      route: {
        ...makeAction().route,
        path_coordinates: [
          { latitude: 32.0, longitude: 35.0 },
          { latitude: 32.5, longitude: 35.5 }, // snapped road node - not the same as the target above.
        ],
      },
    });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes[0].lastMileGap).toEqual([
      { latitude: 32.5, longitude: 35.5 },
      { latitude: 32.6, longitude: 35.6 },
    ]);
    // The bridge is purely additive presentation data - the real path is never touched.
    expect(layer.routes[0].path).toEqual(action.route.path_coordinates);
  });

  it("is null when the route already ends exactly at the target", () => {
    const layer = buildResponseRouteLayer([makeAction()], null); // default fixture: path ends at (32.5, 35.5) == target.

    expect(layer.routes[0].lastMileGap).toBeNull();
  });

  it("is null when the target's own coordinate is unknown", () => {
    const action = makeAction({ target: { ...makeAction().target, latitude: null, longitude: null } });

    const layer = buildResponseRouteLayer([action], null);

    expect(layer.routes[0].lastMileGap).toBeNull();
  });
});
