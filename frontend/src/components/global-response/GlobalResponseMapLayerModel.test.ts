import { describe, expect, it } from "vitest";
import { buildGlobalResponseMapLayer } from "./GlobalResponseMapLayerModel";
import type { ResponsePlanAction, ResponsePlanTarget } from "../../types/responsePlan";
import type { GlobalEventPlan } from "../../types/globalResponsePlan";

function makeTarget(overrides: Partial<ResponsePlanTarget> = {}): ResponsePlanTarget {
  return {
    response_target_id: 1,
    target_type: "active_fire",
    priority_score: 1.0,
    latitude: 32.7,
    longitude: 35.0,
    ...overrides,
  };
}

function makeAction(overrides: Partial<ResponsePlanAction> = {}): ResponsePlanAction {
  return {
    resource: {
      resource_id: "engine-1",
      station_id: "station-1",
      station_name: "Central Station",
      origin: { latitude: 32.0, longitude: 34.8 },
    },
    target: makeTarget({ target_type: "predicted_risk", response_target_id: 2 }),
    route: {
      status: "reachable",
      eta_seconds: 120.0,
      distance_meters: 800.0,
      node_path: [1, 2],
      path_coordinates: [
        { latitude: 32.0, longitude: 34.8 },
        { latitude: 32.7, longitude: 35.0 },
      ],
    },
    ...overrides,
  };
}

function makeEvent(overrides: Partial<GlobalEventPlan> = {}): GlobalEventPlan {
  return {
    fire_event_id: 101,
    response_plan_id: 501,
    severity_level: "high",
    severity_score: 70.0,
    minimum_resources: 2,
    desired_resources: 4,
    assigned_resources: 3,
    coverage_score: 85.0,
    average_eta_seconds: 140.0,
    actions: [makeAction()],
    uncovered_targets: [],
    ...overrides,
  };
}

describe("buildGlobalResponseMapLayer", () => {
  it("returns empty feature lists for no events", () => {
    const layer = buildGlobalResponseMapLayer([], null);

    expect(layer).toEqual({
      focusEventId: null,
      routes: [],
      originMarkers: [],
      fireMarkers: [],
      targetMarkers: [],
      stationMarkers: [],
    });
  });

  it("draws a route only when the action's route is reachable with >=2 path points", () => {
    const drawable = makeAction();
    const unreachable = makeAction({
      resource: { ...drawable.resource, resource_id: "engine-2" },
      route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
    });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [drawable, unreachable] })], null);

    expect(layer.routes).toHaveLength(1);
    expect(layer.routes[0].resourceId).toBe("engine-1");
  });

  it("tags every route/marker with its owning fire_event_id", () => {
    const layer = buildGlobalResponseMapLayer([makeEvent({ fire_event_id: 202 })], null);

    expect(layer.routes[0].fireEventId).toBe(202);
    expect(layer.originMarkers[0].fireEventId).toBe(202);
  });

  it("splits active_fire targets into fireMarkers and other types into targetMarkers", () => {
    const fireAction = makeAction({
      target: makeTarget({ target_type: "active_fire", response_target_id: 10 }),
    });
    const riskAction = makeAction({
      resource: { ...fireAction.resource, resource_id: "engine-2" },
      target: makeTarget({ target_type: "predicted_risk", response_target_id: 11 }),
    });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [fireAction, riskAction] })], null);

    expect(layer.fireMarkers.map((f) => f.targetId)).toEqual([10]);
    expect(layer.targetMarkers.map((t) => t.targetId)).toEqual([11]);
    expect(layer.targetMarkers[0].isPredictedRisk).toBe(true);
  });

  it("treats a null target_type as a plain target marker, never a fire marker", () => {
    const action = makeAction({ target: makeTarget({ target_type: null, response_target_id: 12 }) });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [action] })], null);

    expect(layer.fireMarkers).toHaveLength(0);
    expect(layer.targetMarkers).toHaveLength(1);
    expect(layer.targetMarkers[0].isPredictedRisk).toBe(false);
  });

  it("omits a target marker when latitude/longitude are null, never fabricating a position", () => {
    const action = makeAction({ target: makeTarget({ latitude: null, longitude: null }) });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [action] })], null);

    expect(layer.fireMarkers).toHaveLength(0);
    expect(layer.targetMarkers).toHaveLength(0);
  });

  it("includes uncovered targets as markers too, tagged with their event", () => {
    const event = makeEvent({
      actions: [],
      uncovered_targets: [makeTarget({ target_type: "active_fire", response_target_id: 20 })],
    });
    const layer = buildGlobalResponseMapLayer([event], null);

    expect(layer.fireMarkers).toHaveLength(1);
    expect(layer.fireMarkers[0].targetId).toBe(20);
    expect(layer.fireMarkers[0].fireEventId).toBe(101);
  });

  it("omits an origin marker when the resource's station could not be resolved", () => {
    const action = makeAction({ resource: { ...makeAction().resource, origin: null } });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [action] })], null);

    expect(layer.originMarkers).toHaveLength(0);
    expect(layer.stationMarkers).toHaveLength(0);
  });

  it("deduplicates stations shared by multiple resources into one marker", () => {
    const first = makeAction();
    const second = makeAction({
      resource: { ...first.resource, resource_id: "engine-2" },
    });
    const layer = buildGlobalResponseMapLayer([makeEvent({ actions: [first, second] })], null);

    expect(layer.stationMarkers).toHaveLength(1);
    expect(layer.stationMarkers[0].stationId).toBe("station-1");
  });

  describe("focus", () => {
    it("marks nothing as focused when focusEventId is null", () => {
      const layer = buildGlobalResponseMapLayer([makeEvent({ fire_event_id: 101 })], null);

      expect(layer.focusEventId).toBeNull();
      expect(layer.routes.every((r) => !r.isFocused)).toBe(true);
      expect(layer.originMarkers.every((o) => !o.isFocused)).toBe(true);
    });

    it("marks only the matching event's features as focused", () => {
      const focused = makeEvent({ fire_event_id: 101 });
      const other = makeEvent({
        fire_event_id: 202,
        actions: [makeAction({ resource: { ...makeAction().resource, resource_id: "engine-9", station_id: "station-2" } })],
      });
      const layer = buildGlobalResponseMapLayer([focused, other], 101);

      const focusedRoute = layer.routes.find((r) => r.fireEventId === 101);
      const otherRoute = layer.routes.find((r) => r.fireEventId === 202);
      expect(focusedRoute?.isFocused).toBe(true);
      expect(otherRoute?.isFocused).toBe(false);
    });

    it("marks a shared station as focused if any serving event is focused", () => {
      const focused = makeEvent({ fire_event_id: 101 });
      const other = makeEvent({
        fire_event_id: 202,
        actions: [makeAction()], // same station-1
      });
      const layerFocusFirst = buildGlobalResponseMapLayer([focused, other], 101);
      expect(layerFocusFirst.stationMarkers[0].isFocused).toBe(true);

      const layerFocusSecond = buildGlobalResponseMapLayer([other, focused], 101);
      expect(layerFocusSecond.stationMarkers[0].isFocused).toBe(true);
    });

    it("returns unfocused for an unmatched focusEventId (e.g. an id not present)", () => {
      const layer = buildGlobalResponseMapLayer([makeEvent({ fire_event_id: 101 })], 999);

      expect(layer.focusEventId).toBe(999);
      expect(layer.routes.every((r) => !r.isFocused)).toBe(true);
    });
  });
});
