import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getCurrentResponsePlan, getResponsePlanById } from "./responsePlans";
import { ApiError } from "./errors";
import type { ResponsePlan } from "../types/responsePlan";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

const FULL_PLAN: ResponsePlan = {
  plan_id: 7,
  fire_event_id: 3,
  response_target_set_id: 9,
  route_planning_run_id: 11,
  generated_at: "2026-09-17T13:20:00Z",
  methodology: "genetic_algorithm",
  methodology_version: "1.0.0",
  random_seed: 42,
  status: "complete",
  is_current: true,
  metrics: { plan_score: 95.5, coverage_score: 0.9, average_eta_seconds: 150.0 },
  actions: [
    {
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
        latitude: 32.1,
        longitude: 35.1,
      },
      route: {
        status: "reachable",
        eta_seconds: 120.0,
        distance_meters: 800.0,
        node_path: [1, 2, 3],
        path_coordinates: [
          { latitude: 1.0, longitude: 1.0 },
          { latitude: 2.0, longitude: 2.0 },
          { latitude: 3.0, longitude: 3.0 },
        ],
      },
    },
    {
      resource: {
        resource_id: "engine-2",
        station_id: "station-2",
        station_name: null,
        origin: null,
      },
      target: {
        response_target_id: 2,
        target_type: "predicted_risk",
        priority_score: 0.5,
        latitude: 33.0,
        longitude: 36.0,
      },
      route: {
        status: "unreachable",
        eta_seconds: null,
        distance_meters: null,
        node_path: null,
        path_coordinates: null,
      },
    },
  ],
  uncovered_targets: [
    { response_target_id: 5, target_type: "active_fire", priority_score: 0.3, latitude: 40.0, longitude: 41.0 },
    { response_target_id: 2, target_type: null, priority_score: null, latitude: null, longitude: null },
  ],
  baseline_comparison: {
    baseline_score: 50.0,
    baseline_coverage_score: 0.5,
    baseline_average_eta_seconds: 300.0,
    score_difference: 10.0,
    improvement_percentage: 20.0,
  },
  optimization_config: {
    population_size: 50,
    generation_count: 100,
    mutation_rate: 0.1,
    crossover_rate: 0.8,
    eta_reference_seconds: 600.0,
    initial_assignment_probability: 0.5,
    tournament_size: 3,
    elitism_count: 2,
  },
  no_resources_during_planning: false,
};

describe("getCurrentResponsePlan", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("requests the correct path", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ plan: null }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getCurrentResponsePlan(3);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/fire-events/3/response-plan",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns plan: null unchanged when there is no current plan", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: null })) as unknown as typeof fetch;

    const result = await getCurrentResponsePlan(3);

    expect(result).toEqual({ plan: null });
  });

  it("returns a full enriched plan payload without transformation", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN })) as unknown as typeof fetch;

    const result = await getCurrentResponsePlan(3);

    expect(result).toEqual({ plan: FULL_PLAN });
  });

  it("propagates an ApiError on failure, without a second error system", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ error: { code: "SOME_CODE", message: "server error" } }, { status: 500 }),
      ) as unknown as typeof fetch;

    await expect(getCurrentResponsePlan(3)).rejects.toBeInstanceOf(ApiError);
  });
});

describe("getResponsePlanById", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("requests the correct path", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getResponsePlanById(7);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/response-plans/7",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("preserves a superseded plan (is_current=false) unchanged", async () => {
    const superseded: ResponsePlan = { ...FULL_PLAN, is_current: false };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: superseded })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.is_current).toBe(false);
    expect(result).toEqual({ plan: superseded });
  });

  it("preserves plan status unchanged", async () => {
    const partial: ResponsePlan = { ...FULL_PLAN, status: "partial" };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: partial })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.status).toBe("partial");
  });

  it("preserves actions in backend order", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.actions.map((action) => action.resource.resource_id)).toEqual(["engine-1", "engine-2"]);
  });

  it("preserves uncovered targets in backend order", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.uncovered_targets.map((target) => target.response_target_id)).toEqual([5, 2]);
  });

  it("preserves nullable route information exactly", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    const secondRoute = result.plan?.actions[1]?.route;
    expect(secondRoute?.status).toBe("unreachable");
    expect(secondRoute?.eta_seconds).toBeNull();
    expect(secondRoute?.distance_meters).toBeNull();
    expect(secondRoute?.node_path).toBeNull();
    expect(secondRoute?.path_coordinates).toBeNull();
  });

  it("preserves path_coordinates exactly, in order", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: FULL_PLAN })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.actions[0]?.route.path_coordinates).toEqual([
      { latitude: 1.0, longitude: 1.0 },
      { latitude: 2.0, longitude: 2.0 },
      { latitude: 3.0, longitude: 3.0 },
    ]);
  });

  it("allows baseline_comparison to be null", async () => {
    const withoutBaseline: ResponsePlan = { ...FULL_PLAN, baseline_comparison: null };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: withoutBaseline })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.baseline_comparison).toBeNull();
  });

  it("allows optimization_config to be null", async () => {
    const withoutConfig: ResponsePlan = { ...FULL_PLAN, optimization_config: null };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: withoutConfig })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.optimization_config).toBeNull();
  });

  it("preserves no_resources_during_planning", async () => {
    const noResources: ResponsePlan = { ...FULL_PLAN, no_resources_during_planning: true };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ plan: noResources })) as unknown as typeof fetch;

    const result = await getResponsePlanById(7);

    expect(result.plan?.no_resources_during_planning).toBe(true);
  });

  it("propagates an ApiError on failure (e.g. 404), without a second error system", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ error: { code: "RESPONSE_PLAN_NOT_FOUND", message: "not found" } }, { status: 404 }),
      ) as unknown as typeof fetch;

    await expect(getResponsePlanById(9999)).rejects.toBeInstanceOf(ApiError);
  });
});
