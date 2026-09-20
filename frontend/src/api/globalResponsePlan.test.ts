import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getCurrentGlobalResponsePlan } from "./globalResponsePlan";
import { ApiError } from "./errors";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

const FULL_RESPONSE: GlobalResponsePlanResponse = {
  as_of: "2026-09-20T09:00:00Z",
  plan: {
    run_id: 77,
    started_at: "2026-09-20T08:00:00Z",
    completed_at: "2026-09-20T08:05:00Z",
    status: "completed",
    metrics: { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0 },
    shortage: { total_required: 3, total_desired: 6, total_assigned: 4, unmet_required: 0, unmet_desired: 2 },
    optimization_config: {
      random_seed: 42,
      population_size: 50,
      generation_count: 100,
      mutation_rate: 0.1,
      crossover_rate: 0.8,
    },
    events: [
      {
        fire_event_id: 101,
        response_plan_id: 501,
        severity_level: "high",
        severity_score: 70.0,
        minimum_resources: 2,
        desired_resources: 4,
        assigned_resources: 3,
        coverage_score: 85.0,
        average_eta_seconds: 140.0,
        actions: [],
        uncovered_targets: [],
      },
    ],
  },
};

describe("getCurrentGlobalResponsePlan", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("requests the correct path", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ as_of: "2026-09-20T09:00:00Z", plan: null }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getCurrentGlobalResponsePlan();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/global-response-plan/current",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns plan: null unchanged when no generation has materialized yet", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ as_of: "2026-09-20T09:00:00Z", plan: null })) as unknown as typeof fetch;

    const result = await getCurrentGlobalResponsePlan();

    expect(result).toEqual({ as_of: "2026-09-20T09:00:00Z", plan: null });
  });

  it("returns a full response payload without transformation", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(FULL_RESPONSE)) as unknown as typeof fetch;

    const result = await getCurrentGlobalResponsePlan();

    expect(result).toEqual(FULL_RESPONSE);
  });

  it("preserves shortage totals exactly", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(FULL_RESPONSE)) as unknown as typeof fetch;

    const result = await getCurrentGlobalResponsePlan();

    expect(result.plan?.shortage).toEqual({
      total_required: 3,
      total_desired: 6,
      total_assigned: 4,
      unmet_required: 0,
      unmet_desired: 2,
    });
  });

  it("preserves events in backend order", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(FULL_RESPONSE)) as unknown as typeof fetch;

    const result = await getCurrentGlobalResponsePlan();

    expect(result.plan?.events.map((event) => event.fire_event_id)).toEqual([101]);
  });

  it("allows optimization_config to be null", async () => {
    const withoutConfig: GlobalResponsePlanResponse = {
      ...FULL_RESPONSE,
      plan: FULL_RESPONSE.plan ? { ...FULL_RESPONSE.plan, optimization_config: null } : null,
    };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(withoutConfig)) as unknown as typeof fetch;

    const result = await getCurrentGlobalResponsePlan();

    expect(result.plan?.optimization_config).toBeNull();
  });

  it("propagates an ApiError on failure, without a second error system", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(
        jsonResponse({ error: { code: "SOME_CODE", message: "server error" } }, { status: 500 }),
      ) as unknown as typeof fetch;

    await expect(getCurrentGlobalResponsePlan()).rejects.toBeInstanceOf(ApiError);
  });
});
