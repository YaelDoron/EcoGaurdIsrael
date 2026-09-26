import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { ResponsePlan, ResponsePlanEnvelopeResponse } from "../types/responsePlan";
import { useResponsePlan } from "./useResponsePlan";
import type { ResponsePlanSource } from "./useResponsePlan";

const { getCurrentResponsePlanMock, getResponsePlanByIdMock } = vi.hoisted(() => ({
  getCurrentResponsePlanMock: vi.fn(),
  getResponsePlanByIdMock: vi.fn(),
}));

vi.mock("../api/responsePlans", () => ({
  getCurrentResponsePlan: getCurrentResponsePlanMock,
  getResponsePlanById: getResponsePlanByIdMock,
}));

function makePlan(overrides: Partial<ResponsePlan> = {}): ResponsePlan {
  return {
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
    actions: [],
    uncovered_targets: [],
    baseline_comparison: null,
    optimization_config: null,
    no_resources_during_planning: false,
    ...overrides,
  };
}

function envelope(plan: ResponsePlan | null): ResponsePlanEnvelopeResponse {
  return { plan };
}

describe("useResponsePlan", () => {
  beforeEach(() => {
    getCurrentResponsePlanMock.mockReset();
    getResponsePlanByIdMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("requests the current plan for the given FireEvent ID", async () => {
    getCurrentResponsePlanMock.mockResolvedValue(envelope(makePlan()));

    renderHook(() => useResponsePlan({ kind: "current", fireEventId: 42 }));

    await waitFor(() => expect(getCurrentResponsePlanMock).toHaveBeenCalledWith(42, expect.any(AbortSignal)));
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
  });

  it("requests the plan by the given plan ID", async () => {
    getResponsePlanByIdMock.mockResolvedValue(envelope(makePlan()));

    renderHook(() => useResponsePlan({ kind: "by-id", planId: 99 }));

    await waitFor(() => expect(getResponsePlanByIdMock).toHaveBeenCalledWith(99, expect.any(AbortSignal)));
    expect(getCurrentResponsePlanMock).not.toHaveBeenCalled();
  });

  it("exposes a successful plan unchanged", async () => {
    const plan = makePlan({ plan_id: 55 });
    getResponsePlanByIdMock.mockResolvedValue(envelope(plan));

    const { result } = renderHook(() => useResponsePlan({ kind: "by-id", planId: 55 }));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.plan).toEqual(plan);
    expect(result.current.error).toBeNull();
  });

  it("preserves plan: null as a valid successful result, not an error", async () => {
    getCurrentResponsePlanMock.mockResolvedValue(envelope(null));

    const { result } = renderHook(() => useResponsePlan({ kind: "current", fireEventId: 3 }));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.plan).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("produces a safe error state on API failure", async () => {
    getCurrentResponsePlanMock.mockRejectedValue(new ApiError("boom", 500));

    const { result } = renderHook(() => useResponsePlan({ kind: "current", fireEventId: 3 }));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.error).toBeTruthy();
    expect(result.current.error).not.toContain("boom");
    expect(result.current.plan).toBeNull();
  });

  it("retry repeats the same request", async () => {
    getResponsePlanByIdMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(envelope(makePlan({ plan_id: 7 })));

    const { result } = renderHook(() => useResponsePlan({ kind: "by-id", planId: 7 }));

    await waitFor(() => expect(result.current.error).toBeTruthy());

    result.current.retry();

    await waitFor(() => expect(result.current.plan).not.toBeNull());
    expect(getResponsePlanByIdMock).toHaveBeenCalledTimes(2);
    expect(getResponsePlanByIdMock).toHaveBeenNthCalledWith(2, 7, expect.any(AbortSignal));
  });

  it("loads the new plan when the source changes", async () => {
    getCurrentResponsePlanMock.mockResolvedValue(envelope(makePlan({ fire_event_id: 1, plan_id: 1 })));
    getResponsePlanByIdMock.mockResolvedValue(envelope(makePlan({ plan_id: 2 })));

    const { result, rerender } = renderHook(({ source }: { source: ResponsePlanSource }) => useResponsePlan(source), {
      initialProps: { source: { kind: "current", fireEventId: 1 } as ResponsePlanSource },
    });

    await waitFor(() => expect(result.current.plan?.plan_id).toBe(1));

    rerender({ source: { kind: "by-id", planId: 2 } as ResponsePlanSource });

    await waitFor(() => expect(result.current.plan?.plan_id).toBe(2));
    expect(getResponsePlanByIdMock).toHaveBeenCalledWith(2, expect.any(AbortSignal));
  });

  it("does not retain the previous plan when the new source's request fails", async () => {
    getCurrentResponsePlanMock.mockResolvedValue(envelope(makePlan({ fire_event_id: 1, plan_id: 1 })));
    getResponsePlanByIdMock.mockRejectedValue(new ApiError("boom", 500));

    const { result, rerender } = renderHook(({ source }: { source: ResponsePlanSource }) => useResponsePlan(source), {
      initialProps: { source: { kind: "current", fireEventId: 1 } as ResponsePlanSource },
    });

    await waitFor(() => expect(result.current.plan?.plan_id).toBe(1));

    rerender({ source: { kind: "by-id", planId: 2 } as ResponsePlanSource });

    await waitFor(() => expect(result.current.error).toBeTruthy());
    expect(result.current.plan).toBeNull();
  });
});
