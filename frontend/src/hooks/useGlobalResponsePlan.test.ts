import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { useGlobalResponsePlan } from "./useGlobalResponsePlan";

const { getCurrentGlobalResponsePlanMock } = vi.hoisted(() => ({
  getCurrentGlobalResponsePlanMock: vi.fn(),
}));

vi.mock("../api/globalResponsePlan", () => ({
  getCurrentGlobalResponsePlan: getCurrentGlobalResponsePlanMock,
}));

function makeResponse(overrides: Partial<GlobalResponsePlanResponse> = {}): GlobalResponsePlanResponse {
  return {
    as_of: "2026-09-20T09:00:00Z",
    plan: {
      run_id: 77,
      started_at: "2026-09-20T08:00:00Z",
      completed_at: "2026-09-20T08:05:00Z",
      status: "completed",
      metrics: { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0 },
      shortage: { total_required: 3, total_desired: 6, total_assigned: 4, unmet_required: 0, unmet_desired: 2 },
      optimization_config: null,
      events: [],
    },
    ...overrides,
  };
}

describe("useGlobalResponsePlan", () => {
  beforeEach(() => {
    getCurrentGlobalResponsePlanMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("fetches the current global response plan on mount", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderHook(() => useGlobalResponsePlan());

    await waitFor(() => expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1));
  });

  it("starts in a loading state", () => {
    getCurrentGlobalResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    const { result } = renderHook(() => useGlobalResponsePlan());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.response).toBeNull();
  });

  it("exposes a successful response unchanged", async () => {
    const response = makeResponse();
    getCurrentGlobalResponsePlanMock.mockResolvedValue(response);

    const { result } = renderHook(() => useGlobalResponsePlan());

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.response).toEqual(response);
    expect(result.current.error).toBeNull();
  });

  it("preserves plan: null as a valid successful result, not an error", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse({ plan: null }));

    const { result } = renderHook(() => useGlobalResponsePlan());

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.response?.plan).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("produces a safe error state on API failure", async () => {
    getCurrentGlobalResponsePlanMock.mockRejectedValue(new ApiError("boom", 500));

    const { result } = renderHook(() => useGlobalResponsePlan());

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.error).toBeTruthy();
    expect(result.current.error).not.toContain("boom");
    expect(result.current.response).toBeNull();
  });

  it("retry repeats the same request", async () => {
    getCurrentGlobalResponsePlanMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeResponse());

    const { result } = renderHook(() => useGlobalResponsePlan());

    await waitFor(() => expect(result.current.error).toBeTruthy());

    result.current.retry();

    await waitFor(() => expect(result.current.response).not.toBeNull());
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(2);
  });

  it("ignores a duplicate retry while a request is already in flight", async () => {
    let resolveFirst: (value: GlobalResponsePlanResponse) => void = () => {};
    getCurrentGlobalResponsePlanMock.mockReturnValueOnce(
      new Promise<GlobalResponsePlanResponse>((resolve) => {
        resolveFirst = resolve;
      }),
    );

    const { result } = renderHook(() => useGlobalResponsePlan());

    result.current.retry();
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);

    resolveFirst(makeResponse());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
  });
});
