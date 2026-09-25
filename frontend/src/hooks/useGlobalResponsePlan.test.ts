import { act, renderHook, waitFor } from "@testing-library/react";
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

describe("useGlobalResponsePlan polling", () => {
  async function flush(ms = 0) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  function planWith(completedAt: string, runId = 77): GlobalResponsePlanResponse {
    const base = makeResponse();
    return { ...base, plan: { ...base.plan!, run_id: runId, completed_at: completedAt } };
  }

  beforeEach(() => {
    getCurrentGlobalResponsePlanMock.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("re-fetches every 5 seconds and picks up a newly materialized plan", async () => {
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(planWith("2026-09-20T10:03:00Z", 1))
      .mockResolvedValueOnce(planWith("2026-09-20T10:07:00Z", 2));

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    expect(result.current.response?.plan?.run_id).toBe(1);

    await flush(4999);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);

    await flush(1);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(2);
    expect(result.current.response?.plan?.run_id).toBe(2);
  });

  it("keeps polling indefinitely, including when there is no plan yet (plan: null)", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse({ plan: null }));

    renderHook(() => useGlobalResponsePlan());
    await flush();
    await flush(5000);
    await flush(5000);

    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(3);
  });

  it("does not report isLoading during a poll - only isRefreshing - and keeps the response visible", async () => {
    let resolvePoll: (value: GlobalResponsePlanResponse) => void = () => {};
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(makeResponse())
      .mockReturnValueOnce(new Promise<GlobalResponsePlanResponse>((resolve) => (resolvePoll = resolve)));

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    await flush(5000);

    expect(result.current.isLoading).toBe(false);
    expect(result.current.isRefreshing).toBe(true);
    expect(result.current.response).not.toBeNull();

    resolvePoll(makeResponse());
    await flush();
    expect(result.current.isRefreshing).toBe(false);
  });

  it("keeps the previous response and keeps polling through a failed poll", async () => {
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(planWith("2026-09-20T10:03:00Z", 1))
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(planWith("2026-09-20T10:07:00Z", 2));

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    await flush(5000);

    expect(result.current.error).toBeNull();
    expect(result.current.refreshError).toMatch(/unable to refresh/i);
    expect(result.current.response?.plan?.run_id).toBe(1);

    await flush(5000);
    expect(result.current.refreshError).toBeNull();
    expect(result.current.response?.plan?.run_id).toBe(2);
  });

  it("does not poll after a failed first load, and retry() starts a fresh load that then polls", async () => {
    getCurrentGlobalResponsePlanMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValue(makeResponse());

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    await flush(30000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
    expect(result.current.error).toBeTruthy();

    act(() => {
      result.current.retry();
    });
    await flush();
    expect(result.current.error).toBeNull();
    expect(result.current.response).not.toBeNull();

    await flush(5000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(3);
  });

  it("never has two requests in flight: the next poll is scheduled after the current one settles", async () => {
    let resolveSlow: (value: GlobalResponsePlanResponse) => void = () => {};
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(makeResponse())
      .mockReturnValueOnce(new Promise<GlobalResponsePlanResponse>((resolve) => (resolveSlow = resolve)))
      .mockResolvedValue(makeResponse());

    renderHook(() => useGlobalResponsePlan());
    await flush();
    await flush(5000);
    await flush(60000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(2);

    resolveSlow(makeResponse());
    await flush(5000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(3);
  });

  it("keeps the same plan object when a poll returns an identical plan, but still updates as_of", async () => {
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(makeResponse({ as_of: "2026-09-20T09:00:00Z" }))
      .mockResolvedValueOnce(makeResponse({ as_of: "2026-09-20T09:00:05Z" }));

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    const firstPlan = result.current.response?.plan;

    await flush(5000);

    expect(result.current.response?.as_of).toBe("2026-09-20T09:00:05Z");
    expect(result.current.response?.plan).toBe(firstPlan);
  });

  it("uses a new plan object when the plan content changed", async () => {
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(planWith("2026-09-20T10:03:00Z", 1))
      .mockResolvedValueOnce(planWith("2026-09-20T10:07:00Z", 2));

    const { result } = renderHook(() => useGlobalResponsePlan());
    await flush();
    const firstPlan = result.current.response?.plan;

    await flush(5000);

    expect(result.current.response?.plan).not.toBe(firstPlan);
  });

  it("clears the poll timer on unmount", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    const { unmount } = renderHook(() => useGlobalResponsePlan());
    await flush();
    expect(vi.getTimerCount()).toBe(1);

    unmount();

    expect(vi.getTimerCount()).toBe(0);
    await flush(30000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
  });

  it("does not update state or reschedule when a request resolves after unmount", async () => {
    let resolveInFlight: (value: GlobalResponsePlanResponse) => void = () => {};
    getCurrentGlobalResponsePlanMock.mockReturnValueOnce(
      new Promise<GlobalResponsePlanResponse>((resolve) => (resolveInFlight = resolve)),
    );

    const { unmount } = renderHook(() => useGlobalResponsePlan());
    unmount();
    resolveInFlight(makeResponse());
    await flush();

    expect(vi.getTimerCount()).toBe(0);
    await flush(30000);
    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
  });
});
