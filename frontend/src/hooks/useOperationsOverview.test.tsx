import { render, renderHook, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import { OPERATIONS_OVERVIEW_POLL_INTERVAL_MS } from "../config/polling";
import type { OperationsOverviewResponse } from "../types/operationsOverview";
import { useOperationsOverview } from "./useOperationsOverview";

const { getOperationsOverviewMock } = vi.hoisted(() => ({
  getOperationsOverviewMock: vi.fn(),
}));

vi.mock("../api/operations", () => ({
  getOperationsOverview: getOperationsOverviewMock,
}));

function makeOverview(overrides: Partial<OperationsOverviewResponse> = {}): OperationsOverviewResponse {
  return {
    generated_at: "2026-09-20T12:00:00Z",
    simulation: { enabled: false, run: null },
    fire_danger_areas: [],
    active_fires: [],
    activity_feed: { items: [], limit: 30 },
    ...overrides,
  };
}

describe("useOperationsOverview", () => {
  beforeEach(() => {
    getOperationsOverviewMock.mockReset();
    // `shouldAdvanceTime` lets real wall-clock time keep the fake clock
    // ticking forward too, so Testing Library's own `waitFor` polling (which
    // schedules its own setTimeout checks) is not frozen solid - the hook's
    // 5-second poll interval is still fast-forwarded explicitly via
    // `vi.advanceTimersByTimeAsync` rather than waiting in real time.
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("starts in a loading state and stores data on the initial successful load", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    const { result } = renderHook(() => useOperationsOverview());

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeNull();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toEqual(makeOverview());
    expect(result.current.loadError).toBeNull();
  });

  it("treats empty fire_danger_areas/active_fires/activity_feed.items as a valid, non-error state", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.loadError).toBeNull();
    expect(result.current.data?.fire_danger_areas).toEqual([]);
    expect(result.current.data?.active_fires).toEqual([]);
    expect(result.current.data?.activity_feed.items).toEqual([]);
  });

  it("polls again after the configured interval elapses", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    renderHook(() => useOperationsOverview());
    await waitFor(() => expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS);

    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(2);
  });

  it("does not poll again before the configured interval elapses", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    renderHook(() => useOperationsOverview());
    await waitFor(() => expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1));

    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS - 500);

    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1);
  });

  it("keeps previous data visible and marks isRefreshing during a background poll", async () => {
    getOperationsOverviewMock.mockResolvedValueOnce(makeOverview({ generated_at: "2026-09-20T12:00:00Z" }));
    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data?.generated_at).toBe("2026-09-20T12:00:00Z");

    let resolveSecond!: (value: OperationsOverviewResponse) => void;
    getOperationsOverviewMock.mockReturnValueOnce(
      new Promise<OperationsOverviewResponse>((resolve) => {
        resolveSecond = resolve;
      }),
    );

    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS);
    await waitFor(() => expect(result.current.isRefreshing).toBe(true));

    // Previous data must still be visible while the background poll is in flight.
    expect(result.current.data?.generated_at).toBe("2026-09-20T12:00:00Z");
    expect(result.current.isLoading).toBe(false);

    resolveSecond(makeOverview({ generated_at: "2026-09-20T12:00:05Z" }));
    await waitFor(() => expect(result.current.data?.generated_at).toBe("2026-09-20T12:00:05Z"));
    expect(result.current.isRefreshing).toBe(false);
  });

  it("does not wipe last good data on a transient background poll failure, and continues polling", async () => {
    getOperationsOverviewMock
      .mockResolvedValueOnce(makeOverview({ generated_at: "2026-09-20T12:00:00Z" }))
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeOverview({ generated_at: "2026-09-20T12:00:10Z" }));

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS);
    await waitFor(() => expect(result.current.refreshError).not.toBeNull());
    expect(result.current.data?.generated_at).toBe("2026-09-20T12:00:00Z");
    expect(result.current.loadError).toBeNull();

    // Polling recovers automatically on the next scheduled tick.
    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS);
    await waitFor(() => expect(result.current.data?.generated_at).toBe("2026-09-20T12:00:10Z"));
    expect(result.current.refreshError).toBeNull();
  });

  it("stops polling and issues no further requests after unmount", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    const { unmount } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1));

    unmount();

    await vi.advanceTimersByTimeAsync(OPERATIONS_OVERVIEW_POLL_INTERVAL_MS * 3);

    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1);
  });

  it("never issues a second overlapping request while one is already in flight", async () => {
    let resolveFirst!: (value: OperationsOverviewResponse) => void;
    getOperationsOverviewMock.mockReturnValueOnce(
      new Promise<OperationsOverviewResponse>((resolve) => {
        resolveFirst = resolve;
      }),
    );

    const { result } = renderHook(() => useOperationsOverview());
    expect(result.current.isLoading).toBe(true);

    // A manual refresh while the initial load is still in flight must not
    // start a second request.
    result.current.refresh();
    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1);

    resolveFirst(makeOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1);
  });

  it("treats simulation.enabled=false with run=null as a normal, error-free state", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview({ simulation: { enabled: false, run: null } }));

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.loadError).toBeNull();
    expect(result.current.data?.simulation).toEqual({ enabled: false, run: null });
  });

  it("treats simulation.enabled=true with run=null as a normal, error-free state", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview({ simulation: { enabled: true, run: null } }));

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.loadError).toBeNull();
    expect(result.current.data?.simulation).toEqual({ enabled: true, run: null });
  });

  it("keeps a completed simulation run visible in the returned data", async () => {
    const completedRun = {
      run_id: "run-1",
      state: "completed" as const,
      preset_id: "operations_demo",
      seed: 42,
      mode: "automatic",
      simulation_duration_seconds: 120,
      events_total: 18,
      events_completed: 18,
      events_succeeded: 18,
      events_failed: 0,
      current_event: null,
      started_at: "2026-09-20T11:55:00Z",
      completed_at: "2026-09-20T12:00:00Z",
      wall_clock_elapsed_seconds: 300,
      last_message: "Simulation run completed.",
      error: null,
    };
    getOperationsOverviewMock.mockResolvedValue(makeOverview({ simulation: { enabled: true, run: completedRun } }));

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data?.simulation.run?.state).toBe("completed");
    expect(result.current.data?.simulation.run?.events_completed).toBe(18);
  });

  it("preserves generated_at from the response unchanged", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview({ generated_at: "2026-09-20T12:34:56.789Z" }));

    const { result } = renderHook(() => useOperationsOverview());
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data?.generated_at).toBe("2026-09-20T12:34:56.789Z");
  });

  it("still resolves out of the loading state under React StrictMode's mount->cleanup->remount cycle", async () => {
    // Regression test: `main.tsx` renders the whole app inside <StrictMode>,
    // which in development mounts every component, runs its cleanup, then
    // remounts it - all synchronously, before the first request settles.
    // A real browser session against `npm run dev` was observed stuck on
    // "Loading operations overview..." forever because of this.
    //
    // RTL's `renderHook` does NOT reproduce StrictMode's double-effect-
    // invocation (verified empirically - an effect inside a `renderHook`
    // wrapped in <StrictMode> only ever fires once), which is exactly why
    // this bug shipped past the rest of this file. A plain `render()` of a
    // real component DOES reproduce it, so this test renders one directly.
    getOperationsOverviewMock.mockResolvedValue(makeOverview());

    let latest: ReturnType<typeof useOperationsOverview> | undefined;
    function Probe() {
      latest = useOperationsOverview();
      return null;
    }

    render(
      <StrictMode>
        <Probe />
      </StrictMode>,
    );

    await waitFor(() => expect(latest?.isLoading).toBe(false), { timeout: 2000 });
    expect(latest?.data).toEqual(makeOverview());
    expect(latest?.loadError).toBeNull();
  });
});
