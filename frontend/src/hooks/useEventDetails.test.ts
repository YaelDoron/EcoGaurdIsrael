import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { EventDetailsResult } from "../types/eventDetails";
import { useEventDetails } from "./useEventDetails";

const { getEventDetailsMock } = vi.hoisted(() => ({
  getEventDetailsMock: vi.fn(),
}));

vi.mock("../api/eventDetails", () => ({
  getEventDetails: getEventDetailsMock,
}));

function makeResult(overrides: Partial<EventDetailsResult> = {}): EventDetailsResult {
  return {
    as_of: "2026-09-17T14:00:00Z",
    fire_event: {
      fire_event_id: 12,
      status: "confirmed",
      latitude: 32.731,
      longitude: 35.046,
      detection_confidence: 0.91,
      detected_at: "2026-09-17T13:20:00Z",
      updated_at: "2026-09-17T13:28:00Z",
      methodology: "detector",
      methodology_version: "1.0",
    },
    severity: null,
    ml_assessment: null,
    danger: null,
    detection_evidence: { satellite: [], news: [] },
    spread_predictions: [],
    targets: [],
    stations: [],
    resources: [],
    station_summaries: [],
    current_response_plan: null,
    ...overrides,
  };
}

describe("useEventDetails", () => {
  beforeEach(() => {
    getEventDetailsMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("starts in a loading state and fetches with the given fireEventId", async () => {
    getEventDetailsMock.mockReturnValue(new Promise<never>(() => {}));

    const { result } = renderHook(() => useEventDetails(12));

    expect(result.current.isLoading).toBe(true);
    expect(result.current.data).toBeNull();
    await waitFor(() => expect(getEventDetailsMock).toHaveBeenCalledWith(12, expect.any(AbortSignal)));
  });

  it("stores the data and clears loading on success", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    const { result } = renderHook(() => useEventDetails(12));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.data).toEqual(makeResult());
    expect(result.current.loadError).toBeNull();
    expect(result.current.notFound).toBe(false);
  });

  it("sets notFound (not loadError) on a 404, with no data", async () => {
    getEventDetailsMock.mockRejectedValue(new ApiError("not found", 404));

    const { result } = renderHook(() => useEventDetails(999));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.notFound).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.loadError).toBeNull();
  });

  it("sets loadError (not notFound) on a non-404 failure", async () => {
    getEventDetailsMock.mockRejectedValue(new ApiError("boom", 500));

    const { result } = renderHook(() => useEventDetails(12));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.loadError).toMatch(/unable to load/i);
    expect(result.current.notFound).toBe(false);
    expect(result.current.data).toBeNull();
  });

  it("keeps existing data visible when a refresh fails, surfacing refreshError instead", async () => {
    getEventDetailsMock.mockResolvedValueOnce(makeResult()).mockRejectedValueOnce(new ApiError("boom", 500));

    const { result } = renderHook(() => useEventDetails(12));
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.isRefreshing).toBe(false));
    expect(result.current.refreshError).toMatch(/unable to refresh/i);
    expect(result.current.data).toEqual(makeResult());
  });

  it("updates data after a successful refresh", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeResult({ as_of: "2026-09-17T14:00:00Z" }))
      .mockResolvedValueOnce(makeResult({ as_of: "2026-09-17T15:00:00Z" }));

    const { result } = renderHook(() => useEventDetails(12));
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.data?.as_of).toBe("2026-09-17T15:00:00Z"));
  });

  it("re-fetches from scratch when fireEventId changes", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeResult({ fire_event: { ...makeResult().fire_event, fire_event_id: 12 } }))
      .mockResolvedValueOnce(makeResult({ fire_event: { ...makeResult().fire_event, fire_event_id: 34 } }));

    const { result, rerender } = renderHook(({ id }) => useEventDetails(id), { initialProps: { id: 12 } });
    await waitFor(() => expect(result.current.data?.fire_event.fire_event_id).toBe(12));

    rerender({ id: 34 });

    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.data?.fire_event.fire_event_id).toBe(34));
    expect(getEventDetailsMock).toHaveBeenNthCalledWith(2, 34, expect.any(AbortSignal));
  });
});

describe("useEventDetails polling", () => {
  function makeWithStatus(status: EventDetailsResult["fire_event"]["status"], asOf = "2026-09-17T14:00:00Z") {
    return makeResult({ as_of: asOf, fire_event: { ...makeResult().fire_event, status } });
  }

  async function flush(ms = 0) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  beforeEach(() => {
    getEventDetailsMock.mockReset();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("re-fetches every 5 seconds while the event is active and picks up a status change", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeWithStatus("suspected"))
      .mockResolvedValueOnce(makeWithStatus("confirmed", "2026-09-17T14:00:05Z"));

    const { result } = renderHook(() => useEventDetails(12));
    await flush();
    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
    expect(result.current.data?.fire_event.status).toBe("suspected");

    await flush(4999);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);

    await flush(1);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(2);
    expect(result.current.data?.fire_event.status).toBe("confirmed");
  });

  it("keeps polling a confirmed event", async () => {
    getEventDetailsMock.mockResolvedValue(makeWithStatus("confirmed"));

    renderHook(() => useEventDetails(12));
    await flush();
    await flush(5000);
    await flush(5000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(3);
  });

  it.each(["resolved", "dismissed"] as const)("does not poll a %s event", async (status) => {
    getEventDetailsMock.mockResolvedValue(makeWithStatus(status));

    renderHook(() => useEventDetails(12));
    await flush();
    await flush(30000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
  });

  it("stops polling once the event becomes resolved", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeWithStatus("confirmed"))
      .mockResolvedValue(makeWithStatus("resolved"));

    renderHook(() => useEventDetails(12));
    await flush();
    await flush(5000);
    await flush(30000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(2);
  });

  it("does not poll after a 404", async () => {
    getEventDetailsMock.mockRejectedValue(new ApiError("not found", 404));

    renderHook(() => useEventDetails(999));
    await flush();
    await flush(30000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
  });

  it("keeps polling through a failed poll and keeps the previous data visible", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeWithStatus("confirmed"))
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeWithStatus("confirmed", "2026-09-17T14:00:10Z"));

    const { result } = renderHook(() => useEventDetails(12));
    await flush();
    await flush(5000);
    expect(result.current.refreshError).toMatch(/unable to refresh/i);
    expect(result.current.data?.as_of).toBe("2026-09-17T14:00:00Z");

    await flush(5000);
    expect(result.current.refreshError).toBeNull();
    expect(result.current.data?.as_of).toBe("2026-09-17T14:00:10Z");
  });

  it("never has two requests in flight: the next poll is scheduled after the current one settles", async () => {
    let resolveSlow: (value: EventDetailsResult) => void = () => {};
    getEventDetailsMock
      .mockResolvedValueOnce(makeWithStatus("confirmed"))
      .mockReturnValueOnce(new Promise<EventDetailsResult>((resolve) => (resolveSlow = resolve)))
      .mockResolvedValue(makeWithStatus("confirmed"));

    renderHook(() => useEventDetails(12));
    await flush();
    await flush(5000); // second request starts and stays pending
    await flush(60000);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(2);

    resolveSlow(makeWithStatus("confirmed"));
    await flush(5000);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(3);
  });

  it("clears the poll timer on unmount", async () => {
    getEventDetailsMock.mockResolvedValue(makeWithStatus("confirmed"));

    const { unmount } = renderHook(() => useEventDetails(12));
    await flush();
    expect(vi.getTimerCount()).toBe(1);

    unmount();

    expect(vi.getTimerCount()).toBe(0);
    await flush(30000);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
  });

  it("does not update state or reschedule when a request resolves after unmount", async () => {
    let resolveInFlight: (value: EventDetailsResult) => void = () => {};
    getEventDetailsMock.mockReturnValueOnce(new Promise<EventDetailsResult>((resolve) => (resolveInFlight = resolve)));

    const { unmount } = renderHook(() => useEventDetails(12));
    unmount();
    resolveInFlight(makeWithStatus("confirmed"));
    await flush();

    expect(vi.getTimerCount()).toBe(0);
    await flush(30000);
    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
  });

  it("switching to another event cancels the old poll and polls the new one", async () => {
    getEventDetailsMock.mockResolvedValue(makeWithStatus("confirmed"));

    const { rerender } = renderHook(({ id }) => useEventDetails(id), { initialProps: { id: 12 } });
    await flush();

    rerender({ id: 34 });
    await flush();
    await flush(5000);

    const ids = getEventDetailsMock.mock.calls.map((call) => call[0]);
    expect(ids).toEqual([12, 34, 34]);
  });
});
