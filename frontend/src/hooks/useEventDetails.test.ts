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
    danger: null,
    spread_predictions: [],
    targets: [],
    stations: [],
    resources: [],
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
    await waitFor(() => expect(getEventDetailsMock).toHaveBeenCalledWith(12));
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
    expect(getEventDetailsMock).toHaveBeenNthCalledWith(2, 34);
  });
});
