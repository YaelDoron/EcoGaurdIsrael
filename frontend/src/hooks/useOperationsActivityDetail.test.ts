import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { OperationsActivityDetailResponse } from "../types/operationsActivity";
import { useOperationsActivityDetail } from "./useOperationsActivityDetail";

const { getOperationsActivityDetailMock } = vi.hoisted(() => ({
  getOperationsActivityDetailMock: vi.fn(),
}));

vi.mock("../api/operations", () => ({
  getOperationsActivityDetail: getOperationsActivityDetailMock,
}));

function makeDetail(overrides: Partial<OperationsActivityDetailResponse> = {}): OperationsActivityDetailResponse {
  return {
    activity_type: "fire_event",
    entity_id: 42,
    occurred_at: "2026-09-20T11:00:00Z",
    title: "Fire Event #42",
    location: { latitude: 32.7, longitude: 35.0 },
    details: {
      fire_event_id: 42,
      status: "confirmed",
      detection_confidence: 0.84,
      detected_at: "2026-09-20T11:00:00Z",
      updated_at: "2026-09-20T11:05:00Z",
      latitude: 32.7,
      longitude: 35.0,
      methodology: "ECOGUARD_MULTI_SOURCE_DETECTION",
      methodology_version: "1.0",
      evidence: { satellite_hotspot_ids: [], news_report_ids: [] },
      latest_severity: null,
    },
    ...overrides,
  } as OperationsActivityDetailResponse;
}

describe("useOperationsActivityDetail", () => {
  beforeEach(() => {
    getOperationsActivityDetailMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("makes no request when nothing is selected", () => {
    const { result } = renderHook(() => useOperationsActivityDetail({ activityType: null, entityId: null }));

    expect(getOperationsActivityDetailMock).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
  });

  it("requests the detail when an activity is selected", async () => {
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail());

    const { result } = renderHook(() => useOperationsActivityDetail({ activityType: "fire_event", entityId: 42 }));

    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("fire_event", 42, expect.any(AbortSignal));
    expect(result.current.data).toEqual(makeDetail());
  });

  it("builds a valid request for each of the six activity types", async () => {
    const types: Array<OperationsActivityDetailResponse["activity_type"]> = [
      "fire_danger",
      "satellite_hotspot",
      "news_report",
      "fire_event",
      "fire_severity",
      "global_planning_run",
    ];

    for (const activityType of types) {
      getOperationsActivityDetailMock.mockReset();
      getOperationsActivityDetailMock.mockResolvedValue(makeDetail({ activity_type: activityType } as never));

      const { result, unmount } = renderHook(() =>
        useOperationsActivityDetail({ activityType, entityId: 1 }),
      );
      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(getOperationsActivityDetailMock).toHaveBeenCalledWith(activityType, 1, expect.any(AbortSignal));
      unmount();
    }
  });

  it("loads new detail when the selection changes", async () => {
    getOperationsActivityDetailMock
      .mockResolvedValueOnce(makeDetail({ entity_id: 42 }))
      .mockResolvedValueOnce(makeDetail({ entity_id: 99 }));

    const { result, rerender } = renderHook(
      ({ entityId }: { entityId: number }) =>
        useOperationsActivityDetail({ activityType: "fire_event", entityId }),
      { initialProps: { entityId: 42 } },
    );
    await waitFor(() => expect(result.current.data?.entity_id).toBe(42));

    rerender({ entityId: 99 });

    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.data?.entity_id).toBe(99));
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(2);
  });

  it("exposes notFound cleanly for a missing activity (404), without a generic error", async () => {
    getOperationsActivityDetailMock.mockRejectedValue(new ApiError("not found", 404));

    const { result } = renderHook(() => useOperationsActivityDetail({ activityType: "fire_event", entityId: 999 }));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.notFound).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("does not fetch every feed item automatically - only the single selected one", async () => {
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail());

    renderHook(() => useOperationsActivityDetail({ activityType: "fire_event", entityId: 42 }));
    await vi.waitFor(() => expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1));

    // Simulate time passing with no selection change - still exactly one request.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1);
  });

  it("clearing the selection resets to idle without fetching again", async () => {
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail());

    const initialProps: { activityType: "fire_event" | null; entityId: number | null } = {
      activityType: "fire_event",
      entityId: 42,
    };
    const { result, rerender } = renderHook(
      (props: { activityType: "fire_event" | null; entityId: number | null }) =>
        useOperationsActivityDetail(props),
      { initialProps },
    );
    await waitFor(() => expect(result.current.data).not.toBeNull());

    rerender({ activityType: null, entityId: null });

    expect(result.current.data).toBeNull();
    expect(result.current.isLoading).toBe(false);
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1);
  });
});
