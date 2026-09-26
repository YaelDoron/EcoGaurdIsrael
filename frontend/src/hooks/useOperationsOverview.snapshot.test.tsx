import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OperationsOverviewResponse } from "../types/operationsOverview";
import { MAX_SNAPSHOT_AGE_MS } from "./operationsOverviewSnapshot";
import { useOperationsOverview } from "./useOperationsOverview";

const { getOperationsOverviewMock } = vi.hoisted(() => ({ getOperationsOverviewMock: vi.fn() }));

vi.mock("../api/operations", () => ({ getOperationsOverview: getOperationsOverviewMock }));

function makeOverview(generatedAt: string): OperationsOverviewResponse {
  return {
    generated_at: generatedAt,
    simulation: { enabled: true, run: null },
    fire_danger_areas: [],
    active_fires: [],
    activity_feed: { items: [], limit: 30 },
  };
}

describe("useOperationsOverview - returning to the dashboard", () => {
  beforeEach(() => {
    getOperationsOverviewMock.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("renders the last overview immediately on remount and refreshes it in the background", async () => {
    getOperationsOverviewMock.mockResolvedValueOnce(makeOverview("2026-09-20T12:00:00Z"));
    const first = renderHook(() => useOperationsOverview({ pollingEnabled: false }));
    await waitFor(() => expect(first.result.current.data?.generated_at).toBe("2026-09-20T12:00:00Z"));
    first.unmount(); // e.g. navigating to a Response Plan page

    let resolveRefresh: (value: OperationsOverviewResponse) => void = () => {};
    getOperationsOverviewMock.mockReturnValueOnce(new Promise((resolve) => (resolveRefresh = resolve)));
    const second = renderHook(() => useOperationsOverview({ pollingEnabled: false })); // "Back to Active Fires"

    // No blank "Loading..." page: the previous snapshot is shown at once ...
    expect(second.result.current.isLoading).toBe(false);
    expect(second.result.current.data?.generated_at).toBe("2026-09-20T12:00:00Z");
    // ... while the latest state is fetched in the background and replaces it.
    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(2);
    expect(second.result.current.isRefreshing).toBe(true);
    resolveRefresh(makeOverview("2026-09-20T12:00:30Z"));
    await waitFor(() => expect(second.result.current.data?.generated_at).toBe("2026-09-20T12:00:30Z"));
  });

  it("never reuses a snapshot older than the staleness bound", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getOperationsOverviewMock.mockResolvedValueOnce(makeOverview("2026-09-20T12:00:00Z"));
    const first = renderHook(() => useOperationsOverview({ pollingEnabled: false }));
    await waitFor(() => expect(first.result.current.data).not.toBeNull());
    first.unmount();

    vi.setSystemTime(Date.now() + MAX_SNAPSHOT_AGE_MS + 1_000);
    getOperationsOverviewMock.mockReturnValueOnce(new Promise(() => {}));
    const second = renderHook(() => useOperationsOverview({ pollingEnabled: false }));

    expect(second.result.current.isLoading).toBe(true);
    expect(second.result.current.data).toBeNull();
  });
});
