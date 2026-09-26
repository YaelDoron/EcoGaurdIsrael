import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useEventDetails } from "./useEventDetails";
import { useEventLocationName } from "./useEventLocationName";
import { useEventLocationNames } from "./useEventLocationNames";
import { useResponsePlan } from "./useResponsePlan";

/**
 * Navigating between Dashboard / Event Details / Response Plan must not
 * leave slow reads running in the background (they would occupy the
 * browser's per-host connections and delay the next page), and a plan
 * page's decorative location name must not re-trigger a full Event Details
 * read on every visit.
 */
const { getEventDetailsMock, getCurrentResponsePlanMock, getResponsePlanByIdMock } = vi.hoisted(() => ({
  getEventDetailsMock: vi.fn(),
  getCurrentResponsePlanMock: vi.fn(),
  getResponsePlanByIdMock: vi.fn(),
}));

vi.mock("../api/eventDetails", () => ({ getEventDetails: getEventDetailsMock }));
vi.mock("../api/responsePlans", () => ({
  getCurrentResponsePlan: getCurrentResponsePlanMock,
  getResponsePlanById: getResponsePlanByIdMock,
}));

const pending = () => new Promise<never>(() => {});

function detailsWithNewsLocation(name: string) {
  return { detection_evidence: { satellite: [], news: [{ location_name: name }] } };
}

describe("navigation-scoped requests", () => {
  beforeEach(() => {
    getEventDetailsMock.mockReset();
    getCurrentResponsePlanMock.mockReset();
    getResponsePlanByIdMock.mockReset();
  });

  it("leaving a Response Plan page aborts its in-flight plan read", () => {
    getCurrentResponsePlanMock.mockReturnValue(pending());
    const { unmount } = renderHook(() => useResponsePlan({ kind: "current", fireEventId: 5 }));
    const signal = getCurrentResponsePlanMock.mock.calls[0][1] as AbortSignal;

    unmount();

    expect(signal.aborted).toBe(true);
  });

  it("leaving Event Details aborts its in-flight details read", () => {
    getEventDetailsMock.mockReturnValue(pending());
    const { unmount } = renderHook(() => useEventDetails(5));
    const signal = getEventDetailsMock.mock.calls[0][1] as AbortSignal;

    unmount();

    expect(signal.aborted).toBe(true);
  });

  it("a found event location name is reused on the next visit without another details read", async () => {
    getEventDetailsMock.mockResolvedValue(detailsWithNewsLocation("Judean Hills"));
    const first = renderHook(() => useEventLocationName(7));
    await waitFor(() => expect(first.result.current).toBe("Judean Hills"));
    first.unmount();

    const second = renderHook(() => useEventLocationName(7));
    const several = renderHook(() => useEventLocationNames([7]));

    expect(second.result.current).toBe("Judean Hills");
    expect(several.result.current[7]).toBe("Judean Hills");
    expect(getEventDetailsMock).toHaveBeenCalledTimes(1);
  });

  it("an event without a name yet is looked up again later (no negative caching)", async () => {
    getEventDetailsMock.mockResolvedValueOnce({ detection_evidence: { satellite: [], news: [] } });
    const first = renderHook(() => useEventLocationName(8));
    await waitFor(() => expect(getEventDetailsMock).toHaveBeenCalledTimes(1));
    first.unmount();

    getEventDetailsMock.mockResolvedValueOnce(detailsWithNewsLocation("Galilee"));
    const second = renderHook(() => useEventLocationName(8));

    await waitFor(() => expect(second.result.current).toBe("Galilee"));
    expect(getEventDetailsMock).toHaveBeenCalledTimes(2);
  });

  it("leaving the Global Response Plan aborts its pending location-name lookups", () => {
    getEventDetailsMock.mockReturnValue(pending());
    const { unmount } = renderHook(() => useEventLocationNames([1, 2]));
    const signals = getEventDetailsMock.mock.calls.map((call) => call[1] as AbortSignal);

    unmount();

    expect(signals).toHaveLength(2);
    expect(signals.every((signal) => signal.aborted)).toBe(true);
  });
});
