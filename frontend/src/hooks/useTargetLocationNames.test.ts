import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useTargetLocationNames } from "./useTargetLocationNames";

const { reverseGeocodeMock } = vi.hoisted(() => ({ reverseGeocodeMock: vi.fn() }));

vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));

describe("useTargetLocationNames", () => {
  beforeEach(() => {
    reverseGeocodeMock.mockReset();
  });

  it("returns an empty map until a lookup resolves", () => {
    reverseGeocodeMock.mockReturnValue(new Promise<never>(() => {}));

    const { result } = renderHook(() => useTargetLocationNames([{ id: 1, latitude: 32.7, longitude: 35.0 }]));

    expect(result.current).toEqual({});
  });

  it("maps each target id to its resolved place name", async () => {
    reverseGeocodeMock.mockImplementation((lat: number) => Promise.resolve(lat > 33 ? "Safed" : "Haifa"));

    const { result } = renderHook(() =>
      useTargetLocationNames([
        { id: 1, latitude: 32.7, longitude: 35.0 },
        { id: 2, latitude: 33.5, longitude: 35.5 },
      ]),
    );

    await waitFor(() => expect(result.current).toEqual({ 1: "Haifa", 2: "Safed" }));
    expect(reverseGeocodeMock).toHaveBeenCalledWith(32.7, 35.0);
  });

  it("records null for a failed lookup so callers can fall back", async () => {
    reverseGeocodeMock.mockResolvedValue(null);

    const { result } = renderHook(() => useTargetLocationNames([{ id: 5, latitude: 1, longitude: 2 }]));

    await waitFor(() => expect(result.current).toEqual({ 5: null }));
  });

  it("does not re-request when re-rendered with the same targets", async () => {
    reverseGeocodeMock.mockResolvedValue("Haifa");
    const targets = [{ id: 1, latitude: 32.7, longitude: 35.0 }];

    const { result, rerender } = renderHook(({ t }) => useTargetLocationNames(t), { initialProps: { t: targets } });
    await waitFor(() => expect(result.current).toEqual({ 1: "Haifa" }));
    rerender({ t: [{ id: 1, latitude: 32.7, longitude: 35.0 }] });

    expect(reverseGeocodeMock).toHaveBeenCalledTimes(1);
  });
});
