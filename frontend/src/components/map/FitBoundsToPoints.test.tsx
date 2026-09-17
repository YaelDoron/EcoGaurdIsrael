import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FitBoundsToPoints } from "./FitBoundsToPoints";
import { createFakeMap, MapInstanceContext } from "../../test/reactLeafletStub";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function renderWithFakeMap(points: { lat: number; lng: number }[], map = createFakeMap()) {
  render(
    <MapInstanceContext.Provider value={map}>
      <FitBoundsToPoints points={points} />
    </MapInstanceContext.Provider>,
  );
  return map;
}

describe("FitBoundsToPoints", () => {
  it("does nothing when there are no points", () => {
    const setView = vi.fn();
    const fitBounds = vi.fn();
    renderWithFakeMap([], createFakeMap({ setView, fitBounds }));

    expect(setView).not.toHaveBeenCalled();
    expect(fitBounds).not.toHaveBeenCalled();
  });

  it("centers on the single point at the configured zoom when there is exactly one", () => {
    const setView = vi.fn();
    renderWithFakeMap([{ lat: 32.7, lng: 35.0 }], createFakeMap({ setView }));

    expect(setView).toHaveBeenCalledWith([32.7, 35.0], 13);
  });

  it("respects a custom singlePointZoom", () => {
    const setView = vi.fn();
    const map = createFakeMap({ setView });
    render(
      <MapInstanceContext.Provider value={map}>
        <FitBoundsToPoints points={[{ lat: 1, lng: 2 }]} singlePointZoom={9} />
      </MapInstanceContext.Provider>,
    );

    expect(setView).toHaveBeenCalledWith([1, 2], 9);
  });

  it("fits bounds spanning every point when there are multiple", () => {
    const fitBounds = vi.fn();
    renderWithFakeMap(
      [
        { lat: 32.0, lng: 34.5 },
        { lat: 33.0, lng: 35.5 },
        { lat: 32.5, lng: 35.0 },
      ],
      createFakeMap({ fitBounds }),
    );

    expect(fitBounds).toHaveBeenCalledTimes(1);
    const [bounds, options] = fitBounds.mock.calls[0];
    expect(bounds.getSouthWest().lat).toBeCloseTo(32.0);
    expect(bounds.getSouthWest().lng).toBeCloseTo(34.5);
    expect(bounds.getNorthEast().lat).toBeCloseTo(33.0);
    expect(bounds.getNorthEast().lng).toBeCloseTo(35.5);
    expect(options).toEqual({ padding: [48, 48] });
  });

  it("re-fits when the points prop changes", () => {
    const setView = vi.fn();
    const map = createFakeMap({ setView });
    const { rerender } = render(
      <MapInstanceContext.Provider value={map}>
        <FitBoundsToPoints points={[{ lat: 1, lng: 1 }]} />
      </MapInstanceContext.Provider>,
    );
    expect(setView).toHaveBeenCalledWith([1, 1], 13);

    rerender(
      <MapInstanceContext.Provider value={map}>
        <FitBoundsToPoints points={[{ lat: 2, lng: 2 }]} />
      </MapInstanceContext.Provider>,
    );

    expect(setView).toHaveBeenCalledWith([2, 2], 13);
  });
});
