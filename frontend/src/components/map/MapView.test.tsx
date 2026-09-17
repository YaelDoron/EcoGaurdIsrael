import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MapView } from "./MapView";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

describe("MapView", () => {
  it("renders a labeled map region containing a tile layer", () => {
    render(<MapView boundsPoints={[{ lat: 32.7, lng: 35.0 }]} ariaLabel="Map of Event #1" />);

    expect(screen.getByRole("region", { name: "Map of Event #1" })).toBeInTheDocument();
    expect(screen.getByTestId("tile-layer")).toBeInTheDocument();
  });

  it("renders its children inside the map", () => {
    render(
      <MapView boundsPoints={[{ lat: 32.7, lng: 35.0 }]}>
        <div data-testid="custom-layer">layer content</div>
      </MapView>,
    );

    expect(screen.getByTestId("custom-layer")).toBeInTheDocument();
  });

  it("renders without crashing when there are no bounds points yet", () => {
    render(<MapView boundsPoints={[]} />);

    expect(screen.getByTestId("map-container")).toBeInTheDocument();
  });
});
