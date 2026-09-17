import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ResponsePlanMapLayer } from "./ResponsePlanMapLayer";
import type { ResponseRouteLayerData } from "./ResponseRouteLayerModel";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeLayer(overrides: Partial<ResponseRouteLayerData> = {}): ResponseRouteLayerData {
  return {
    routes: [
      {
        actionKey: "engine-1",
        resourceId: "engine-1",
        targetId: 1,
        path: [
          { latitude: 32.0, longitude: 35.0 },
          { latitude: 32.5, longitude: 35.5 },
        ],
        isSelected: false,
      },
    ],
    originMarkers: [
      { actionKey: "engine-1", resourceId: "engine-1", coordinate: { latitude: 32.0, longitude: 35.0 }, isSelected: false },
    ],
    targetMarkers: [
      { actionKey: "engine-1", targetId: 1, coordinate: { latitude: 32.5, longitude: 35.5 }, isSelected: false },
    ],
    ...overrides,
  };
}

describe("ResponsePlanMapLayer", () => {
  it("renders nothing when the layer has no routes or markers", () => {
    const { container } = render(
      <ResponsePlanMapLayer layer={{ routes: [], originMarkers: [], targetMarkers: [] }} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("draws a polyline for each route with its exact persisted path, in order", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    const polyline = screen.getByTestId("polyline");
    expect(JSON.parse(polyline.getAttribute("data-positions")!)).toEqual([
      [32.0, 35.0],
      [32.5, 35.5],
    ]);
  });

  it("gives a selected route a different color/weight than an unselected one", () => {
    const { rerender } = render(<ResponsePlanMapLayer layer={makeLayer()} />);
    const unselectedColor = screen.getByTestId("polyline").getAttribute("data-color");
    const unselectedWeight = screen.getByTestId("polyline").getAttribute("data-weight");

    rerender(
      <ResponsePlanMapLayer
        layer={makeLayer({ routes: [{ ...makeLayer().routes[0], isSelected: true }] })}
      />,
    );

    const polyline = screen.getByTestId("polyline");
    expect(polyline.getAttribute("data-color")).not.toBe(unselectedColor);
    expect(polyline.getAttribute("data-weight")).not.toBe(unselectedWeight);
  });

  it("renders an origin marker at its exact persisted coordinate", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    const markers = screen.getAllByTestId("marker");
    const origin = markers.find((marker) => marker.getAttribute("data-lat") === "32");
    expect(origin).toHaveAttribute("data-lng", "35");
  });

  it("renders a target marker at its exact persisted coordinate", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    const markers = screen.getAllByTestId("marker");
    const target = markers.find((marker) => marker.getAttribute("data-lat") === "32.5");
    expect(target).toHaveAttribute("data-lng", "35.5");
  });

  it("renders only the markers it is given when a route has no origin marker", () => {
    render(<ResponsePlanMapLayer layer={makeLayer({ originMarkers: [] })} />);

    expect(screen.getAllByTestId("marker")).toHaveLength(1);
  });

  it("renders only the markers it is given when a route has no target marker", () => {
    render(<ResponsePlanMapLayer layer={makeLayer({ targetMarkers: [] })} />);

    expect(screen.getAllByTestId("marker")).toHaveLength(1);
  });

  it("draws no route line when the layer has no drawable routes", () => {
    render(<ResponsePlanMapLayer layer={makeLayer({ routes: [] })} />);

    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });
});
