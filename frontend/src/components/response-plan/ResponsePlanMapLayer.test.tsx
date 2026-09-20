import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ResponsePlanMapLayer, routePathOptions } from "./ResponsePlanMapLayer";
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
      { actionKey: "engine-1", resourceId: "engine-1", resourceIds: ["engine-1"], stationName: "Central Station", coordinate: { latitude: 32.0, longitude: 35.0 }, isSelected: false },
    ],
    targetMarkers: [
      { actionKey: "engine-1", targetId: 1, targetType: "active_fire", coordinate: { latitude: 32.5, longitude: 35.5 }, isSelected: false },
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

  it("draws the selected route thick, solid, and primary blue", () => {
    render(<ResponsePlanMapLayer layer={makeLayer({ routes: [{ ...makeLayer().routes[0], isSelected: true }] })} />);

    const polyline = screen.getByTestId("polyline");
    expect(polyline).toHaveAttribute("data-weight", "6");
    expect(polyline).toHaveAttribute("data-color", "var(--color-focus-ring)");
    expect(polyline).not.toHaveAttribute("data-dash");
  });

  it("draws non-selected routes thin, dashed, and grey while another route is selected", () => {
    const base = makeLayer().routes[0];
    render(
      <ResponsePlanMapLayer
        layer={makeLayer({
          routes: [
            { ...base, actionKey: "engine-1", isSelected: true },
            { ...base, actionKey: "engine-2", resourceId: "engine-2", isSelected: false },
          ],
        })}
      />,
    );

    const other = screen.getAllByTestId("polyline")[1];
    expect(other).toHaveAttribute("data-weight", "3");
    expect(other).toHaveAttribute("data-color", "var(--color-text-muted)");
    expect(other).toHaveAttribute("data-dash", "6 8");
  });

  it("draws every route the same, solid, when nothing is selected", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    const polyline = screen.getByTestId("polyline");
    expect(polyline).toHaveAttribute("data-color", "var(--color-focus-ring)");
    expect(polyline).not.toHaveAttribute("data-dash");
  });

  it("titles the origin popup with the English station name, with the resource id below it", () => {
    const layer = makeLayer();
    layer.originMarkers[0].stationName = "פארק הכרמל";
    const { container } = render(<ResponsePlanMapLayer layer={layer} />);

    const title = container.querySelector(".origin-popup__station") as HTMLElement;
    const resource = container.querySelector(".origin-popup__truck") as HTMLElement;
    expect(title).toHaveTextContent("Carmel Park");
    expect(resource).toHaveTextContent("engine-1");
    expect(title.compareDocumentPosition(resource) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(container.textContent).not.toMatch(/[֐-׿]/);
  });

  it("lists every truck from the origin as pills under an Allocated header", () => {
    const layer = makeLayer();
    layer.originMarkers[0].resourceIds = ["TRUCK-87-2", "TRUCK-88-1"];
    const { container } = render(<ResponsePlanMapLayer layer={layer} />);

    expect(screen.getByText("Allocated in current plan")).toBeInTheDocument();
    const pills = Array.from(container.querySelectorAll(".origin-popup__truck")).map((el) => el.textContent);
    expect(pills).toEqual(["TRUCK-87-2", "TRUCK-88-1"]);
  });

  it("shows a fiery fire glyph for the target marker", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    const fire = document.querySelector("[data-fire-icon]") as HTMLElement;
    expect(fire).not.toBeNull();
    expect(fire.style.color).toBe("rgb(255, 69, 0)");
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

  it("explicitly resets dashArray and opacity for a route that becomes selected (Leaflet keeps stale keys)", () => {
    const selected = routePathOptions(true, true);
    expect("dashArray" in selected).toBe(true);
    expect(selected.dashArray).toBeUndefined();
    expect(selected.opacity).toBe(1);

    const neutral = routePathOptions(false, false);
    expect("dashArray" in neutral).toBe(true);
    expect(neutral.opacity).toBe(1);
  });

  it("brings only the selected route to the front", () => {
    const base = makeLayer().routes[0];
    render(
      <ResponsePlanMapLayer
        layer={makeLayer({
          routes: [
            { ...base, actionKey: "engine-1", isSelected: false },
            { ...base, actionKey: "engine-2", resourceId: "engine-2", isSelected: true },
          ],
        })}
      />,
    );

    const [first, second] = screen.getAllByTestId("polyline");
    expect(first).not.toHaveAttribute("data-front");
    expect(second).toHaveAttribute("data-front", "true");
  });

  it("titles the target popup with its id and type, matching the list header", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} />);

    expect(screen.getByText("Target #1 - Active fire")).toBeInTheDocument();
  });

  it("titles the target popup with the region name when one is provided", () => {
    render(<ResponsePlanMapLayer layer={makeLayer()} locationName="Modiin" />);

    expect(screen.getByText("Active fire - Modiin")).toBeInTheDocument();
  });

  it("prefers the target's own reverse-geocoded place in the popup, then the event location", () => {
    const { rerender } = render(
      <ResponsePlanMapLayer layer={makeLayer()} locationName="Modiin" targetLocations={{ 1: "Haifa" }} />,
    );
    expect(screen.getByText("Active fire - Haifa")).toBeInTheDocument();

    rerender(<ResponsePlanMapLayer layer={makeLayer()} locationName="Modiin" targetLocations={{ 1: null }} />);
    expect(screen.getByText("Active fire - Modiin")).toBeInTheDocument();
  });
});
