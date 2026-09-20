import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { GlobalResponseMapLayer } from "./GlobalResponseMapLayer";
import type { GlobalResponseMapLayerData } from "./GlobalResponseMapLayerModel";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeLayer(overrides: Partial<GlobalResponseMapLayerData> = {}): GlobalResponseMapLayerData {
  return {
    focusEventId: null,
    routes: [
      {
        actionKey: "101:engine-1",
        fireEventId: 101,
        resourceId: "engine-1",
        targetId: 1,
        path: [
          { latitude: 32.0, longitude: 35.0 },
          { latitude: 32.5, longitude: 35.5 },
        ],
        isFocused: false,
      },
    ],
    originMarkers: [
      {
        actionKey: "101:engine-1",
        fireEventId: 101,
        resourceId: "engine-1",
        coordinate: { latitude: 32.0, longitude: 35.0 },
        isFocused: false,
      },
    ],
    fireMarkers: [
      { key: "101-1", fireEventId: 101, targetId: 1, coordinate: { latitude: 32.5, longitude: 35.5 }, isFocused: false },
    ],
    targetMarkers: [
      {
        key: "101-2",
        fireEventId: 101,
        targetId: 2,
        isPredictedRisk: true,
        coordinate: { latitude: 32.6, longitude: 35.6 },
        isFocused: false,
      },
    ],
    stationMarkers: [
      {
        stationId: "station-1",
        stationName: "Central Station",
        coordinate: { latitude: 32.0, longitude: 35.0 },
        isFocused: false,
      },
    ],
    ...overrides,
  };
}

function emptyLayer(): GlobalResponseMapLayerData {
  return {
    focusEventId: null,
    routes: [],
    originMarkers: [],
    fireMarkers: [],
    targetMarkers: [],
    stationMarkers: [],
  };
}

describe("GlobalResponseMapLayer", () => {
  it("renders nothing when the layer has no routes or markers", () => {
    const { container } = render(<GlobalResponseMapLayer layer={emptyLayer()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("draws a polyline for each route with its exact persisted path, in order", () => {
    render(<GlobalResponseMapLayer layer={makeLayer()} />);

    const polyline = screen.getByTestId("polyline");
    expect(JSON.parse(polyline.getAttribute("data-positions")!)).toEqual([
      [32.0, 35.0],
      [32.5, 35.5],
    ]);
  });

  it("renders a marker for the origin, fire, target, and station", () => {
    render(<GlobalResponseMapLayer layer={makeLayer()} />);

    expect(screen.getAllByTestId("marker")).toHaveLength(4);
  });

  it("draws no route line when the layer has no drawable routes", () => {
    render(<GlobalResponseMapLayer layer={makeLayer({ routes: [] })} />);

    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });

  it("shows the fire event id and resource id in marker popups", () => {
    render(<GlobalResponseMapLayer layer={makeLayer()} />);

    expect(screen.getByText("Resource: engine-1")).toBeInTheDocument();
    expect(screen.getByText("Response target #2")).toBeInTheDocument();
    expect(screen.getAllByText("Event #101").length).toBeGreaterThan(0);
  });

  it("shows the English event name in popups instead of the raw event id", () => {
    render(<GlobalResponseMapLayer layer={makeLayer()} eventLabels={{ 101: "Haifa Subdistrict" }} />);

    expect(screen.getAllByText("Haifa Subdistrict").length).toBeGreaterThan(0);
    expect(screen.queryByText("Event #101")).not.toBeInTheDocument();
  });

  it("shows the station name in its popup", () => {
    render(<GlobalResponseMapLayer layer={makeLayer()} />);

    expect(screen.getByText("Central Station")).toBeInTheDocument();
  });

  it("falls back to the station id when no station name is available", () => {
    render(
      <GlobalResponseMapLayer
        layer={makeLayer({
          stationMarkers: [
            { stationId: "station-2", stationName: null, coordinate: { latitude: 33.0, longitude: 36.0 }, isFocused: false },
          ],
        })}
      />,
    );

    expect(screen.getByText("station-2")).toBeInTheDocument();
  });

  describe("focus styling", () => {
    it("gives a focused route a different color/weight than an unfocused one", () => {
      const unfocusedLayer = makeLayer({ focusEventId: 999, routes: [{ ...makeLayer().routes[0], isFocused: false }] });
      const { rerender } = render(<GlobalResponseMapLayer layer={unfocusedLayer} />);
      const unfocusedColor = screen.getByTestId("polyline").getAttribute("data-color");
      const unfocusedWeight = screen.getByTestId("polyline").getAttribute("data-weight");

      rerender(
        <GlobalResponseMapLayer
          layer={makeLayer({ focusEventId: 101, routes: [{ ...makeLayer().routes[0], isFocused: true }] })}
        />,
      );

      const polyline = screen.getByTestId("polyline");
      expect(polyline.getAttribute("data-color")).not.toBe(unfocusedColor);
      expect(polyline.getAttribute("data-weight")).not.toBe(unfocusedWeight);
    });

    it("draws the focused route thick, fully opaque, in the primary color", () => {
      render(
        <GlobalResponseMapLayer
          layer={makeLayer({ focusEventId: 101, routes: [{ ...makeLayer().routes[0], isFocused: true }] })}
        />,
      );

      const polyline = screen.getByTestId("polyline");
      expect(polyline).toHaveAttribute("data-weight", "6");
      expect(polyline).toHaveAttribute("data-opacity", "1");
      expect(polyline).toHaveAttribute("data-color", "var(--color-focus-ring)");
    });

    it("draws non-focused routes dimmed, thin, and neutral while a focus is active", () => {
      render(
        <GlobalResponseMapLayer
          layer={makeLayer({ focusEventId: 999, routes: [{ ...makeLayer().routes[0], isFocused: false }] })}
        />,
      );

      const polyline = screen.getByTestId("polyline");
      expect(polyline).toHaveAttribute("data-weight", "3");
      expect(polyline).toHaveAttribute("data-opacity", "0.25");
      expect(polyline).toHaveAttribute("data-color", "var(--color-text-muted)");
    });

    it("gives routes a distinct per-event color, not the dimmed grey, when no focus is set", () => {
      render(<GlobalResponseMapLayer layer={makeLayer({ focusEventId: null })} />);

      const polyline = screen.getByTestId("polyline");
      expect(polyline.getAttribute("data-color")).not.toBe("var(--color-text-muted)");
      expect(polyline).toHaveAttribute("data-weight", "4");
    });

    it("does not dim any marker when no focus is set", () => {
      render(<GlobalResponseMapLayer layer={makeLayer({ focusEventId: null })} />);

      const icons = screen.getAllByTestId("marker-icon");
      for (const icon of icons) {
        expect(icon.innerHTML).not.toContain("opacity:");
      }
    });

    it("dims a marker belonging to a non-focused event", () => {
      render(
        <GlobalResponseMapLayer
          layer={makeLayer({
            focusEventId: 999,
            originMarkers: [{ ...makeLayer().originMarkers[0], isFocused: false }],
          })}
        />,
      );

      const icons = screen.getAllByTestId("marker-icon");
      const dimmed = icons.some((icon) => icon.innerHTML.includes("opacity:0.25"));
      expect(dimmed).toBe(true);
    });

    it("does not dim the focused event's own marker", () => {
      render(
        <GlobalResponseMapLayer
          layer={makeLayer({
            focusEventId: 101,
            fireMarkers: [{ ...makeLayer().fireMarkers[0], isFocused: true }],
            originMarkers: [],
            targetMarkers: [],
            stationMarkers: [],
            routes: [],
          })}
        />,
      );

      const icon = screen.getByTestId("marker-icon");
      expect(icon.innerHTML).not.toContain("opacity:0.25");
    });
  });
});
