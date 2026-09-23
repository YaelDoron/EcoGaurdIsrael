import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OperationsMap } from "./OperationsMap";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import type { FireDangerArea } from "../../types/fireDanger";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeArea(overrides: Partial<FireDangerArea> = {}): FireDangerArea {
  return {
    area_id: "area-carmel",
    area_name: "Carmel",
    center: { latitude: 32.731, longitude: 35.046 },
    radius_km: 5,
    assessment: null,
    ...overrides,
  };
}

function makeFire(overrides: Partial<ActiveFireEvent> = {}): ActiveFireEvent {
  return {
    fire_event_id: 15,
    status: "confirmed",
    latitude: 32.7,
    longitude: 35.0,
    detection_confidence: 0.9,
    detected_at: "2026-09-20T11:00:00Z",
    updated_at: "2026-09-20T11:05:00Z",
    created_at: "2026-09-20T11:00:30Z",
    severity: null,
    location_name: null,
    ml_summary: null,
    ...overrides,
  };
}

describe("OperationsMap", () => {
  it("renders the map region and a compact legend even with no data", () => {
    render(<OperationsMap fireDangerAreas={[]} activeFires={[]} />);

    expect(screen.getByRole("region", { name: "National operations map" })).toBeInTheDocument();
    expect(screen.getByText("Legend")).toBeInTheDocument();
    expect(screen.getByText("Fire danger")).toBeInTheDocument();
    expect(screen.getByText("Active fires")).toBeInTheDocument();
  });

  it("renders the legend as a compact overlay, not a full-width strip below the map", () => {
    render(<OperationsMap fireDangerAreas={[]} activeFires={[]} />);

    const legend = screen.getByText("Legend").closest(".operations-map-legend");
    expect(legend).toBeInTheDocument();
    // The legend is a sibling overlay inside `.operations-map`, absolutely
    // positioned over the map (see OperationsMapLegend.css) - never its own
    // block-level row below the map container.
    const mapWrapper = document.querySelector(".operations-map");
    expect(mapWrapper).toContainElement(legend as HTMLElement);
  });

  it("renders both a Fire Danger circle and an active-fire marker simultaneously", () => {
    render(<OperationsMap fireDangerAreas={[makeArea()]} activeFires={[makeFire()]} />);

    expect(screen.getByTestId("circle")).toBeInTheDocument();
    expect(screen.getByTestId("marker")).toBeInTheDocument();
  });

  it("does not crash and still renders the map region when both arrays are empty", () => {
    render(<OperationsMap fireDangerAreas={[]} activeFires={[]} />);

    expect(screen.queryByTestId("circle")).not.toBeInTheDocument();
    expect(screen.queryByTestId("marker")).not.toBeInTheDocument();
    expect(screen.getByTestId("map-container")).toBeInTheDocument();
  });

  it("disables mouse-wheel zoom so the page can scroll normally over the map", () => {
    render(<OperationsMap fireDangerAreas={[]} activeFires={[]} />);

    expect(screen.getByTestId("map-container")).toHaveAttribute("data-scroll-wheel-zoom", "false");
  });

  it("only disables scroll-wheel zoom - dragging/panning and the +/- zoom controls stay at Leaflet's defaults", () => {
    // Leaflet enables dragging and the zoom control by default; this is a
    // source-level guard that we never explicitly turn either off, since
    // jsdom cannot exercise real Leaflet control rendering.
    const source = readFileSync("src/components/dashboard/OperationsMap.tsx", "utf-8");

    expect(source).not.toMatch(/dragging=\{?false/);
    expect(source).not.toMatch(/zoomControl=\{?false/);
  });
});
