import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OperationalLayer } from "./OperationalLayer";
import type { FireStation, FirefightingResource } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

const STATIONS: FireStation[] = [
  { station_id: "S1", name: "Central Station", latitude: 32.0, longitude: 34.8, station_type: null, address: null },
  { station_id: "S2", name: "North Station", latitude: 32.5, longitude: 35.2, station_type: null, address: null },
];

function makeResource(overrides: Partial<FirefightingResource> = {}): FirefightingResource {
  return { resource_id: "R1", station_id: "S1", status: "available", ...overrides };
}

describe("OperationalLayer", () => {
  it("renders nothing when there are no resources", () => {
    const { container } = render(<OperationalLayer resources={[]} stations={STATIONS} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("groups multiple resources at the same station into a single marker", () => {
    render(
      <OperationalLayer
        resources={[makeResource({ resource_id: "R1" }), makeResource({ resource_id: "R2" })]}
        stations={STATIONS}
      />,
    );

    expect(screen.getAllByTestId("marker")).toHaveLength(1);
    expect(screen.getByText(/R1: Available/)).toBeInTheDocument();
    expect(screen.getByText(/R2: Available/)).toBeInTheDocument();
  });

  it("renders one marker per station that has resources", () => {
    render(
      <OperationalLayer
        resources={[makeResource({ station_id: "S1" }), makeResource({ resource_id: "R2", station_id: "S2" })]}
        stations={STATIONS}
      />,
    );

    expect(screen.getAllByTestId("marker")).toHaveLength(2);
  });

  it("skips a resource whose station is not in the fetched station list rather than guessing a location", () => {
    render(<OperationalLayer resources={[makeResource({ station_id: "UNKNOWN" })]} stations={STATIONS} />);

    expect(screen.queryAllByTestId("marker")).toHaveLength(0);
  });

  it("colors the station marker green when at least one resource is available", () => {
    render(
      <OperationalLayer
        resources={[makeResource({ status: "assigned" }), makeResource({ resource_id: "R2", status: "available" })]}
        stations={STATIONS}
      />,
    );

    const icon = screen.getByTestId("marker-icon").querySelector("span") as HTMLElement;
    expect(icon.style.background).toBe("var(--color-success)");
  });

  it("colors the station marker amber when resources are assigned but none available", () => {
    render(<OperationalLayer resources={[makeResource({ status: "assigned" })]} stations={STATIONS} />);

    const icon = screen.getByTestId("marker-icon").querySelector("span") as HTMLElement;
    expect(icon.style.background).toBe("var(--color-warning)");
  });

  it("colors the station marker muted when every resource is unavailable", () => {
    render(<OperationalLayer resources={[makeResource({ status: "unavailable" })]} stations={STATIONS} />);

    const icon = screen.getByTestId("marker-icon").querySelector("span") as HTMLElement;
    expect(icon.style.background).toBe("var(--color-text-muted)");
  });
});
