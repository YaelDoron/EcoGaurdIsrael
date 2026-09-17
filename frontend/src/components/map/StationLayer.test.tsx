import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StationLayer } from "./StationLayer";
import type { FireStation } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeStation(overrides: Partial<FireStation> = {}): FireStation {
  return {
    station_id: "S1",
    name: "Central Station",
    latitude: 32.0,
    longitude: 34.8,
    station_type: "urban",
    address: "1 Main St",
    ...overrides,
  };
}

describe("StationLayer", () => {
  it("renders nothing when there are no stations", () => {
    const { container } = render(<StationLayer stations={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one marker per station at its coordinates", () => {
    render(
      <StationLayer
        stations={[makeStation({ station_id: "S1" }), makeStation({ station_id: "S2", latitude: 32.1 })]}
      />,
    );

    const markers = screen.getAllByTestId("marker");
    expect(markers).toHaveLength(2);
  });

  it("shows name, type, and address in the popup, omitting fields the backend sent as null", () => {
    render(<StationLayer stations={[makeStation({ station_type: null, address: null })]} />);

    expect(screen.getByText("Central Station")).toBeInTheDocument();
    expect(screen.queryByText(/Type:/)).not.toBeInTheDocument();
    expect(screen.queryByText("1 Main St")).not.toBeInTheDocument();
  });
});
