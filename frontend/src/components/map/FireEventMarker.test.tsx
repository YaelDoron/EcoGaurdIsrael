import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FireEventMarker } from "./FireEventMarker";
import type { FireEventSummary } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeFireEvent(overrides: Partial<FireEventSummary> = {}): FireEventSummary {
  return {
    fire_event_id: 12,
    status: "confirmed",
    latitude: 32.731,
    longitude: 35.046,
    detection_confidence: 0.91,
    detected_at: "2026-09-17T13:20:00Z",
    updated_at: "2026-09-17T13:28:00Z",
    methodology: "detector",
    methodology_version: "1.0",
    ...overrides,
  };
}

describe("FireEventMarker", () => {
  it("renders exactly one marker positioned at the FireEvent's coordinates", () => {
    render(<FireEventMarker fireEvent={makeFireEvent({ latitude: 32.731, longitude: 35.046 })} />);

    const markers = screen.getAllByTestId("marker");
    expect(markers).toHaveLength(1);
    expect(markers[0]).toHaveAttribute("data-lat", "32.731");
    expect(markers[0]).toHaveAttribute("data-lng", "35.046");
  });

  it("shows the fire event id and status label in its popup", () => {
    render(<FireEventMarker fireEvent={makeFireEvent({ fire_event_id: 7, status: "suspected" })} />);

    expect(screen.getByText("Event #7")).toBeInTheDocument();
    expect(screen.getByText("Status: Suspected")).toBeInTheDocument();
  });

  it("falls back to raw coordinates in the body when no location name is known", () => {
    render(<FireEventMarker fireEvent={makeFireEvent({ latitude: 32.731, longitude: 35.046 })} />);

    expect(screen.getByText(/32\.731/)).toBeInTheDocument();
  });

  it("shows '{name} Wildfire' as the title and the location name (not coordinates) in the body when known", () => {
    render(<FireEventMarker fireEvent={makeFireEvent({ fire_event_id: 7 })} locationName="Galilee" />);

    expect(screen.getByText("Galilee Wildfire")).toBeInTheDocument();
    expect(screen.queryByText("Event #7")).not.toBeInTheDocument();
    expect(screen.getByText("Galilee")).toBeInTheDocument();
    expect(screen.queryByText(/32\.731/)).not.toBeInTheDocument();
  });

  it.each([
    ["suspected", "rgb(255, 69, 0)"],
    ["confirmed", "rgb(255, 34, 0)"],
    ["resolved", "rgb(156, 163, 175)"],
    ["dismissed", "rgb(156, 163, 175)"],
  ] as const)("draws a fire glyph for status %s (fiery when active, muted when inactive)", (status, expectedColor) => {
    render(<FireEventMarker fireEvent={makeFireEvent({ status })} />);

    const icon = screen.getByTestId("marker-icon").querySelector("[data-fire-icon]") as HTMLElement;
    expect(icon.querySelector("svg")).not.toBeNull();
    expect(icon.style.color).toBe(expectedColor);
  });
});
