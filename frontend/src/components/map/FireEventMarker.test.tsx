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

  it.each([
    ["suspected", "var(--color-warning)"],
    ["confirmed", "var(--color-danger)"],
    ["resolved", "var(--color-success)"],
    ["dismissed", "var(--color-text-muted)"],
  ] as const)("colors the marker for status %s using the shared status tone", (status, expectedColor) => {
    render(<FireEventMarker fireEvent={makeFireEvent({ status })} />);

    const icon = screen.getByTestId("marker-icon").querySelector("span") as HTMLElement;
    expect(icon.style.background).toBe(expectedColor);
  });
});
