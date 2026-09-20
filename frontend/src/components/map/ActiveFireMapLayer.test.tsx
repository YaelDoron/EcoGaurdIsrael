import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ActiveFireMapLayer } from "./ActiveFireMapLayer";
import type { ActiveFireEvent } from "../../types/activeFireEvents";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeFire(overrides: Partial<ActiveFireEvent> = {}): ActiveFireEvent {
  return {
    fire_event_id: 15,
    status: "confirmed",
    latitude: 32.731,
    longitude: 35.046,
    detection_confidence: 0.9,
    detected_at: "2026-09-20T11:00:00Z",
    updated_at: "2026-09-20T11:05:00Z",
    created_at: "2026-09-20T11:00:30Z",
    severity: null,
    location_name: null,
    ...overrides,
  };
}

function iconHtml(marker: HTMLElement): string {
  return marker.querySelector('[data-testid="marker-icon"]')?.innerHTML ?? "";
}

describe("ActiveFireMapLayer", () => {
  it("renders nothing for an empty active_fires array", () => {
    const { container } = render(<ActiveFireMapLayer activeFires={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one marker per active fire at its persisted coordinates", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[makeFire({ fire_event_id: 1 }), makeFire({ fire_event_id: 2, latitude: 33.0, longitude: 35.5 })]}
      />,
    );

    const markers = screen.getAllByTestId("marker");
    expect(markers).toHaveLength(2);
    expect(markers[1]).toHaveAttribute("data-lat", "33");
    expect(markers[1]).toHaveAttribute("data-lng", "35.5");
  });

  it("does not fabricate an area name - popup shows only id, status, severity, coordinates when location_name is null", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ fire_event_id: 42, location_name: null })]} />);

    expect(screen.getByText("Fire Event #42")).toBeInTheDocument();
    expect(screen.queryByText(/carmel/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/golan/i)).not.toBeInTheDocument();
  });

  it("shows the real persisted location_name in the popup when present", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ fire_event_id: 42, location_name: "Galilee Demo Area" })]} />);

    expect(screen.getByText("Fire Event #42")).toBeInTheDocument();
    expect(screen.getByText("Galilee Demo Area")).toBeInTheDocument();
  });

  it("renders a neutral severity badge when severity is null, never fabricating a level", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ severity: null })]} />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("applies the emphasized icon class for CONFIRMED + HIGH severity", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "confirmed",
            severity: { assessment_id: 1, status: "valid", score: 65, level: "high", assessed_at: "2026-09-20T11:00:00Z" },
          }),
        ]}
      />,
    );

    const marker = screen.getByTestId("marker");
    expect(iconHtml(marker)).toContain("active-fire-icon__halo");
  });

  it("applies the emphasized icon class for CONFIRMED + CRITICAL severity", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "confirmed",
            severity: {
              assessment_id: 1,
              status: "valid",
              score: 90,
              level: "critical",
              assessed_at: "2026-09-20T11:00:00Z",
            },
          }),
        ]}
      />,
    );

    const marker = screen.getByTestId("marker");
    expect(iconHtml(marker)).toContain("active-fire-icon__halo");
  });

  it("does NOT apply emphasis for SUSPECTED + CRITICAL (status must also be confirmed)", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "suspected",
            severity: {
              assessment_id: 1,
              status: "valid",
              score: 90,
              level: "critical",
              assessed_at: "2026-09-20T11:00:00Z",
            },
          }),
        ]}
      />,
    );

    const marker = screen.getByTestId("marker");
    expect(iconHtml(marker)).not.toContain("active-fire-icon__halo");
  });

  it("does not apply emphasis for CONFIRMED + LOW/MODERATE severity", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "confirmed",
            severity: {
              assessment_id: 1,
              status: "valid",
              score: 20,
              level: "moderate",
              assessed_at: "2026-09-20T11:00:00Z",
            },
          }),
        ]}
      />,
    );

    const marker = screen.getByTestId("marker");
    expect(iconHtml(marker)).not.toContain("active-fire-icon__halo");
  });

  it("does not apply emphasis when severity is null even for a confirmed fire", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ status: "confirmed", severity: null })]} />);

    const marker = screen.getByTestId("marker");
    expect(iconHtml(marker)).not.toContain("active-fire-icon__halo");
  });

  it("labels a SUSPECTED event's popup as Detection: Suspected", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ status: "suspected" })]} />);

    expect(screen.getByText("Detection:")).toBeInTheDocument();
    expect(screen.getByText("Suspected")).toBeInTheDocument();
  });

  it("labels a CONFIRMED event's popup as Detection: Confirmed", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ status: "confirmed" })]} />);

    expect(screen.getByText("Detection:")).toBeInTheDocument();
    expect(screen.getByText("Confirmed")).toBeInTheDocument();
  });

  it("explicitly labels Severity, distinct from Detection status", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "confirmed",
            severity: { assessment_id: 1, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-20T11:00:00Z" },
          }),
        ]}
      />,
    );

    expect(screen.getByText("Severity:")).toBeInTheDocument();
    expect(screen.getByText("Critical")).toBeInTheDocument();
  });

  it("never labels a detection status (Suspected/Confirmed) as Severity", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ status: "suspected", severity: null })]} />);

    const popup = screen.getByTestId("popup");
    // "Suspected" appears only next to the "Detection:" label, never the "Severity:" one.
    expect(popup.textContent).toContain("Detection:");
    expect(popup.textContent?.replace("Detection:", "")).not.toContain("Severity: Suspected");
  });

  it("never presents a Severity value (e.g. High) as the Detection status", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[
          makeFire({
            status: "confirmed",
            severity: { assessment_id: 1, status: "valid", score: 65, level: "high", assessed_at: "2026-09-20T11:00:00Z" },
          }),
        ]}
      />,
    );

    // "High" is the Severity value - "Confirmed" remains the Detection value.
    expect(screen.getByText("Confirmed")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
  });

  it("never writes 'Confidence: Suspected' - Suspected is a status, not a numeric confidence", () => {
    render(<ActiveFireMapLayer activeFires={[makeFire({ status: "suspected" })]} />);

    expect(screen.queryByText(/Confidence:/)).not.toBeInTheDocument();
  });

  it("renders a hollow (3px ring, white center) marker for SUSPECTED, distinct from CONFIRMED's solid fill", () => {
    render(
      <ActiveFireMapLayer
        activeFires={[makeFire({ fire_event_id: 1, status: "suspected" }), makeFire({ fire_event_id: 2, status: "confirmed" })]}
      />,
    );

    const [suspectedMarker, confirmedMarker] = screen.getAllByTestId("marker");
    expect(iconHtml(suspectedMarker)).toContain("border:3px solid");
    expect(iconHtml(suspectedMarker)).toContain("background:#ffffff");
    expect(iconHtml(confirmedMarker)).toContain("border:2px solid #ffffff");
    expect(iconHtml(suspectedMarker)).not.toEqual(iconHtml(confirmedMarker));
  });
});
