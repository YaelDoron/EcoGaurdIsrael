import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ActiveFiresPanel } from "./ActiveFiresPanel";
import type { ActiveFireEvent } from "../../types/activeFireEvents";

function makeEvent(overrides: Partial<ActiveFireEvent> = {}): ActiveFireEvent {
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
    ml_summary: null,
    ...overrides,
  };
}

function renderPanel(activeFires: ActiveFireEvent[], globalPlanningRunId: number | null = null) {
  return render(
    <MemoryRouter>
      <ActiveFiresPanel activeFires={activeFires} globalPlanningRunId={globalPlanningRunId} />
    </MemoryRouter>,
  );
}

describe("ActiveFiresPanel", () => {
  it("shows an empty-state message when there are no active fires", () => {
    renderPanel([]);

    expect(screen.getByText("No active wildfire events")).toBeInTheDocument();
  });

  it("renders one card per active fire, preserving server order", () => {
    renderPanel([makeEvent({ fire_event_id: 9 }), makeEvent({ fire_event_id: 2 })]);

    const headings = screen.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(["Event #9", "Event #2"]);
  });

  it("shows a count badge equal to the number of active fires", () => {
    renderPanel([makeEvent({ fire_event_id: 1 }), makeEvent({ fire_event_id: 2 })]);

    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("does not cap the list to a small inner-scroll height for the standard demo case", () => {
    // jsdom does not apply imported CSS, so this is a source-level guard:
    // the standard 1-2 fire demo must not need to scroll to see every card.
    const css = readFileSync("src/components/dashboard/ActiveFiresPanel.css", "utf-8");

    expect(css).not.toMatch(/max-height/);
    expect(css).not.toMatch(/overflow-y/);
  });

  it("does not show View Response Plan when no real global planning run is available", () => {
    renderPanel([], null);

    expect(screen.queryByRole("button", { name: /view response plan/i })).not.toBeInTheDocument();
  });

  it("shows View Response Plan when a real global planning run id is available", () => {
    renderPanel([], 42);

    expect(screen.getByRole("button", { name: "View Response Plan" })).toBeInTheDocument();
  });
});
