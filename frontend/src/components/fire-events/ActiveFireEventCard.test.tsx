import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { ActiveFireEventCard } from "./ActiveFireEventCard";
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

function renderCard(event: ActiveFireEvent) {
  return render(
    <MemoryRouter>
      <ActiveFireEventCard event={event} />
    </MemoryRouter>,
  );
}

describe("ActiveFireEventCard emphasis (Task A8, Part 9)", () => {
  it("emphasizes a CONFIRMED fire with HIGH severity", () => {
    renderCard(
      makeEvent({
        status: "confirmed",
        severity: { assessment_id: 1, status: "valid", score: 60, level: "high", assessed_at: "2026-09-20T11:00:00Z" },
      }),
    );

    const article = screen.getByRole("article");
    expect(article).toHaveAttribute("data-emphasized", "true");
    expect(article.className).toContain("fire-event-card--emphasized");
  });

  it("emphasizes a CONFIRMED fire with CRITICAL severity", () => {
    renderCard(
      makeEvent({
        status: "confirmed",
        severity: {
          assessment_id: 1,
          status: "valid",
          score: 90,
          level: "critical",
          assessed_at: "2026-09-20T11:00:00Z",
        },
      }),
    );

    expect(screen.getByRole("article")).toHaveAttribute("data-emphasized", "true");
  });

  it("does not emphasize a SUSPECTED fire even with CRITICAL severity", () => {
    renderCard(
      makeEvent({
        status: "suspected",
        severity: {
          assessment_id: 1,
          status: "valid",
          score: 90,
          level: "critical",
          assessed_at: "2026-09-20T11:00:00Z",
        },
      }),
    );

    const article = screen.getByRole("article");
    expect(article).toHaveAttribute("data-emphasized", "false");
    expect(article.className).not.toContain("fire-event-card--emphasized");
  });

  it("does not emphasize a CONFIRMED fire with LOW/MODERATE severity", () => {
    renderCard(
      makeEvent({
        status: "confirmed",
        severity: {
          assessment_id: 1,
          status: "valid",
          score: 10,
          level: "low",
          assessed_at: "2026-09-20T11:00:00Z",
        },
      }),
    );

    expect(screen.getByRole("article")).toHaveAttribute("data-emphasized", "false");
  });

  it("does not emphasize a CONFIRMED fire with null severity", () => {
    renderCard(makeEvent({ status: "confirmed", severity: null }));

    expect(screen.getByRole("article")).toHaveAttribute("data-emphasized", "false");
  });

  it("navigates to the existing Event Details route", () => {
    renderCard(makeEvent({ fire_event_id: 77 }));

    expect(screen.getByRole("link", { name: "View Event" })).toHaveAttribute("href", "/events/77");
  });
});

describe("ActiveFireEventCard Rule/AI scores (ML Task 7)", () => {
  it("shows Rule and AI as a compact row when an ml_summary is available", () => {
    renderCard(
      makeEvent({
        status: "confirmed",
        detection_confidence: 0.88,
        ml_summary: { available: true, model_score: 0.9 },
      }),
    );

    expect(screen.getByText("Detection")).toBeInTheDocument();
    expect(screen.getByText("Rule 0.88")).toBeInTheDocument();
    expect(screen.getByText("AI 0.90")).toBeInTheDocument();
    // The old label must be gone entirely.
    expect(screen.queryByText("Confidence")).not.toBeInTheDocument();
    expect(screen.queryByText("88%")).not.toBeInTheDocument();
  });

  it("shows Rule and AI for a SUSPECTED event without changing its status", () => {
    renderCard(
      makeEvent({
        status: "suspected",
        detection_confidence: 0.6,
        ml_summary: { available: true, model_score: 0.36 },
      }),
    );

    expect(screen.getByText("Rule 0.60")).toBeInTheDocument();
    expect(screen.getByText("AI 0.36")).toBeInTheDocument();
    expect(screen.getByText("Suspected")).toBeInTheDocument();
  });

  it("still renders Rule score and does not crash when there is no ml_summary at all", () => {
    renderCard(makeEvent({ detection_confidence: 0.6, ml_summary: null }));

    expect(screen.getByText("Rule 0.60")).toBeInTheDocument();
    expect(screen.getByText("AI -")).toBeInTheDocument();
  });

  it('shows "AI -" (never "AI 0.00") when ml_summary exists but is unavailable', () => {
    renderCard(
      makeEvent({
        detection_confidence: 0.6,
        ml_summary: { available: false, model_score: null },
      }),
    );

    expect(screen.getByText("Rule 0.60")).toBeInTheDocument();
    expect(screen.getByText("AI -")).toBeInTheDocument();
    expect(screen.queryByText("AI 0.00")).not.toBeInTheDocument();
    expect(screen.queryByText(/AI 0(\.0+)?$/)).not.toBeInTheDocument();
  });
});

describe("ActiveFireEventCard location display", () => {
  it("shows the real persisted location_name under the event id when present", () => {
    renderCard(makeEvent({ fire_event_id: 37, location_name: "Galilee Demo Area" }));

    expect(screen.getByText("Event #37")).toBeInTheDocument();
    expect(screen.getByText("Galilee Demo Area")).toBeInTheDocument();
  });

  it("renders an explicit 'Location unavailable' state rather than pretending a location exists", () => {
    renderCard(makeEvent({ fire_event_id: 37, location_name: null }));

    expect(screen.getByText("Event #37")).toBeInTheDocument();
    expect(screen.getByText("Location unavailable")).toBeInTheDocument();
  });
});

describe("ActiveFireEventCard Opened time (created_at, not detected_at)", () => {
  it("labels the timestamp row 'Opened' and shows created_at, not detected_at", () => {
    renderCard(
      makeEvent({
        detected_at: "2026-09-20T11:44:00Z",
        created_at: "2026-09-20T11:46:00Z",
      }),
    );

    expect(screen.getByText("Opened")).toBeInTheDocument();
    expect(screen.queryByText("Detected")).not.toBeInTheDocument();
    const openedTime = document.querySelector("time");
    expect(openedTime).toHaveAttribute("dateTime", "2026-09-20T11:46:00Z");
  });
});
