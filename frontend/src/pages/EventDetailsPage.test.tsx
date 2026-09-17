import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { EventDetailsResult } from "../types/eventDetails";
import { EventDetailsPage } from "./EventDetailsPage";

const { getEventDetailsMock } = vi.hoisted(() => ({
  getEventDetailsMock: vi.fn(),
}));

vi.mock("../api/eventDetails", () => ({
  getEventDetails: getEventDetailsMock,
}));

vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

function makeResult(overrides: Partial<EventDetailsResult> = {}): EventDetailsResult {
  return {
    as_of: "2026-09-17T14:00:00Z",
    fire_event: {
      fire_event_id: 12,
      status: "confirmed",
      latitude: 32.731,
      longitude: 35.046,
      detection_confidence: 0.91,
      detected_at: "2026-09-17T13:20:00Z",
      updated_at: "2026-09-17T13:28:00Z",
      methodology: "detector",
      methodology_version: "1.0",
    },
    severity: null,
    danger: null,
    spread_predictions: [],
    targets: [],
    stations: [],
    resources: [],
    current_response_plan: null,
    ...overrides,
  };
}

function renderPage(fireEventId = "12") {
  return render(
    <MemoryRouter initialEntries={[`/events/${fireEventId}`]}>
      <Routes>
        <Route path="/events/:fireEventId" element={<EventDetailsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("EventDetailsPage", () => {
  beforeEach(() => {
    getEventDetailsMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the event id heading and a loading state on initial render", () => {
    getEventDetailsMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage("12");

    expect(screen.getByRole("heading", { name: "Event #12" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it('shows an "Event Not Found" state on a 404, without a Retry button', async () => {
    getEventDetailsMock.mockRejectedValue(new ApiError("not found", 404));

    renderPage("999");

    expect(await screen.findByRole("heading", { name: "Event Not Found" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("shows a generic error state with Retry on a non-404 failure, and recovers on retry", async () => {
    const user = userEvent.setup();
    getEventDetailsMock.mockRejectedValueOnce(new ApiError("boom", 500)).mockResolvedValueOnce(makeResult());

    renderPage();
    await screen.findByRole("button", { name: "Retry" });
    await user.click(screen.getByRole("button", { name: "Retry" }));

    await screen.findByText("Data as of:", { exact: false });
    expect(screen.queryByRole("heading", { name: "Unable to load event details." })).not.toBeInTheDocument();
  });

  it("always shows a Back to Active Events link, even while loading", () => {
    getEventDetailsMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage();

    expect(screen.getByRole("link", { name: "Back to Active Events" })).toHaveAttribute("href", "/events");
  });

  it("displays FireEvent identity, status, coordinates, confidence, and timestamps", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: {
          fire_event_id: 55,
          status: "suspected",
          latitude: 32.5,
          longitude: 35.1,
          detection_confidence: 0.72,
          detected_at: "2026-09-17T13:20:00Z",
          updated_at: "2026-09-17T13:28:00Z",
          methodology: "detector",
          methodology_version: "1.0",
        },
      }),
    );

    renderPage("55");
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("Suspected")).toBeInTheDocument();
    expect(scoped.getByText("32.5000, 35.1000")).toBeInTheDocument();
    expect(scoped.getByText("72%")).toBeInTheDocument();
    expect(fireEventSection.querySelectorAll("time")).toHaveLength(2); // detected_at + updated_at
  });

  it('shows "Not available" for severity when there is no assessment at all', async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ severity: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const severityFact = screen.getByText("Severity").closest("div") as HTMLElement;
    expect(within(severityFact).getByText("Not available")).toBeInTheDocument();
  });

  it("shows severity level and score when the latest assessment is valid", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        severity: { assessment_id: 44, status: "valid", score: 81.4, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Critical")).toBeInTheDocument();
    expect(screen.getByText("Severity score: 81.4")).toBeInTheDocument();
  });

  it("does not fabricate a severity level when the assessment is insufficient_data", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        severity: { assessment_id: 45, status: "insufficient_data", score: null, level: null, assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Insufficient data for a severity assessment.")).toBeInTheDocument();
    for (const label of ["Low", "Moderate", "High", "Critical"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });

  it('shows "Not available" for danger (always null today)', async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ danger: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const dangerRow = screen.getByText("Danger").closest("div") as HTMLElement;
    expect(within(dangerRow).getByText("Not available")).toBeInTheDocument();
  });

  it("shows an explicit empty-state message when there are no response targets", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ targets: [] }));

    renderPage();

    expect(await screen.findByText("No response targets")).toBeInTheDocument();
    expect(
      screen.getByText("No response targets have been generated for this event yet."),
    ).toBeInTheDocument();
  });

  it("lists each response target when present", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        targets: [
          {
            target_order: 0,
            target_type: "active_fire",
            latitude: 32.731,
            longitude: 35.046,
            priority_score: 1.0,
            prediction_horizon_minutes: null,
          },
          {
            target_order: 1,
            target_type: "predicted_risk",
            latitude: 32.74,
            longitude: 35.05,
            priority_score: 0.6,
            prediction_horizon_minutes: 30,
          },
        ],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const targetsSection = screen.getByRole("heading", { name: "Response Targets" }).closest("section") as HTMLElement;
    const scoped = within(targetsSection);
    expect(scoped.getByText("Active fire")).toBeInTheDocument();
    expect(scoped.getByText("Predicted risk")).toBeInTheDocument();
    expect(scoped.queryByText("No response targets")).not.toBeInTheDocument();
  });

  it("shows an explicit empty-state message when there are no resources", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ resources: [] }));

    renderPage();

    expect(await screen.findByText("No firefighting resources")).toBeInTheDocument();
  });

  it("lists each resource when present", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        resources: [{ resource_id: "R1", station_id: "S1", status: "available" }],
      }),
    );

    renderPage();

    expect(await screen.findByText("R1")).toBeInTheDocument();
    expect(screen.getByText("Available")).toBeInTheDocument();
  });

  it("does not show a View Current Response Plan link when there is no current plan", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ current_response_plan: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("link", { name: "View Current Response Plan" })).not.toBeInTheDocument();
  });

  it("shows a View Current Response Plan link to /events/{id}/plan when a current plan exists", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, fire_event_id: 12 },
        current_response_plan: {
          plan_id: 77,
          generated_at: "2026-09-17T13:30:00Z",
          methodology: "ga",
          methodology_version: "1.0",
          plan_score: 90,
          coverage_score: 1,
          average_eta_seconds: 120,
          actions: [],
          uncovered_target_ids: [],
          baseline_comparison: null,
        },
      }),
    );

    renderPage("12");
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByRole("link", { name: "View Current Response Plan" })).toHaveAttribute(
      "href",
      "/events/12/plan",
    );
  });

  it("renders the map with a marker for the fire event", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByRole("region", { name: "Map of Event #12" })).toBeInTheDocument();
    expect(screen.getAllByTestId("marker").length).toBeGreaterThanOrEqual(1);
  });

  it("hides a layer's markers when its layer control is unchecked", async () => {
    const user = userEvent.setup();
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        stations: [
          { station_id: "S1", name: "Central", latitude: 32.0, longitude: 34.8, station_type: null, address: null },
        ],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const stationsToggle = screen.getByRole("checkbox", { name: /fire stations/i });
    expect(stationsToggle).toBeChecked();
    // Fire event marker + station marker = 2 markers before toggling off.
    expect(screen.getAllByTestId("marker")).toHaveLength(2);

    await user.click(stationsToggle);

    expect(screen.getAllByTestId("marker")).toHaveLength(1);
  });

  it("refreshes data when Refresh is clicked", async () => {
    const user = userEvent.setup();
    getEventDetailsMock
      .mockResolvedValueOnce(makeResult({ as_of: "2026-09-17T14:00:00Z" }))
      .mockResolvedValueOnce(makeResult({ as_of: "2026-09-17T15:00:00Z" }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    await user.click(screen.getByRole("button", { name: "Refresh" }));

    await waitFor(() => {
      const asOfParagraph = document.querySelector(".event-details-page__as-of time");
      expect(asOfParagraph).toHaveAttribute("dateTime", "2026-09-17T15:00:00Z");
    });
  });
});
