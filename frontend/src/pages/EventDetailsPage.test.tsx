import { render, screen, within } from "@testing-library/react";
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

const { reverseGeocodeMock } = vi.hoisted(() => ({ reverseGeocodeMock: vi.fn() }));

vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));

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
    detection_evidence: { satellite: [], news: [] },
    spread_predictions: [],
    targets: [],
    stations: [],
    resources: [],
    station_summaries: [],
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
    reverseGeocodeMock.mockReset();
    reverseGeocodeMock.mockResolvedValue(null);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the event id heading and a loading state on initial render", () => {
    getEventDetailsMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage("12");

    expect(screen.getByRole("heading", { level: 1, name: "Loading Event…" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: /Event #12/ })).not.toBeInTheDocument();
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

  it("always shows the Active Events breadcrumb link, even while loading", () => {
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
    expect(fireEventSection.querySelectorAll("time")).toHaveLength(1); // detected_at only
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
    expect(screen.queryByText(/Severity score/)).not.toBeInTheDocument();
    expect(screen.queryByText("81.4")).not.toBeInTheDocument();
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

  it("hides the Danger fact entirely when danger is null", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ danger: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByText("Danger")).not.toBeInTheDocument();
  });

  it("shows the Danger fact when a danger assessment has a level", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        danger: { assessment_id: 1, status: "valid", score: 55.5, level: "high", assessed_at: "2026-09-17T13:00:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Danger")).toBeInTheDocument();
    expect(screen.getByText("Danger score: 55.5")).toBeInTheDocument();
  });

  it("does not render a Response Targets card, but still plots targets on the map", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        targets: [
          { target_order: 0, target_type: "active_fire", latitude: 32.731, longitude: 35.046, priority_score: 1.0, prediction_horizon_minutes: null },
          { target_order: 1, target_type: "predicted_risk", latitude: 32.74, longitude: 35.05, priority_score: 0.6, prediction_horizon_minutes: 30 },
        ],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("heading", { name: "Response Targets" })).not.toBeInTheDocument();
    // fire event marker + two target markers
    expect(screen.getAllByTestId("marker")).toHaveLength(3);
  });

  it("does not render a raw firefighting-resources list on the page", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        resources: [{ resource_id: "R1", station_id: "S1", status: "available" }],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("heading", { name: "Firefighting Resources" })).not.toBeInTheDocument();
    expect(screen.queryByText("R1")).not.toBeInTheDocument();
  });

  it("shows an empty state in the Detection Evidence section when there is no evidence", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ detection_evidence: { satellite: [], news: [] } }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const evidenceSection = screen
      .getByRole("heading", { name: "Detection Evidence" })
      .closest("section") as HTMLElement;
    expect(within(evidenceSection).getByText("No detection evidence")).toBeInTheDocument();
  });

  it("shows satellite and news evidence in the Detection Evidence section when present", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        detection_evidence: {
          satellite: [
            {
              id: 501,
              detected_at: "2026-09-17T13:15:00Z",
              latitude: 32.7,
              longitude: 35.0,
              confidence: "high",
              frp: 15.2,
              brightness: 310.5,
              satellite: "Terra",
              instrument: "MODIS",
              day_night: "D",
            },
          ],
          news: [
            {
              id: 701,
              title: "Blaze reported near reserve",
              summary: "A wildfire was reported near the nature reserve.",
              source: "haaretz",
              observed_at: "2026-09-17T13:10:00Z",
              location_name: "Modiin",
              latitude: 31.9,
              longitude: 35.0,
            },
          ],
        },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const evidenceSection = screen
      .getByRole("heading", { name: "Detection Evidence" })
      .closest("section") as HTMLElement;
    const scoped = within(evidenceSection);
    expect(scoped.getByText("Terra")).toBeInTheDocument();
    expect(scoped.getByText("Blaze reported near reserve")).toBeInTheDocument();
  });

  it("shows an explicit empty-state message when there is no spread prediction", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ spread_predictions: [] }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const spreadSection = screen
      .getByRole("heading", { name: "Spread Prediction" })
      .closest("section") as HTMLElement;
    expect(within(spreadSection).getByText("No spread prediction")).toBeInTheDocument();
  });

  it("shows a metric card per spread-prediction horizon", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        spread_predictions: [
          {
            horizon_minutes: 30,
            status: "valid",
            predicted_at: "2026-09-17T13:25:00Z",
            cells: [
              {
                latitude: 32.74,
                longitude: 35.05,
                spread_probability: 0.6,
                spread_risk_score: 60,
                reached_step: 1,
                reached_minutes: 5,
              },
            ],
          },
          {
            horizon_minutes: 60,
            status: "insufficient_data",
            predicted_at: "2026-09-17T13:25:00Z",
            cells: [],
          },
        ],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const spreadSection = screen
      .getByRole("heading", { name: "Spread Prediction" })
      .closest("section") as HTMLElement;
    const scoped = within(spreadSection);
    expect(scoped.getByText("30 min horizon")).toBeInTheDocument();
    expect(scoped.getByText("Spread predicted")).toBeInTheDocument();
    expect(scoped.getByText("1 predicted cell")).toBeInTheDocument();
    expect(scoped.getByText("60 min horizon")).toBeInTheDocument();
    expect(scoped.getByText("Insufficient data")).toBeInTheDocument();
    expect(scoped.queryByText(/predicted cells/)).not.toBeInTheDocument();
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

  it("does not render the old subtitle", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByText("Full details and interactive map for this wildfire event.")).not.toBeInTheDocument();
  });

  it("appends a persisted location name from news evidence to the title", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        detection_evidence: {
          satellite: [],
          news: [
            {
              id: 1,
              title: "t",
              summary: "s",
              source: "src",
              observed_at: "2026-09-17T13:10:00Z",
              location_name: "Modiin",
              latitude: null,
              longitude: null,
            },
          ],
        },
      }),
    );

    renderPage();

    expect(await screen.findByRole("heading", { level: 1, name: "Modiin Wildfire Event" })).toBeInTheDocument();
  });

  it("falls back to a generic event title, not coordinates, when no location name is available", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const title = screen.getByRole("heading", { level: 1, name: "Wildfire Event #12" });
    expect(title).not.toHaveTextContent(/32\.7310/);
  });

  it("renders no subtitle under the header", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(document.querySelector(".page-header__description")).toBeNull();
  });

  it("never draws response routes on the Event Details map, even with a current plan", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        current_response_plan: {
          plan_id: 77,
          generated_at: "2026-09-17T13:30:00Z",
          methodology: "ga",
          methodology_version: "1.0",
          plan_score: 90,
          coverage_score: 1,
          average_eta_seconds: 120,
          actions: [
            {
              resource_id: "R1",
              station_id: "S1",
              response_target_id: 1,
              target_type: "active_fire",
              target_priority: 1,
              eta_seconds: 120,
              route_distance_meters: 500,
              node_path: [1, 2, 3],
            },
          ],
          uncovered_target_ids: [],
          baseline_comparison: null,
        },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });

  it("softens a Critical severity when the event is only Suspected", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, status: "suspected" },
        severity: { assessment_id: 1, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Pending verification / Critical")).toHaveClass("badge--warning");
    expect(screen.queryByText("Critical")).not.toBeInTheDocument();
  });

  it("shows the normal Critical badge when the event is Confirmed", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        severity: { assessment_id: 1, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Critical")).toBeInTheDocument();
    expect(screen.queryByText("Pending verification / Critical")).not.toBeInTheDocument();
  });

  it("stacks all four cards in one column beside the map", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const grid = screen.getByRole("heading", { name: "Assessments" }).closest(".event-details-page__cards") as HTMLElement;
    expect(grid).not.toBeNull();
    for (const name of ["Fire Event", "Assessments", "Detection Evidence", "Spread Prediction"]) {
      expect(within(grid).getByRole("heading", { name })).toBeInTheDocument();
    }
    expect(within(grid).queryByRole("heading", { name: "Map" })).not.toBeInTheDocument();
  });

  it("places the map before the technical-detail sections (map-first layout)", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const headings = screen.getAllByRole("heading", { level: 2 }).map((heading) => heading.textContent);
    const mapIndex = headings.indexOf("Map");
    const fireEventIndex = headings.indexOf("Fire Event");
    expect(mapIndex).toBeGreaterThanOrEqual(0);
    expect(fireEventIndex).toBeGreaterThan(mapIndex);
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

  it("has no manual Refresh button in the header", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("button", { name: /refresh/i })).not.toBeInTheDocument();
  });

  it("does not show an Updated field in the Fire Event card", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByText("Updated")).not.toBeInTheDocument();
    expect(screen.getByText("Detected")).toBeInTheDocument();
  });

  it("titles the page with the location and shows the event id as a badge beside Data as of", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        detection_evidence: {
          satellite: [],
          news: [{ id: 1, title: "t", summary: "s", source: "x", observed_at: "2026-09-17T13:10:00Z", location_name: "Haifa Subdistrict", latitude: null, longitude: null }],
        },
      }),
    );

    renderPage();

    expect(await screen.findByRole("heading", { level: 1, name: "Haifa Subdistrict Wildfire Event" })).toBeInTheDocument();
    const asOf = document.querySelector(".event-details-page__as-of") as HTMLElement;
    expect(within(asOf).getByText("Event #12")).toBeInTheDocument();
  });

  it("uses a place geocoded from the fire's coordinates when the event has no location name", async () => {
    reverseGeocodeMock.mockResolvedValue("Haifa");
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();

    expect(await screen.findByRole("heading", { level: 1, name: "Haifa Wildfire Event" })).toBeInTheDocument();
    expect(reverseGeocodeMock).toHaveBeenCalledWith(32.731, 35.046);
    const asOf = document.querySelector(".event-details-page__as-of") as HTMLElement;
    expect(within(asOf).getByText("Event #12")).toBeInTheDocument();
  });

  it("does not repeat the event id as a badge when the title already shows it", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const asOf = document.querySelector(".event-details-page__as-of") as HTMLElement;
    expect(within(asOf).queryByText("Event #12")).not.toBeInTheDocument();
  });

  it("says 'No spread predicted' for a valid run with zero predicted cells, not raw Valid (0 predicted cells)", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        spread_predictions: [{ horizon_minutes: 30, status: "valid", predicted_at: "2026-09-17T13:25:00Z", cells: [] }],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("No spread predicted")).toBeInTheDocument();
    expect(screen.queryByText("Valid")).not.toBeInTheDocument();
    expect(screen.queryByText(/0 predicted cells/)).not.toBeInTheDocument();
  });

  it("does not repeat the event id in the Fire Event card", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const card = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(within(card).queryByText("Fire Event ID")).not.toBeInTheDocument();
    expect(within(card).queryByText("12")).not.toBeInTheDocument();
  });

  it("lays the Fire Event facts out as a compact 2-column grid including confidence", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const card = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(card.querySelector("dl.event-details-page__facts--grid")).not.toBeNull();
    for (const label of ["Status", "Coordinates", "Detected", "Confidence"]) {
      expect(within(card).getByText(label)).toBeInTheDocument();
    }
  });

  it("shows Suspected status and Critical severity as inline pill badges with strong warning/critical tones", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, status: "suspected" },
        severity: { assessment_id: 1, status: "valid", score: 90, level: "high", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Suspected")).toHaveClass("badge", "badge--warning");
    expect(screen.getByText("High")).toHaveClass("badge");
    expect(document.querySelector(".event-details-page")).not.toBeNull();
  });

  it("renders a Critical severity on a confirmed event as the solid critical badge", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        severity: { assessment_id: 1, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Critical")).toHaveClass("badge", "badge--critical");
  });

  it("keeps the map and the card column as siblings in one stretched layout row", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const layout = document.querySelector(".event-details-page__layout") as HTMLElement;
    expect(layout.querySelector(":scope > .event-details-page__section--map")).not.toBeNull();
    expect(layout.querySelector(":scope > .event-details-page__cards")).not.toBeNull();
  });

  it("shows a subtle breadcrumb link above the title and no boxed Back button in the action group", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const crumb = screen.getByRole("link", { name: "Back to Active Events" });
    expect(crumb).toHaveClass("event-details-page__breadcrumb");
    expect(crumb).toHaveTextContent("Active Events");
    expect(crumb.querySelector("svg")).not.toBeNull();
    const title = screen.getByRole("heading", { level: 1 });
    expect(crumb.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(document.querySelector(".page-header__actions")).toBeNull();
  });

  it("shows the Response Plan pill inside the Fire Event card header, not the header bar or the column bottom", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        current_response_plan: {
          plan_id: 77, generated_at: "2026-09-17T13:30:00Z", methodology: "ga", methodology_version: "1.0",
          plan_score: 90, coverage_score: 1, average_eta_seconds: 120, actions: [], uncovered_target_ids: [], baseline_comparison: null,
        },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const cta = screen.getByRole("link", { name: "View Current Response Plan" });
    expect(cta).toHaveTextContent("Response Plan");
    expect(cta.querySelector("svg")).not.toBeNull();
    expect(cta).toHaveClass("event-details-page__action--primary");
    expect(cta).not.toHaveClass("event-details-page__action--block");

    const fireCard = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(fireCard).toContainElement(cta);
    const header = cta.closest(".event-details-page__card-header") as HTMLElement;
    expect(within(header).getByRole("heading", { name: "Fire Event" })).toBeInTheDocument();
    expect((document.querySelector(".event-details-page__cards") as HTMLElement).lastElementChild).not.toBe(cta);
    expect(document.querySelector(".page-header__actions")).toBeNull();
  });

  it("renders no Response Plan action in the Fire Event card when there is no current plan", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ current_response_plan: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("link", { name: "View Current Response Plan" })).not.toBeInTheDocument();
  });

  it("marks only the 'No spread predicted' fallback as a quiet/muted state", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        spread_predictions: [
          { horizon_minutes: 30, status: "valid", predicted_at: "2026-09-17T13:25:00Z", cells: [] },
          {
            horizon_minutes: 60,
            status: "valid",
            predicted_at: "2026-09-17T13:25:00Z",
            cells: [{ latitude: 1, longitude: 1, spread_probability: 0.5, spread_risk_score: 50, reached_step: 1, reached_minutes: 5 }],
          },
        ],
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("No spread predicted").closest(".event-details-page__spread-horizon--empty")).not.toBeNull();
    expect(screen.getByText("Spread predicted").closest(".event-details-page__spread-horizon--empty")).toBeNull();
  });

  it("keeps Status and Severity badges as compact inline pills that do not stretch the row", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        severity: { assessment_id: 1, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const status = screen.getByText("Confirmed");
    const severity = screen.getByText("Critical");
    for (const badge of [status, severity]) {
      expect(badge).toHaveClass("badge");
      expect(badge.closest(".event-details-page")).not.toBeNull();
    }
  });
});
