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
    ml_assessment: null,
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
    expect(scoped.getByText("0.72")).toBeInTheDocument();
    expect(fireEventSection.querySelectorAll("time")).toHaveLength(1); // detected_at only
  });

  // -------------------------------------------------------------------------
  // ML assessment (ML Task 6 frontend exposure)
  // -------------------------------------------------------------------------

  it("shows Rule Score and AI Model Score when an ML assessment is available", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, status: "confirmed" },
        ml_assessment: { available: true, mode: "shadow", rule_confidence: 0.8, model_score: 0.993351, agreement: "agree_fire" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("Confirmed")).toBeInTheDocument();
    expect(scoped.getByText("Rule Score")).toBeInTheDocument();
    expect(scoped.getByText("0.80")).toBeInTheDocument();
    expect(scoped.getByText("AI Model Score")).toBeInTheDocument();
    expect(scoped.getByText("0.99")).toBeInTheDocument();
    // Never the raw unrounded float.
    expect(scoped.queryByText("0.993351")).not.toBeInTheDocument();
  });

  it("omits AI Model Score (but keeps Rule Score) when there is no ml_assessment row", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ ml_assessment: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("Rule Score")).toBeInTheDocument();
    expect(scoped.queryByText("AI Model Score")).not.toBeInTheDocument();
  });

  it('shows "Unavailable" (never 0) for AI Model Score when the ML classifier failed', async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        ml_assessment: { available: false, mode: "shadow", rule_confidence: 0.8, model_score: null, agreement: "ml_unavailable" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("AI Model Score")).toBeInTheDocument();
    expect(scoped.getByText("Unavailable")).toBeInTheDocument();
    expect(scoped.queryByText("0.00")).not.toBeInTheDocument();
    expect(scoped.queryByText("0")).not.toBeInTheDocument();
  });

  it("falls back to fire_event.detection_confidence for Rule Score when there is no ml_assessment", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, detection_confidence: 0.65 },
        ml_assessment: null,
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(within(fireEventSection).getByText("0.65")).toBeInTheDocument();
  });

  it("prefers ml_assessment.rule_confidence over fire_event.detection_confidence when both are present", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, detection_confidence: 0.5 },
        ml_assessment: { available: true, mode: "shadow", rule_confidence: 0.8, model_score: 0.6, agreement: "agree_fire" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("0.80")).toBeInTheDocument();
    expect(scoped.queryByText("0.50")).not.toBeInTheDocument();
  });

  it("never renders an AI Confidence/Fire Probability/Certainty label", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        ml_assessment: { available: true, mode: "shadow", rule_confidence: 0.8, model_score: 0.99, agreement: "agree_fire" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    for (const forbidden of ["AI Confidence", "Fire Probability", "AI Probability of Fire", "Certainty"]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });

  it("does not add a second Detection Status / AI section elsewhere on the page", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        ml_assessment: { available: true, mode: "shadow", rule_confidence: 0.8, model_score: 0.99, agreement: "agree_fire" },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getAllByText("AI Model Score")).toHaveLength(1);
    expect(screen.queryByRole("heading", { name: /AI/i })).not.toBeInTheDocument();
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
    expect(scoped.getByText("1 nearby area reached the propagation threshold")).toBeInTheDocument();
    expect(scoped.getByText("60 min horizon")).toBeInTheDocument();
    expect(scoped.getByText("Insufficient data")).toBeInTheDocument();
    // Historical insufficient_data row without a stored reason.
    expect(scoped.getByText("The stored prediction does not specify why.")).toBeInTheDocument();
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
          actions: [
            {
              resource_id: "R1",
              station_id: "S1",
              response_target_id: 1,
              target_type: "active_fire",
              target_priority: 100,
              eta_seconds: 120,
              route_distance_meters: 1000,
              node_path: [1, 2],
            },
          ],
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

  it("shows the persisted location name (not raw coordinates or Event #id) in the fire marker's popup", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        detection_evidence: {
          satellite: [],
          // Already English: the backend ingestion pipeline translates
          // location_name before it is ever saved - the frontend displays
          // it as-is and never re-translates it.
          news: [
            {
              id: 1,
              title: "t",
              summary: "s",
              source: "src",
              observed_at: "2026-09-17T13:10:00Z",
              location_name: "Galilee",
              latitude: null,
              longitude: null,
            },
          ],
        },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const popup = screen.getAllByTestId("popup")[0];
    expect(within(popup).getByText("Galilee Wildfire")).toBeInTheDocument();
    expect(within(popup).queryByText("Event #12")).not.toBeInTheDocument();
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

  it("shows the persisted location name in the title exactly as saved, never mixing languages", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        detection_evidence: {
          satellite: [],
          // Backend-translated before persistence - see
          // NewsMonitoringAgent.translate_report / SimulationEventExecutor.
          news: [
            {
              id: 1,
              title: "t",
              summary: "s",
              source: "src",
              observed_at: "2026-09-17T13:10:00Z",
              location_name: "Galilee",
              latitude: null,
              longitude: null,
            },
          ],
        },
      }),
    );

    renderPage();

    const title = await screen.findByRole("heading", { level: 1 });
    expect(title).toHaveTextContent("Galilee Wildfire Event");
    expect(title.textContent).not.toMatch(/[֐-׿]/);
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

  it("lays the Fire Event facts out as a compact 2-column grid including rule score", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult());

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const card = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(card.querySelector("dl.event-details-page__facts--grid")).not.toBeNull();
    for (const label of ["Status", "Coordinates", "Detected", "Rule Score"]) {
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
          plan_score: 90, coverage_score: 1, average_eta_seconds: 120,
          actions: [
            {
              resource_id: "R1", station_id: "S1", response_target_id: 1, target_type: "active_fire",
              target_priority: 100, eta_seconds: 120, route_distance_meters: 1000, node_path: [1, 2],
            },
          ],
          uncovered_target_ids: [], baseline_comparison: null,
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

  it("renders no Response Plan link in the Fire Event card when there is no current plan", async () => {
    getEventDetailsMock.mockResolvedValue(makeResult({ current_response_plan: null }));

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("link", { name: "View Current Response Plan" })).not.toBeInTheDocument();
  });

  it("shows a disabled/pending Response Plan button (not nothing) on first load, before a plan exists yet, for a still-active event", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({ fire_event: { ...makeResult().fire_event, status: "confirmed" }, current_response_plan: null }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const pending = screen.getByText("Response Plan pending…");
    expect(pending).toHaveAttribute("aria-disabled", "true");
    expect(screen.queryByRole("link", { name: "View Current Response Plan" })).not.toBeInTheDocument();
  });

  // -------------------------------------------------------------------------
  // Final Fire Detection semantics (Task 9C): SUSPECTED = monitoring, CONFIRMED = response eligible
  // -------------------------------------------------------------------------

  const AI_ASSESSMENT = {
    available: true,
    mode: "ai_hybrid_v5",
    rule_confidence: 0.875,
    model_score: 0.53,
    agreement: "agree_fire",
    model_name: "fire_detection_hgb_v5",
    model_version: "5.0",
    policy_version: "ai_hybrid_policy_v5.0",
    policy_status: "suspected",
    history_available: true,
    satellite_pass_count: 2,
    current_satellite_pixel_count: 3,
  };

  it("shows a monitoring state - not 'Response Plan pending' - for a SUSPECTED event with no plan", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({ fire_event: { ...makeResult().fire_event, status: "suspected" }, current_response_plan: null }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByTestId("monitoring-state")).toHaveTextContent("Monitoring - awaiting additional evidence");
    expect(screen.queryByText(/Response Plan/)).not.toBeInTheDocument();
    expect(screen.queryByText(/pending/i)).not.toBeInTheDocument();
    expect(screen.getByTestId("status-help")).toHaveTextContent(
      "SUSPECTED means the system detected evidence consistent with a possible wildfire but does not yet have enough corroborating evidence to confirm it.",
    );
  });

  it("explains CONFIRMED as high AI likelihood plus corroborating current satellite evidence - not certainty (AI mode)", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, status: "confirmed" },
        ml_assessment: { ...AI_ASSESSMENT, policy_status: "confirmed" },
        current_response_plan: null,
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const help = screen.getByTestId("status-help");
    expect(help).toHaveTextContent(/high AI likelihood together with corroborating current satellite evidence/);
    expect(help).toHaveTextContent(/not certainty/);
    expect(screen.getByText("Response Plan pending…")).toBeInTheDocument(); // CONFIRMED keeps the existing experience
    expect(screen.queryByTestId("monitoring-state")).not.toBeInTheDocument();
  });

  it("shows the AI explainability fields for an ai_hybrid_v5 event, labelled as likelihood (never certainty)", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: {
          ...makeResult().fire_event,
          status: "suspected",
          detection_confidence: 0.64,
          methodology: "ECOGUARD_AI_HYBRID_DETECTION",
        },
        ml_assessment: AI_ASSESSMENT,
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    const scoped = within(fireEventSection);
    expect(scoped.getByText("Latest AI likelihood").nextSibling).toHaveTextContent("53%");
    expect(scoped.getByText("Peak AI likelihood").nextSibling).toHaveTextContent("64%");
    expect(scoped.queryByText("Rule Score")).not.toBeInTheDocument(); // the rule result is diagnostics only in this mode

    const aiSection = screen.getByRole("heading", { name: "AI Assessment" }).closest("section") as HTMLElement;
    const ai = within(aiSection);
    expect(ai.getByText("AI Hybrid V5")).toBeInTheDocument();
    expect(ai.getByText("Suspected")).toBeInTheDocument();
    expect(ai.getByText("Satellite passes").nextSibling).toHaveTextContent("2");
    expect(ai.getByText("Current satellite pixels").nextSibling).toHaveTextContent("3");
    expect(ai.getByText("Detection mode").nextSibling).toHaveTextContent("AI Hybrid V5");
    expect(ai.getByText("Latest AI verdict").nextSibling).toHaveTextContent("Suspected");
    // Implementation details are not part of the operator view (the API still returns them).
    expect(ai.queryByText(/Event history used/i)).not.toBeInTheDocument();
    expect(ai.queryByText("Model")).not.toBeInTheDocument();
    expect(ai.queryByText("Policy")).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").not.toMatch(/HGB V5|AI Hybrid Policy v5\.0|ai_hybrid_v5|Event history used/);

    const pageText = document.body.textContent ?? "";
    expect(pageText).not.toMatch(/certain/i); // "certainty" never appears
    expect(pageText).not.toMatch(/joblib|\.json|C:\|\/models\/|satellite_frp_trend|feature_names/); // no paths, no ML feature names
  });

  it.each(["suspected", "confirmed"] as const)(
    "shows Latest 79% and Peak 87% AI likelihood as distinct values for a %s ai_hybrid_v5 event",
    async (status) => {
      getEventDetailsMock.mockResolvedValue(
        makeResult({
          fire_event: {
            ...makeResult().fire_event,
            status,
            detection_confidence: 0.87,
            methodology: "ECOGUARD_AI_HYBRID_DETECTION",
          },
          ml_assessment: { ...AI_ASSESSMENT, model_score: 0.79, policy_status: status },
        }),
      );

      renderPage();
      await screen.findByText("Data as of:", { exact: false });

      const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
      const scoped = within(fireEventSection);
      expect(scoped.getByText("Latest AI likelihood").nextSibling).toHaveTextContent("79%");
      expect(scoped.getByText("Peak AI likelihood").nextSibling).toHaveTextContent("87%");
      expect(scoped.queryByText("Rule Score")).not.toBeInTheDocument();
      expect(scoped.queryByText("AI Model Score")).not.toBeInTheDocument();
      expect(scoped.queryByText("0.79")).not.toBeInTheDocument();
    },
  );

  it("does not present a rule-created confidence as 'Peak AI likelihood' even when the persisted mode is AI", async () => {
    // e.g. an event first decided by the rule path and later re-assessed by the AI path: detection_confidence is a RULE score.
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: {
          ...makeResult().fire_event,
          status: "confirmed",
          detection_confidence: 0.875,
          methodology: "ECOGUARD_MULTI_SOURCE_DETECTION",
        },
        ml_assessment: { ...AI_ASSESSMENT, model_score: 0.68 },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.getByText("Latest AI likelihood").nextSibling).toHaveTextContent("68%");
    expect(screen.queryByText("Peak AI likelihood")).not.toBeInTheDocument();
    expect(screen.queryByText("88%")).not.toBeInTheDocument();
  });

  it("CONFIRMED AI event: likelihoods stay in the Fire Event card (not repeated), the AI Assessment block is trimmed", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: {
          ...makeResult().fire_event,
          status: "confirmed",
          detection_confidence: 0.87,
          methodology: "ECOGUARD_AI_HYBRID_DETECTION",
        },
        ml_assessment: {
          ...AI_ASSESSMENT,
          model_score: 0.83,
          policy_status: "confirmed",
          satellite_pass_count: 4,
          current_satellite_pixel_count: 3,
        },
      }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    const fireEventSection = screen.getByRole("heading", { name: "Fire Event" }).closest("section") as HTMLElement;
    expect(within(fireEventSection).getByText("Latest AI likelihood").nextSibling).toHaveTextContent("83%");
    expect(within(fireEventSection).getByText("Peak AI likelihood").nextSibling).toHaveTextContent("87%");

    const aiSection = screen.getByRole("heading", { name: "AI Assessment" }).closest("section") as HTMLElement;
    const ai = within(aiSection);
    expect(ai.getByText("Detection mode").nextSibling).toHaveTextContent("AI Hybrid V5");
    expect(ai.getByText("Latest AI verdict").nextSibling).toHaveTextContent("Confirmed");
    expect(ai.getByText("Satellite passes").nextSibling).toHaveTextContent("4");
    expect(ai.getByText("Current satellite pixels").nextSibling).toHaveTextContent("3");
    expect(ai.queryByText(/AI likelihood/)).not.toBeInTheDocument(); // one source of truth for the numbers
    expect(aiSection.textContent ?? "").not.toMatch(/HGB|Policy|history used|%|ai_hybrid_v5/i);
    expect(screen.getByTestId("status-help")).toHaveTextContent(/not certainty/);
  });

  it("does not render an AI Assessment section for a non-AI mode", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({ ml_assessment: { available: true, mode: "shadow", rule_confidence: 0.8, model_score: 0.9, agreement: "agree_fire" } }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByRole("heading", { name: "AI Assessment" })).not.toBeInTheDocument();
    expect(screen.getByText("Rule Score")).toBeInTheDocument();
  });

  it("renders no Response Plan action at all for a resolved event with no plan - none will ever be generated", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({ fire_event: { ...makeResult().fire_event, status: "resolved" }, current_response_plan: null }),
    );

    renderPage();
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByText(/Response Plan/)).not.toBeInTheDocument();
  });

  it("shows an active, clickable 'No Resources Available' button - not the stuck pending state - when a plan exists but allocated zero resources", async () => {
    getEventDetailsMock.mockResolvedValue(
      makeResult({
        fire_event: { ...makeResult().fire_event, fire_event_id: 789, status: "confirmed" },
        current_response_plan: {
          plan_id: 900,
          generated_at: "2026-09-17T13:30:00Z",
          methodology: "global_genetic_resource_allocation",
          methodology_version: "1.0",
          plan_score: 0,
          coverage_score: 0,
          average_eta_seconds: null,
          actions: [],
          uncovered_target_ids: [1],
          baseline_comparison: null,
        },
      }),
    );

    renderPage("789");
    await screen.findByText("Data as of:", { exact: false });

    expect(screen.queryByText("Response Plan pending…")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "View Current Response Plan" })).not.toBeInTheDocument();

    const cta = screen.getByRole("link", { name: "View Response Plan (No Resources Available)" });
    expect(cta).toHaveTextContent("No Resources Available");
    expect(cta).toHaveAttribute("href", "/events/789/plan");
    expect(cta).toHaveClass("event-details-page__action--warning");
    expect(cta).not.toHaveAttribute("aria-disabled");
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

  describe("fire-spread methodology 1.1 wording", () => {
    const PREDICTED_AT = "2026-09-17T13:25:00Z";

    function cell(spread_probability: number, latitude: number) {
      return {
        latitude,
        longitude: 35.05,
        spread_probability,
        spread_risk_score: spread_probability * 100,
        reached_step: 1,
        reached_minutes: 5,
      };
    }

    function ring(probability: number, count: number) {
      return Array.from({ length: count }, (_, index) => cell(probability, 32.7 + index * 0.001));
    }

    async function renderSpreadSection(spread_predictions: EventDetailsResult["spread_predictions"]) {
      getEventDetailsMock.mockResolvedValue(makeResult({ spread_predictions }));
      renderPage();
      await screen.findByText("Data as of:", { exact: false });
      const section = screen.getByRole("heading", { name: "Spread Prediction" }).closest("section") as HTMLElement;
      return within(section);
    }

    it("presents a valid risk-only ring as spread risk only, never as predicted spread", async () => {
      const scoped = await renderSpreadSection([
        { horizon_minutes: 30, status: "valid", predicted_at: PREDICTED_AT, cells: ring(0.38, 8) },
      ]);

      expect(scoped.getByText("Spread risk only")).toBeInTheDocument();
      expect(
        scoped.getByText("8 nearby areas with predicted spread risk; none reached the propagation threshold"),
      ).toBeInTheDocument();
      expect(scoped.queryByText("Spread predicted")).not.toBeInTheDocument();
      expect(scoped.queryByText(/predicted cells/)).not.toBeInTheDocument();
      expect(scoped.queryByText(/\bcells?\b/)).not.toBeInTheDocument();
    });

    it("counts spreading and risk-only cells separately for a mixed prediction", async () => {
      const scoped = await renderSpreadSection([
        {
          horizon_minutes: 30,
          status: "valid",
          predicted_at: PREDICTED_AT,
          cells: [...ring(0.62, 3), ...ring(0.3, 5).map((c, i) => ({ ...c, latitude: 32.8 + i * 0.001 }))],
        },
      ]);

      expect(scoped.getByText("Spread predicted")).toBeInTheDocument();
      expect(scoped.getByText("3 nearby areas reached the propagation threshold · 5 additional areas show spread risk")).toBeInTheDocument();
    });

    it("treats a probability exactly at the propagation threshold as spreading", async () => {
      const scoped = await renderSpreadSection([
        { horizon_minutes: 30, status: "valid", predicted_at: PREDICTED_AT, cells: [cell(0.5, 32.7)] },
      ]);

      expect(scoped.getByText("Spread predicted")).toBeInTheDocument();
      expect(scoped.getByText("1 nearby area reached the propagation threshold")).toBeInTheDocument();
    });

    it("shows the stored reason for missing vegetation", async () => {
      const scoped = await renderSpreadSection([
        {
          horizon_minutes: 30,
          status: "insufficient_data",
          predicted_at: PREDICTED_AT,
          cells: [],
          insufficient_data_reason: "missing_vegetation",
        },
      ]);

      expect(scoped.getByText("Insufficient data")).toBeInTheDocument();
      expect(scoped.getByText("Vegetation data required by the spread model is unavailable.")).toBeInTheDocument();
    });

    it("words unsupported vegetation differently from missing vegetation and never mentions Copernicus", async () => {
      const scoped = await renderSpreadSection([
        {
          horizon_minutes: 30,
          status: "insufficient_data",
          predicted_at: PREDICTED_AT,
          cells: [],
          insufficient_data_reason: "unsupported_vegetation",
        },
      ]);

      expect(
        scoped.getByText("The vegetation category here is not supported by the current spread model."),
      ).toBeInTheDocument();
      expect(scoped.queryByText(/unavailable/)).not.toBeInTheDocument();
      expect(screen.queryByText(/copernicus/i)).not.toBeInTheDocument();
    });

    it("falls back to an explicit unknown-reason note for a historical row without a reason", async () => {
      const scoped = await renderSpreadSection([
        {
          horizon_minutes: 60,
          status: "insufficient_data",
          predicted_at: PREDICTED_AT,
          cells: [],
          insufficient_data_reason: null,
        },
      ]);

      expect(scoped.getByText("Insufficient data")).toBeInTheDocument();
      expect(scoped.getByText("The stored prediction does not specify why.")).toBeInTheDocument();
    });

    it("counts identical 30/60-minute cell locations once in the spread layer total", async () => {
      await renderSpreadSection([
        { horizon_minutes: 30, status: "valid", predicted_at: PREDICTED_AT, cells: ring(0.38, 8) },
        { horizon_minutes: 60, status: "valid", predicted_at: PREDICTED_AT, cells: ring(0.38, 8) },
      ]);

      const spreadToggle = screen.getByRole("checkbox", { name: /predicted spread/i });
      const count = spreadToggle.closest("label")?.querySelector(".layer-controls__count");
      expect(count?.textContent).toBe("8");
    });
  });
});
