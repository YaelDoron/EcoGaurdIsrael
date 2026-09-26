import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OperationsActivityDrawer } from "./OperationsActivityDrawer";
import { ApiError } from "../../api/errors";
import type { OperationsActivityDetailResponse } from "../../types/operationsActivity";
import type { OperationsActivityFeedItem } from "../../types/operationsOverview";

const { getOperationsActivityDetailMock } = vi.hoisted(() => ({
  getOperationsActivityDetailMock: vi.fn(),
}));

vi.mock("../../api/operations", () => ({
  getOperationsActivityDetail: getOperationsActivityDetailMock,
}));

function selectedItem(overrides: Partial<OperationsActivityFeedItem> = {}): OperationsActivityFeedItem {
  return {
    activity_id: "fire_event:42",
    entity_id: 42,
    occurred_at: "2026-09-20T11:00:00Z",
    title: "Fire Event #42",
    location: { latitude: 32.7, longitude: 35.0 },
    activity_type: "fire_event",
    preview: { status: "confirmed", confidence: 0.84 },
    ...overrides,
  } as OperationsActivityFeedItem;
}

function renderDrawer(item: OperationsActivityFeedItem | null, onClose = vi.fn()) {
  return render(
    <MemoryRouter>
      <OperationsActivityDrawer selectedItem={item} onClose={onClose} />
    </MemoryRouter>,
  );
}

describe("OperationsActivityDrawer", () => {
  beforeEach(() => {
    getOperationsActivityDetailMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when no item is selected", () => {
    const { container } = renderDrawer(null);
    expect(container).toBeEmptyDOMElement();
  });

  it.each([
    ["ECOGUARD_AI_HYBRID_DETECTION", "Peak AI likelihood", "84%"],
    ["ECOGUARD_MULTI_SOURCE_DETECTION", "Confidence", "84%"],
  ])("labels a %s fire event's detection value as %s (never AI 'confidence')", async (methodology, label, value) => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "fire_event",
      entity_id: 42,
      occurred_at: "2026-09-20T11:00:00Z",
      title: "Fire Event #42",
      location: null,
      details: {
        fire_event_id: 42,
        status: "confirmed",
        detection_confidence: 0.84,
        detected_at: "2026-09-20T11:00:00Z",
        updated_at: "2026-09-20T11:05:00Z",
        created_at: "2026-09-20T10:59:30Z",
        latitude: 32.7,
        longitude: 35.0,
        methodology,
        methodology_version: "1.0",
        location_name: "Carmel Demo Area",
        evidence: { satellite_hotspot_ids: [], news_report_ids: [] },
        latest_severity: null,
      },
    } satisfies OperationsActivityDetailResponse);

    renderDrawer(selectedItem());

    const term = await screen.findByText(label);
    expect(term.nextElementSibling).toHaveTextContent(value);
    if (label === "Peak AI likelihood") {
      expect(screen.queryByText("Confidence")).not.toBeInTheDocument();
    }
  });

  it("fetches only the selected item's detail, exactly once", async () => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "fire_event",
      entity_id: 42,
      occurred_at: "2026-09-20T11:00:00Z",
      title: "Fire Event #42",
      location: null,
      details: {
        fire_event_id: 42,
        status: "confirmed",
        detection_confidence: 0.84,
        detected_at: "2026-09-20T11:00:00Z",
        updated_at: "2026-09-20T11:05:00Z",
        created_at: "2026-09-20T10:59:30Z",
        latitude: 32.7,
        longitude: 35.0,
        methodology: "ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version: "1.0",
        location_name: "Carmel Demo Area",
        evidence: { satellite_hotspot_ids: [], news_report_ids: [] },
        latest_severity: null,
      },
    } satisfies OperationsActivityDetailResponse);

    renderDrawer(selectedItem());

    await waitFor(() => expect(screen.getByText("View Fire Event")).toBeInTheDocument());
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1);
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("fire_event", 42, expect.anything());
  });

  it("links to the existing Event Details route for a FireEvent activity", async () => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "fire_event",
      entity_id: 42,
      occurred_at: "2026-09-20T11:00:00Z",
      title: "Fire Event #42",
      location: null,
      details: {
        fire_event_id: 42,
        status: "confirmed",
        detection_confidence: 0.84,
        detected_at: "2026-09-20T11:00:00Z",
        updated_at: "2026-09-20T11:05:00Z",
        created_at: "2026-09-20T10:59:30Z",
        latitude: 32.7,
        longitude: 35.0,
        methodology: "ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version: "1.0",
        location_name: "Carmel Demo Area",
        evidence: { satellite_hotspot_ids: [], news_report_ids: [] },
        latest_severity: null,
      },
    } satisfies OperationsActivityDetailResponse);

    renderDrawer(selectedItem());

    const link = await screen.findByText("View Fire Event");
    expect(link.closest("a")).toHaveAttribute("href", "/events/42");
  });

  it("renders a link per real response plan id for a Global Planning Run activity, never fabricated", async () => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "global_planning_run",
      entity_id: 7,
      occurred_at: "2026-09-20T11:00:00Z",
      title: "Global response plan generated",
      location: null,
      details: {
        global_planning_run_id: 7,
        status: "completed",
        trigger: "manual",
        started_at: "2026-09-20T10:55:00Z",
        completed_at: "2026-09-20T11:00:00Z",
        methodology: "GA_MULTI_INCIDENT",
        methodology_version: "1.0",
        fire_event_ids: [1, 2],
        response_plan_ids: [101, 102],
        coverage_score: 88.4,
        average_eta_seconds: 620,
        shortage_total_required: null,
        shortage_total_desired: null,
        shortage_total_assigned: null,
        shortage_unmet_required: null,
        shortage_unmet_desired: null,
        ga_population_size: null,
        ga_generation_count: null,
        ga_mutation_rate: null,
        ga_crossover_rate: null,
        members: [],
      },
    } satisfies OperationsActivityDetailResponse);

    renderDrawer(selectedItem({ activity_type: "global_planning_run", entity_id: 7 }));

    expect(await screen.findByText("View Response Plan #101")).toHaveAttribute("href", "/plans/101");
    expect(screen.getByText("View Response Plan #102")).toHaveAttribute("href", "/plans/102");
  });

  it("shows a not-found state for a 404", async () => {
    getOperationsActivityDetailMock.mockRejectedValue(new ApiError("Not found", 404));

    renderDrawer(selectedItem());

    expect(await screen.findByText("Not found")).toBeInTheDocument();
  });

  it("shows a retryable error state on failure", async () => {
    getOperationsActivityDetailMock.mockRejectedValue(new Error("network down"));

    renderDrawer(selectedItem());

    expect(await screen.findByText("Unable to load details")).toBeInTheDocument();
  });

  it("calls onClose when the close control is activated", async () => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "fire_event",
      entity_id: 42,
      occurred_at: "2026-09-20T11:00:00Z",
      title: "Fire Event #42",
      location: null,
      details: {
        fire_event_id: 42,
        status: "confirmed",
        detection_confidence: 0.84,
        detected_at: "2026-09-20T11:00:00Z",
        updated_at: "2026-09-20T11:05:00Z",
        created_at: "2026-09-20T10:59:30Z",
        latitude: 32.7,
        longitude: 35.0,
        methodology: "ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version: "1.0",
        location_name: "Carmel Demo Area",
        evidence: { satellite_hotspot_ids: [], news_report_ids: [] },
        latest_severity: null,
      },
    } satisfies OperationsActivityDetailResponse);

    const user = userEvent.setup();
    const onClose = vi.fn();
    renderDrawer(selectedItem(), onClose);

    await user.click(await screen.findByRole("button", { name: "Close activity details" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders the persisted weather-conditions detail (never a recalculated score)", async () => {
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "weather_conditions",
      entity_id: 1,
      occurred_at: "2026-09-20T10:11:42Z",
      title: "Weather Conditions — Galilee Demo Area",
      location: { latitude: 32.915, longitude: 35.345 },
      details: {
        fire_danger_assessment_id: 1,
        area_name: "Galilee Demo Area",
        fire_danger_level: "very_high",
        assessed_at: "2026-09-20T10:11:55Z",
        readings: [
          {
            station_id: 3,
            station_name: "Galilee Station",
            observation_id: 10,
            observed_at: "2026-09-20T10:11:00Z",
            temperature: 34,
            relative_humidity: 19,
            wind_speed: 28,
            wind_gust: null,
          },
        ],
      },
    } satisfies OperationsActivityDetailResponse);

    renderDrawer(
      selectedItem({
        activity_id: "weather_conditions:1",
        entity_id: 1,
        activity_type: "weather_conditions",
        title: "Weather Conditions — Galilee Demo Area",
        preview: {
          area_name: "Galilee Demo Area",
          fire_danger_level: "very_high",
          fire_danger_assessment_id: 1,
          temperature_c: 34,
          relative_humidity_pct: 19,
          wind_speed_kmh: 28,
          wind_gust_kmh: null,
        },
      }),
    );

    expect(await screen.findByText("Galilee Demo Area")).toBeInTheDocument();
    expect(screen.getByText("Very high")).toBeInTheDocument();
    expect(screen.getByText("Galilee Station")).toBeInTheDocument();
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("weather_conditions", 1, expect.anything());
  });
});
