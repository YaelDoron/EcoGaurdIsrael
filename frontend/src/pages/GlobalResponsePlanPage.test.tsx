import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { GlobalResponsePlanPage } from "./GlobalResponsePlanPage";

const { getCurrentGlobalResponsePlanMock } = vi.hoisted(() => ({
  getCurrentGlobalResponsePlanMock: vi.fn(),
}));

// A live demo run, so the demo-session gate (hooks/demoSession.ts) shows the plan.
vi.mock("../api/simulation", () => ({
  getCurrentSimulationRun: vi.fn().mockResolvedValue({ run_id: "run-live", state: "running" }),
}));
vi.mock("../api/globalResponsePlan", () => ({
  getCurrentGlobalResponsePlan: getCurrentGlobalResponsePlanMock,
}));

const { getEventDetailsMock } = vi.hoisted(() => ({ getEventDetailsMock: vi.fn() }));

vi.mock("../api/eventDetails", () => ({ getEventDetails: getEventDetailsMock }));

const { reverseGeocodeMock } = vi.hoisted(() => ({ reverseGeocodeMock: vi.fn() }));

vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));

vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

function makeResponse(overrides: Partial<GlobalResponsePlanResponse> = {}): GlobalResponsePlanResponse {
  return {
    as_of: "2026-09-20T09:00:00Z",
    plan: {
      run_id: 77,
      started_at: "2026-09-20T08:00:00Z",
      completed_at: "2026-09-20T08:05:00Z",
      status: "completed",
      metrics: { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0 },
      shortage: { total_required: 3, total_desired: 6, total_assigned: 4, unmet_required: 0, unmet_desired: 2 },
      optimization_config: null,
      events: [
        {
          fire_event_id: 101,
          response_plan_id: 501,
          severity_level: "high",
          severity_score: 70.0,
          minimum_resources: 2,
          desired_resources: 4,
          assigned_resources: 3,
          coverage_score: 85.0,
          average_eta_seconds: 140.0,
          actions: [
            {
              resource: {
                resource_id: "engine-1",
                station_id: "station-1",
                station_name: "Central Station",
                origin: { latitude: 32.0, longitude: 34.8 },
              },
              target: {
                response_target_id: 1,
                target_type: "active_fire",
                priority_score: 0.9,
                latitude: 32.7,
                longitude: 35.0,
              },
              route: {
                status: "reachable",
                eta_seconds: 120.0,
                distance_meters: 800.0,
                node_path: [1, 2],
                path_coordinates: null,
              },
            },
          ],
          uncovered_targets: [],
        },
        {
          fire_event_id: 202,
          response_plan_id: 502,
          severity_level: "moderate",
          severity_score: 40.0,
          minimum_resources: 1,
          desired_resources: 2,
          assigned_resources: 1,
          coverage_score: 60.0,
          average_eta_seconds: 200.0,
          actions: [],
          uncovered_targets: [],
        },
      ],
    },
    ...overrides,
  };
}

function LocationProbe() {
  return <div data-testid="search">{useLocation().search}</div>;
}

function renderAtPath(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/global-response-plan" element={<GlobalResponsePlanPage />} />
      </Routes>
      <LocationProbe />
    </MemoryRouter>,
  );
}

describe("GlobalResponsePlanPage", () => {
  beforeEach(() => {
    getEventDetailsMock.mockReset();
    getEventDetailsMock.mockRejectedValue(new Error("no details"));
    reverseGeocodeMock.mockReset();
    reverseGeocodeMock.mockResolvedValue(null);
    getCurrentGlobalResponsePlanMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows a loading state on initial render", () => {
    getCurrentGlobalResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    renderAtPath("/global-response-plan");

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it("shows a generic error state with Retry on failure, and recovers on retry", async () => {
    const user = userEvent.setup();
    getCurrentGlobalResponsePlanMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByRole("button", { name: "Retry" });
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText(/Last updated:/)).toBeInTheDocument();
  });

  it("shows an empty state when no generation has materialized yet", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse({ plan: null }));

    renderAtPath("/global-response-plan");

    expect(await screen.findByText("No materialized generation yet")).toBeInTheDocument();
  });

  it("shows only a clean 'Last updated' line - no run id, algorithm time, or success badge", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    const meta = screen.getByLabelText("Run details");
    expect(meta).toHaveTextContent(/Last updated: \d{1,2} \w+ \d{4}, \d{2}:\d{2}/);
    expect(meta.querySelector("time")).toHaveAttribute("dateTime", "2026-09-20T08:05:00Z");
    expect(screen.queryByText("Run ID")).not.toBeInTheDocument();
    expect(screen.queryByText("77")).not.toBeInTheDocument();
    expect(screen.queryByText("Algorithm Run Time")).not.toBeInTheDocument();
    expect(screen.queryByText(/Calculation/)).not.toBeInTheDocument();
  });

  it("flags a run that did not complete cleanly with a badge", async () => {
    const base = makeResponse();
    getCurrentGlobalResponsePlanMock.mockResolvedValue(
      makeResponse({ plan: { ...(base.plan as NonNullable<typeof base.plan>), status: "partial" } }),
    );

    renderAtPath("/global-response-plan");

    expect(await screen.findByText("Partial plan")).toBeInTheDocument();
  });

  it("renders the global metrics panel with coverage, ETA and a resource summary - no fitness score", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    expect(screen.getByText("Global Plan Metrics")).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Global Plan Metrics" })).getByText("Coverage")).toBeInTheDocument();
    expect(screen.queryByText("Fitness Score")).not.toBeInTheDocument();
    expect(screen.queryByText("91.5")).not.toBeInTheDocument();
    expect(screen.getByText("Assigned: 4")).toBeInTheDocument();
    expect(screen.getByLabelText("Resource shortage")).toHaveTextContent("Unmet: 2");
  });

  it("does not render internal optimization (GA) details", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    expect(screen.queryByText("Optimization Details")).not.toBeInTheDocument();
  });

  it("shows run metadata as a flex row of chips, not a table", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    const meta = screen.getByLabelText("Run details");
    expect(meta.querySelector("dl")).toBeNull();
    expect(within(meta).getByText(/Last updated:/)).toBeInTheDocument();
  });

  it("renders the map", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    expect(screen.getByRole("region", { name: "Global Response Map" })).toBeInTheDocument();
  });

  it("renders the event group list with every materialized event", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    expect(screen.getByRole("link", { name: "Event #101" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Event #202" })).toBeInTheDocument();
  });

  it("titles each event group with its persisted location name instead of the raw id", async () => {
    // Already English: the backend ingestion pipeline translates
    // location_name before it is ever saved - the frontend displays it
    // as-is and never re-translates it.
    getEventDetailsMock.mockImplementation((id: number) =>
      Promise.resolve({
        detection_evidence: { satellite: [], news: [{ location_name: id === 101 ? "Haifa Subdistrict" : "Haifa" }] },
      }),
    );
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");

    expect(await screen.findByRole("link", { name: "Haifa Subdistrict" })).toHaveAttribute("href", "/events/101");
    expect(await screen.findByRole("link", { name: "Haifa" })).toHaveAttribute("href", "/events/202");
  });

  it("falls back to a place reverse-geocoded from the event's own target coordinates when there is no news location name", async () => {
    getEventDetailsMock.mockResolvedValue({ detection_evidence: { satellite: [], news: [] } });
    reverseGeocodeMock.mockImplementation((latitude: number, longitude: number) =>
      Promise.resolve(latitude === 32.7 && longitude === 35.0 ? "Nof HaGalil" : null),
    );
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");

    expect(await screen.findByRole("link", { name: "Nof HaGalil" })).toHaveAttribute("href", "/events/101");
    expect(reverseGeocodeMock).toHaveBeenCalledWith(32.7, 35.0);
  });

  it("still falls back to Event #id when neither a news location name nor any target coordinate is available", async () => {
    getEventDetailsMock.mockResolvedValue({ detection_evidence: { satellite: [], news: [] } });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    // Event 202 has no actions/uncovered_targets, so no coordinate exists to geocode.
    expect(screen.getByRole("link", { name: "Event #202" })).toBeInTheDocument();
  });

  it("splits the layout into a details column and a map column", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    const layout = screen.getByRole("region", { name: "Global Response Map" }).closest(".global-response-plan-page__layout") as HTMLElement;
    const details = layout.querySelector(".global-response-plan-page__details") as HTMLElement;
    expect(within(details).getByText(/Last updated:/)).toBeInTheDocument();
    expect(within(details).getByRole("link", { name: "Event #101" })).toBeInTheDocument();
    expect(within(details).queryByRole("region", { name: "Global Response Map" })).not.toBeInTheDocument();
  });

  it("hides the Uncovered Targets card when every event is fully covered", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");
    await screen.findByText(/Last updated:/);

    expect(screen.queryByText("Uncovered Targets")).not.toBeInTheDocument();
  });

  describe("focusEventId", () => {
    it("reads focusEventId from the URL and marks the matching group as focused", async () => {
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan?focusEventId=101");
      await screen.findByText(/Last updated:/);

      const focusedGroup = screen.getByRole("link", { name: "Event #101" }).closest("article") as HTMLElement;
      expect(within(focusedGroup).getByRole("button", { name: "Focused" })).toBeInTheDocument();
    });

    it("does not trigger a new fetch when focusEventId is present in the URL", async () => {
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan?focusEventId=101");
      await screen.findByText(/Last updated:/);

      expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
    });

    it("focusing an event via the group list updates the highlighted group, without a new fetch", async () => {
      const user = userEvent.setup();
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan");
      await screen.findByText(/Last updated:/);

      const group202 = screen.getByRole("link", { name: "Event #202" }).closest("article") as HTMLElement;
      await user.click(within(group202).getByRole("button", { name: "Focus on map" }));

      expect(within(group202).getByRole("button", { name: "Focused" })).toBeInTheDocument();
      expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
    });

    it("clicking the Focused button again releases the focus (toggle)", async () => {
      const user = userEvent.setup();
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan?focusEventId=101");
      await screen.findByText(/Last updated:/);
      const group101 = screen.getByRole("link", { name: "Event #101" }).closest("article") as HTMLElement;

      await user.click(within(group101).getByRole("button", { name: "Focused" }));

      expect(within(group101).getByRole("button", { name: "Focus on map" })).toHaveAttribute("aria-pressed", "false");
      expect(screen.getByTestId("search")).toHaveTextContent(/^$/);
    });

    it("toggling the focused group off resets every group to unfocused, without a new fetch", async () => {
      const user = userEvent.setup();
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan?focusEventId=101");
      await screen.findByText(/Last updated:/);
      await user.click(screen.getByRole("button", { name: "Focused" }));

      expect(screen.getByTestId("search").textContent).toBe("");
      expect(screen.queryByRole("button", { name: "Focused" })).not.toBeInTheDocument();
      expect(screen.getAllByRole("button", { name: "Focus on map" })).toHaveLength(2);
      expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(1);
    });

    it("has no Show All Fires button", async () => {
      getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

      renderAtPath("/global-response-plan?focusEventId=101");
      await screen.findByText(/Last updated:/);

      expect(screen.queryByRole("button", { name: "Show All Fires" })).not.toBeInTheDocument();
    });
  });
});

describe("GlobalResponsePlanPage breadcrumb", () => {
  beforeEach(() => {
    getCurrentGlobalResponsePlanMock.mockReset();
    getEventDetailsMock.mockReset();
    getEventDetailsMock.mockRejectedValue(new Error("no details"));
  });

  it("shows a subtle 'Back to Active Fires' link above the title, not in the header actions", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(makeResponse());

    renderAtPath("/global-response-plan");

    const link = await screen.findByRole("link", { name: "Back to Active Fires" });
    const heading = screen.getByRole("heading", { level: 1, name: "Global Response Plan" });
    expect(link).toHaveAttribute("href", "/events");
    expect(link.closest("nav")).toHaveAttribute("aria-label", "Breadcrumb");
    expect(link.compareDocumentPosition(heading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(link.closest(".page-header__actions")).toBeNull();
  });

  it("keeps the breadcrumb while loading", () => {
    getCurrentGlobalResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    renderAtPath("/global-response-plan");

    expect(screen.getByRole("link", { name: "Back to Active Fires" })).toBeInTheDocument();
  });
});
