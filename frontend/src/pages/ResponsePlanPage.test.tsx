import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { ResponsePlan } from "../types/responsePlan";
import { ResponsePlanPage } from "./ResponsePlanPage";

const { getCurrentResponsePlanMock, getResponsePlanByIdMock } = vi.hoisted(() => ({
  getCurrentResponsePlanMock: vi.fn(),
  getResponsePlanByIdMock: vi.fn(),
}));

vi.mock("../api/responsePlans", () => ({
  getCurrentResponsePlan: getCurrentResponsePlanMock,
  getResponsePlanById: getResponsePlanByIdMock,
}));

const { getEventDetailsMock } = vi.hoisted(() => ({ getEventDetailsMock: vi.fn() }));

vi.mock("../api/eventDetails", () => ({ getEventDetails: getEventDetailsMock }));

const { reverseGeocodeMock } = vi.hoisted(() => ({ reverseGeocodeMock: vi.fn() }));

vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));

vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

function makePlan(overrides: Partial<ResponsePlan> = {}): ResponsePlan {
  return {
    plan_id: 7,
    fire_event_id: 3,
    response_target_set_id: 9,
    route_planning_run_id: 11,
    generated_at: "2026-09-17T13:20:00Z",
    methodology: "genetic_algorithm",
    methodology_version: "1.0.0",
    random_seed: 42,
    status: "complete",
    is_current: true,
    metrics: { plan_score: 95.5, coverage_score: 90.0, average_eta_seconds: 150.0 },
    actions: [],
    uncovered_targets: [],
    baseline_comparison: null,
    optimization_config: null,
    no_resources_during_planning: false,
    ...overrides,
  };
}

function renderAtPath(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/events/:fireEventId/plan" element={<ResponsePlanPage />} />
        <Route path="/plans/:planId" element={<ResponsePlanPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ResponsePlanPage", () => {
  beforeEach(() => {
    getCurrentResponsePlanMock.mockReset();
    getResponsePlanByIdMock.mockReset();
    getEventDetailsMock.mockReset();
    reverseGeocodeMock.mockReset();
    reverseGeocodeMock.mockResolvedValue(null);
    getEventDetailsMock.mockResolvedValue({ detection_evidence: { satellite: [], news: [] } });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("loads the current plan using the fireEventId route param", async () => {
    getCurrentResponsePlanMock.mockResolvedValue({ plan: makePlan() });

    renderAtPath("/events/3/plan");

    await screen.findByText("Generated:");
    expect(getCurrentResponsePlanMock).toHaveBeenCalledWith(3, expect.any(AbortSignal));
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
  });

  it("loads the plan by the planId route param", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ plan_id: 55 }) });

    renderAtPath("/plans/55");

    await screen.findByText("Generated:");
    expect(getResponsePlanByIdMock).toHaveBeenCalledWith(55, expect.any(AbortSignal));
    expect(getCurrentResponsePlanMock).not.toHaveBeenCalled();
  });

  it("shows LoadingState on initial render", () => {
    getCurrentResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    renderAtPath("/events/3/plan");

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
    expect(screen.getByRole("heading", { level: 1, name: "Loading Response Plan…" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { level: 1, name: "Response Plan" })).not.toBeInTheDocument();
  });

  it("shows a safe ErrorState with Retry when the load fails", async () => {
    getCurrentResponsePlanMock.mockRejectedValue(new ApiError("SELECT * FROM secrets failed", 500));

    renderAtPath("/events/3/plan");

    expect(await screen.findByRole("heading", { name: "Unable to load the response plan." })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("SELECT * FROM secrets");
  });

  it("shows EmptyState when the current-plan API returns plan: null", async () => {
    getCurrentResponsePlanMock.mockResolvedValue({ plan: null });

    renderAtPath("/events/3/plan");

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(
      screen.getByText("There is currently no response plan available for this wildfire event."),
    ).toBeInTheDocument();
  });

  it("titles the page by its event and never shows the internal plan id", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ plan_id: 55, fire_event_id: 12 }) });

    renderAtPath("/plans/55");

    expect(await screen.findByRole("heading", { level: 1, name: "Response Plan for Event #12" })).toBeInTheDocument();
    expect(screen.queryByText(/Plan #55/)).not.toBeInTheDocument();
    expect(screen.queryByText("55")).not.toBeInTheDocument();
    expect(screen.queryByText("Plan ID")).not.toBeInTheDocument();
  });

  it("shows metadata as compact chips/badges, not a definition-list table", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ generated_at: "2026-09-17T13:20:00" }) });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    const meta = screen.getByLabelText("Plan details");
    expect(meta.querySelector("dl")).toBeNull();
    expect(within(meta).getByText("Generated:")).toBeInTheDocument();
    expect(within(meta).getByText("13:20", { exact: false })).toBeInTheDocument();
  });

  it("renders the generated timestamp", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ generated_at: "2026-09-17T13:20:00Z" }) });

    renderAtPath("/plans/7");

    await screen.findByText("Generated:");
    const time = document.querySelector("time");
    expect(time).toHaveAttribute("dateTime", "2026-09-17T13:20:00Z");
  });

  it("does not show a CURRENT tag when is_current is true", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ is_current: true }) });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    expect(screen.queryByText("CURRENT")).not.toBeInTheDocument();
    expect(screen.queryByText("SUPERSEDED")).not.toBeInTheDocument();
  });

  it("shows SUPERSEDED when is_current is false", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ is_current: false }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("SUPERSEDED")).toBeInTheDocument();
    expect(screen.queryByText("CURRENT")).not.toBeInTheDocument();
  });

  it("displays the persisted status without recomputation", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ status: "no_feasible_assignments", actions: [] }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("No feasible assignments")).toBeInTheDocument();
  });

  it("displays a partial plan status distinctly", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ status: "partial" }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("Partial")).toBeInTheDocument();
  });

  it("does not show technical badges: methodology, Calculation: Success, or a Fire event chip", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({ methodology: "genetic_algorithm", methodology_version: "2.3.0", status: "complete" }),
    });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    expect(screen.queryByText(/genetic_algorithm/)).not.toBeInTheDocument();
    expect(screen.queryByText("Calculation: Success")).not.toBeInTheDocument();
    expect(screen.queryByText("Fire event")).not.toBeInTheDocument();
  });

  it("titles target groups with the parent event's location name", async () => {
    getEventDetailsMock.mockResolvedValue({
      detection_evidence: {
        satellite: [],
        news: [{ id: 1, title: "t", summary: "s", source: "x", observed_at: "2026-09-17T13:10:00Z", location_name: "Modiin", latitude: null, longitude: null }],
      },
    });
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "s1", station_name: "One", origin: null },
            target: { response_target_id: 4, target_type: "active_fire", priority_score: 1, latitude: 1, longitude: 1 },
            route: { status: "reachable", eta_seconds: 60, distance_meters: 1, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    expect(await screen.findByRole("heading", { name: "Active fire - Modiin" })).toBeInTheDocument();
  });

  it("links back to the correct FireEvent route", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ fire_event_id: 42 }) });

    renderAtPath("/plans/7");

    const link = await screen.findByRole("link", { name: "Back to Event" });
    expect(link).toHaveAttribute("href", "/events/42");
  });

  it("puts the back link in a breadcrumb above the title, not in the header actions", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ fire_event_id: 42 }) });

    renderAtPath("/plans/7");

    const link = await screen.findByRole("link", { name: "Back to Event" });
    const heading = screen.getByRole("heading", { level: 1 });
    expect(link.closest("nav")).toHaveAttribute("aria-label", "Breadcrumb");
    expect(link.compareDocumentPosition(heading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(link.closest(".page-header__actions")).toBeNull();
  });

  it("does not call the API and shows a safe state for a non-numeric plan id", async () => {
    renderAtPath("/plans/abc");

    expect(await screen.findByRole("heading", { name: "Invalid response plan request." })).toBeInTheDocument();
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
    expect(getCurrentResponsePlanMock).not.toHaveBeenCalled();
  });

  it("does not call the API and shows a safe state for a zero plan id", async () => {
    renderAtPath("/plans/0");

    expect(await screen.findByRole("heading", { name: "Invalid response plan request." })).toBeInTheDocument();
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
  });

  it("does not call the API and shows a safe state for a negative plan id", async () => {
    renderAtPath("/plans/-5");

    expect(await screen.findByRole("heading", { name: "Invalid response plan request." })).toBeInTheDocument();
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
  });

  it("does not call the API and shows a safe state for a non-numeric fireEventId", async () => {
    renderAtPath("/events/abc/plan");

    expect(await screen.findByRole("heading", { name: "Invalid response plan request." })).toBeInTheDocument();
    expect(getCurrentResponsePlanMock).not.toHaveBeenCalled();
  });

  it("renders the Plan Metrics section from plan.metrics", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({ metrics: { plan_score: 87.3, coverage_score: 60.0, average_eta_seconds: 125 } }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("Average Travel Time");
    expect(screen.queryByText("Plan Metrics")).not.toBeInTheDocument();
    expect(screen.queryByText("87.3")).not.toBeInTheDocument();
    expect(screen.queryByText("Coverage")).not.toBeInTheDocument();
    expect(screen.queryByText("60.0%")).not.toBeInTheDocument();
    expect(screen.getByText("2m 5s")).toBeInTheDocument();
  });

  it("shows only a small time-of-day 'Generated:' string, not the date or an Algorithm Run Time badge", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ generated_at: "2026-09-17T13:33:00" }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("Generated:")).toBeInTheDocument();
    expect(document.querySelector(".response-plan-page__generated time")).toHaveTextContent("13:33");
    expect(screen.queryByText("Algorithm Run Time")).not.toBeInTheDocument();
    expect(screen.queryByText(/Sep|2026/)).not.toBeInTheDocument();
  });

  it("hides the Baseline Comparison section entirely when plan.baseline_comparison is null", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ baseline_comparison: null }) });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    expect(screen.queryByText("Baseline Comparison")).not.toBeInTheDocument();
    expect(screen.queryByText("Baseline comparison not available")).not.toBeInTheDocument();
  });

  it("splits the page into an actions column and a map column", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan() });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    const layout = screen.getByRole("region", { name: "Map of Response Plan for Event #3" }).closest(".response-plan-page__layout") as HTMLElement;
    const details = layout.querySelector(".response-plan-page__details") as HTMLElement;
    expect(within(details).getByText("Response Actions")).toBeInTheDocument();
    expect(within(details).queryByRole("region", { name: "Map of Response Plan for Event #3" })).not.toBeInTheDocument();
  });

  it("renders the Baseline Comparison section from plan.baseline_comparison when present", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        baseline_comparison: {
          baseline_score: 50.0,
          baseline_coverage_score: 40.0,
          baseline_average_eta_seconds: 300,
          score_difference: 10.0,
          improvement_percentage: 20.0,
        },
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("Baseline Comparison");
    expect(screen.getByText("50.0")).toBeInTheDocument();
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    expect(screen.getByText("5m 0s")).toBeInTheDocument();
    expect(screen.getByText("+10.0")).toBeInTheDocument();
    expect(screen.getByText("+20.0%")).toBeInTheDocument();
  });

  it("shows the partial-plan notice for a partial plan", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ status: "partial" }) });

    renderAtPath("/plans/7");

    expect(
      await screen.findByText("This response plan does not cover all response targets."),
    ).toBeInTheDocument();
  });

  it("shows the no-resources notice when no_resources_during_planning is true", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ no_resources_during_planning: true }) });

    renderAtPath("/plans/7");

    expect(
      await screen.findByText("No firefighting resources were available in the planning snapshot used for this plan."),
    ).toBeInTheDocument();
  });

  it("shows both partial and no-resources notices together when the backend returns that combination", async () => {
    getResponsePlanByIdMock.mockResolvedValue(
      { plan: makePlan({ status: "partial", no_resources_during_planning: true }) },
    );

    renderAtPath("/plans/7");

    expect(
      await screen.findByText("This response plan does not cover all response targets."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No firefighting resources were available in the planning snapshot used for this plan."),
    ).toBeInTheDocument();
  });

  it("renders the Response Actions section with every backend action in order", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "reachable", eta_seconds: 120, distance_meters: 800, node_path: null, path_coordinates: null },
          },
          {
            resource: { resource_id: "engine-2", station_id: "station-2", station_name: null, origin: null },
            target: { response_target_id: 2, target_type: "predicted_risk", priority_score: 0.4, latitude: 2, longitude: 2 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("Response Actions");
    const resourceIds = screen.getAllByText(/^engine-[12]$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["engine-1", "engine-2"]);
  });

  it("shows the Response Actions empty state when plan.actions is empty", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ actions: [] }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("No response actions")).toBeInTheDocument();
  });

  it("renders Uncovered Targets after Response Actions, with every backend target", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        uncovered_targets: [
          { response_target_id: 5, target_type: "active_fire", priority_score: 0.3, latitude: 40.0, longitude: 41.0 },
          { response_target_id: 2, target_type: null, priority_score: null, latitude: null, longitude: null },
        ],
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("Uncovered Targets");
    const headings = screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent);
    const actionsIndex = headings.indexOf("Response Actions");
    const uncoveredIndex = headings.indexOf("Uncovered Targets");
    expect(actionsIndex).toBeGreaterThanOrEqual(0);
    expect(uncoveredIndex).toBeGreaterThan(actionsIndex);

    expect(screen.getByText("#5")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
  });

  it("shows the all-covered state when plan.uncovered_targets is empty", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ uncovered_targets: [] }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("All response targets are covered by this plan.")).toBeInTheDocument();
  });

  it("does not show internal optimization (GA) details in the operational UI", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({ methodology: "genetic_algorithm", methodology_version: "1.0.0", random_seed: 4242 }),
    });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    expect(screen.queryByText("Optimization Details")).not.toBeInTheDocument();
    expect(screen.queryByText("Technical details")).not.toBeInTheDocument();
    expect(screen.queryByText("4242")).not.toBeInTheDocument();
  });

  it("still shows loading/error/empty/metrics/actions behavior unchanged alongside the new sections", async () => {
    getCurrentResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    renderAtPath("/events/3/plan");

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it("selecting a Response Action highlights only that action, without reordering the list", async () => {
    const user = userEvent.setup();
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "reachable", eta_seconds: 120, distance_meters: 800, node_path: null, path_coordinates: null },
          },
          {
            resource: { resource_id: "engine-2", station_id: "station-2", station_name: null, origin: null },
            target: { response_target_id: 2, target_type: "predicted_risk", priority_score: 0.4, latitude: 2, longitude: 2 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");
    await screen.findByText("Response Actions");

    // The first listed action is highlighted by default.
    expect(screen.getAllByRole("button", { name: "Selected" })).toHaveLength(1);
    const selectButtons = screen.getAllByRole("button", { name: "Highlight on map" });
    expect(selectButtons).toHaveLength(1);

    await user.click(selectButtons[0]);

    expect(await screen.findByRole("button", { name: "Selected" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getAllByRole("button", { name: "Highlight on map" })).toHaveLength(1);
    const resourceIds = screen.getAllByText(/^engine-[12]$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["engine-1", "engine-2"]);
  });

  // -------------------------------------------------------------------------
  // US 6.2 map integration (Integration Task 2)
  // -------------------------------------------------------------------------

  it("renders the shared US 6.2 map", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan() });

    renderAtPath("/plans/7");

    expect(await screen.findByRole("region", { name: "Map of Response Plan for Event #3" })).toBeInTheDocument();
    expect(screen.getByTestId("tile-layer")).toBeInTheDocument();
  });

  it("draws a reachable action's persisted route with coordinate order preserved", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: { latitude: 32.0, longitude: 35.0 } },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 32.9, longitude: 35.9 },
            route: {
              status: "reachable",
              eta_seconds: 120,
              distance_meters: 800,
              node_path: [1, 2, 3],
              path_coordinates: [
                { latitude: 32.0, longitude: 35.0 },
                { latitude: 32.5, longitude: 35.5 },
                { latitude: 32.9, longitude: 35.9 },
              ],
            },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    const polyline = await screen.findByTestId("polyline");
    expect(JSON.parse(polyline.getAttribute("data-positions")!)).toEqual([
      [32.0, 35.0],
      [32.5, 35.5],
      [32.9, 35.9],
    ]);
  });

  it("renders the origin marker at the resource's persisted coordinates", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: { latitude: 32.1, longitude: 35.2 } },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 32.9, longitude: 35.9 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    const markers = await screen.findAllByTestId("marker");
    const origin = markers.find((marker) => marker.getAttribute("data-lat") === "32.1");
    expect(origin).toHaveAttribute("data-lng", "35.2");
  });

  it("renders the target marker at the target's persisted coordinates", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 32.9, longitude: 35.9 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    const marker = await screen.findByTestId("marker");
    expect(marker).toHaveAttribute("data-lat", "32.9");
    expect(marker).toHaveAttribute("data-lng", "35.9");
  });

  it("selecting a Response Action highlights only its own route", async () => {
    const user = userEvent.setup();
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: { latitude: 32.0, longitude: 35.0 } },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 32.9, longitude: 35.9 },
            route: {
              status: "reachable",
              eta_seconds: 120,
              distance_meters: 800,
              node_path: null,
              path_coordinates: [
                { latitude: 32.0, longitude: 35.0 },
                { latitude: 32.9, longitude: 35.9 },
              ],
            },
          },
          {
            resource: { resource_id: "engine-2", station_id: "station-2", station_name: "North", origin: { latitude: 33.0, longitude: 36.0 } },
            target: { response_target_id: 2, target_type: "active_fire", priority_score: 0.5, latitude: 33.9, longitude: 36.9 },
            route: {
              status: "reachable",
              eta_seconds: 90,
              distance_meters: 500,
              node_path: null,
              path_coordinates: [
                { latitude: 33.0, longitude: 36.0 },
                { latitude: 33.9, longitude: 36.9 },
              ],
            },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");
    const selectButtons = await screen.findAllByRole("button", { name: "Highlight on map" });

    await user.click(selectButtons[1]);

    const polylines = await screen.findAllByTestId("polyline");
    const engine1Route = polylines.find(
      (line) => JSON.parse(line.getAttribute("data-positions")!)[0][0] === 32.0,
    )!;
    const engine2Route = polylines.find(
      (line) => JSON.parse(line.getAttribute("data-positions")!)[0][0] === 33.0,
    )!;
    expect(engine2Route.getAttribute("data-color")).not.toBe(engine1Route.getAttribute("data-color"));
  });

  it("draws no route line for an unreachable action", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("engine-1");
    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });

  it("draws no route line for an unmappable action", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "unmappable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("engine-1");
    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });

  it("draws no fake route line for a reachable action with no persisted path_coordinates", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "reachable", eta_seconds: 60, distance_meters: 400, node_path: [1, 2], path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("engine-1");
    expect(screen.queryByTestId("polyline")).not.toBeInTheDocument();
  });

  it("does not crash and keeps the action visible when its resource has no persisted origin", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: null },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    expect((await screen.findAllByText("engine-1")).length).toBeGreaterThan(0);
    // Exactly one marker (the target) - no origin marker was fabricated.
    expect(screen.getAllByTestId("marker")).toHaveLength(1);
  });

  it("does not crash and keeps the action visible when its target has no persisted coordinates", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "station-1", station_name: "Central", origin: { latitude: 32.0, longitude: 35.0 } },
            target: { response_target_id: 1, target_type: null, priority_score: null, latitude: null, longitude: null },
            route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    expect((await screen.findAllByText("engine-1")).length).toBeGreaterThan(0);
    // Exactly one marker (the origin) - no target marker was fabricated.
    expect(screen.getAllByTestId("marker")).toHaveLength(1);
  });

  it("performs no network call while rendering the map beyond the one plan fetch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan() });

    renderAtPath("/plans/7");
    await screen.findByText("Generated:");

    expect(fetchSpy).not.toHaveBeenCalled();
    fetchSpy.mockRestore();
  });

  it("renders whichever route the user highlights as the solid thick line, and the primary as dashed once deselected", async () => {
    const user = userEvent.setup();
    const route = (a: number, b: number) => ({
      status: "reachable" as const,
      eta_seconds: 60,
      distance_meters: 100,
      node_path: null,
      path_coordinates: [
        { latitude: a, longitude: b },
        { latitude: a + 0.1, longitude: b + 0.1 },
      ],
    });
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "s1", station_name: "One", origin: { latitude: 32, longitude: 35 } },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 32.1, longitude: 35.1 },
            route: route(32, 35),
          },
          {
            resource: { resource_id: "engine-2", station_id: "s2", station_name: "Two", origin: { latitude: 33, longitude: 36 } },
            target: { response_target_id: 1, target_type: "active_fire", priority_score: 0.9, latitude: 33.1, longitude: 36.1 },
            route: route(33, 36),
          },
        ],
      }),
    });

    renderAtPath("/plans/7");
    await screen.findByText("Response Actions");

    let lines = screen.getAllByTestId("polyline");
    expect(lines[0]).toHaveAttribute("data-weight", "6");
    expect(lines[1]).toHaveAttribute("data-dash", "6 8");

    await user.click(screen.getByRole("button", { name: "Highlight on map" }));

    lines = screen.getAllByTestId("polyline");
    expect(lines[1]).toHaveAttribute("data-weight", "6");
    expect(lines[1]).not.toHaveAttribute("data-dash");
    expect(lines[1]).toHaveAttribute("data-opacity", "1");
    expect(lines[0]).toHaveAttribute("data-weight", "3");
    expect(lines[0]).toHaveAttribute("data-dash", "6 8");
  });

  it("titles a target group with its reverse-geocoded place, using the event location until it resolves", async () => {
    let resolveGeocode: (value: string | null) => void = () => {};
    reverseGeocodeMock.mockReturnValue(new Promise<string | null>((resolve) => (resolveGeocode = resolve)));
    getEventDetailsMock.mockResolvedValue({
      detection_evidence: {
        satellite: [],
        news: [{ id: 1, title: "t", summary: "s", source: "x", observed_at: "2026-09-17T13:10:00Z", location_name: "Modiin", latitude: null, longitude: null }],
      },
    });
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "s1", station_name: "One", origin: null },
            target: { response_target_id: 223, target_type: "active_fire", priority_score: 1, latitude: 32.7323, longitude: 35.0373 },
            route: { status: "reachable", eta_seconds: 60, distance_meters: 1, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    // Fetching: fall back to the parent event's location, never "Target #223".
    expect(await screen.findByRole("heading", { name: "Active fire - Modiin" })).toBeInTheDocument();
    expect(reverseGeocodeMock).toHaveBeenCalledWith(32.7323, 35.0373);

    resolveGeocode("Haifa");
    expect(await screen.findByRole("heading", { name: "Active fire - Haifa" })).toBeInTheDocument();
    expect(screen.queryByText(/Target #223/)).not.toBeInTheDocument();
  });

  it("titles the page with the event's location and shows the event id as a badge beside Generated", async () => {
    getEventDetailsMock.mockResolvedValue({
      detection_evidence: {
        satellite: [],
        news: [{ id: 1, title: "t", summary: "s", source: "x", observed_at: "2026-09-17T13:10:00Z", location_name: "Haifa Subdistrict", latitude: null, longitude: null }],
      },
    });
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ fire_event_id: 12 }) });

    renderAtPath("/plans/7");

    expect(await screen.findByRole("heading", { level: 1, name: "Response Plan for Haifa Subdistrict" })).toBeInTheDocument();
    const meta = screen.getByLabelText("Plan details");
    expect(within(meta).getByText("Generated:")).toBeInTheDocument();
    expect(within(meta).getByText("Event #12")).toBeInTheDocument();
  });

  it("falls back to the event id in the title, without a duplicate badge, when no location is known", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ fire_event_id: 12 }) });

    renderAtPath("/plans/7");

    expect(await screen.findByRole("heading", { level: 1, name: "Response Plan for Event #12" })).toBeInTheDocument();
    expect(within(screen.getByLabelText("Plan details")).queryByText("Event #12")).not.toBeInTheDocument();
  });

  it("titles the page with a place geocoded from the plan's targets when the event has no location name", async () => {
    reverseGeocodeMock.mockResolvedValue("Haifa");
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        fire_event_id: 12,
        actions: [
          {
            resource: { resource_id: "engine-1", station_id: "s1", station_name: "One", origin: null },
            target: { response_target_id: 4, target_type: "active_fire", priority_score: 1, latitude: 32.7, longitude: 35.0 },
            route: { status: "reachable", eta_seconds: 60, distance_meters: 1, node_path: null, path_coordinates: null },
          },
        ],
      }),
    });

    renderAtPath("/plans/7");

    expect(await screen.findByRole("heading", { level: 1, name: "Response Plan for Haifa" })).toBeInTheDocument();
    expect(within(screen.getByLabelText("Plan details")).getByText("Event #12")).toBeInTheDocument();
  });
});
