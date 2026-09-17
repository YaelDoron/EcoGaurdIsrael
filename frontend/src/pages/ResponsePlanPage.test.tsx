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
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("loads the current plan using the fireEventId route param", async () => {
    getCurrentResponsePlanMock.mockResolvedValue({ plan: makePlan() });

    renderAtPath("/events/3/plan");

    await screen.findByText("Plan ID");
    expect(getCurrentResponsePlanMock).toHaveBeenCalledWith(3);
    expect(getResponsePlanByIdMock).not.toHaveBeenCalled();
  });

  it("loads the plan by the planId route param", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ plan_id: 55 }) });

    renderAtPath("/plans/55");

    await screen.findByText("Plan ID");
    expect(getResponsePlanByIdMock).toHaveBeenCalledWith(55);
    expect(getCurrentResponsePlanMock).not.toHaveBeenCalled();
  });

  it("shows LoadingState on initial render", () => {
    getCurrentResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));

    renderAtPath("/events/3/plan");

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
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

  it("displays the correct Plan ID and FireEvent ID", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ plan_id: 55, fire_event_id: 12 }) });

    renderAtPath("/plans/55");

    await screen.findByText("Plan ID");
    expect(screen.getByText("55")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("renders the generated timestamp", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ generated_at: "2026-09-17T13:20:00Z" }) });

    renderAtPath("/plans/7");

    await screen.findByText("Plan ID");
    const time = document.querySelector("time");
    expect(time).toHaveAttribute("dateTime", "2026-09-17T13:20:00Z");
  });

  it("shows CURRENT when is_current is true", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ is_current: true }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("CURRENT")).toBeInTheDocument();
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

  it("displays methodology and methodology version", async () => {
    getResponsePlanByIdMock.mockResolvedValue(
      { plan: makePlan({ methodology: "genetic_algorithm", methodology_version: "2.3.0" }) },
    );

    renderAtPath("/plans/7");

    expect(await screen.findByText("genetic_algorithm (v2.3.0)")).toBeInTheDocument();
  });

  it("links back to the correct FireEvent route", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ fire_event_id: 42 }) });

    renderAtPath("/plans/7");

    const link = await screen.findByRole("link", { name: "Back to Event" });
    expect(link).toHaveAttribute("href", "/events/42");
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

    await screen.findByText("Plan Metrics");
    expect(screen.getByText("87.3")).toBeInTheDocument();
    expect(screen.getByText("60.0%")).toBeInTheDocument();
    expect(screen.getByText("2m 5s")).toBeInTheDocument();
  });

  it('shows "Baseline comparison not available" when plan.baseline_comparison is null', async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ baseline_comparison: null }) });

    renderAtPath("/plans/7");

    expect(await screen.findByText("Baseline comparison not available")).toBeInTheDocument();
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

  it("renders Optimization Details in the successful plan view, with methodology/version/seed always shown", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({ methodology: "genetic_algorithm", methodology_version: "1.0.0", random_seed: 42 }),
    });

    renderAtPath("/plans/7");

    await screen.findByText("Optimization Details");
    expect(screen.getByText("Technical details")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("shows the optimization config unavailable message when optimization_config is null", async () => {
    getResponsePlanByIdMock.mockResolvedValue({ plan: makePlan({ optimization_config: null }) });

    renderAtPath("/plans/7");

    expect(
      await screen.findByText("Detailed optimization configuration is not available for this response plan."),
    ).toBeInTheDocument();
  });

  it("displays the persisted optimization config fields when present", async () => {
    getResponsePlanByIdMock.mockResolvedValue({
      plan: makePlan({
        optimization_config: {
          population_size: 50,
          generation_count: 100,
          mutation_rate: 0.1,
          crossover_rate: 0.8,
          eta_reference_seconds: 600,
          initial_assignment_probability: 0.5,
          tournament_size: 9,
          elitism_count: 4,
        },
      }),
    });

    renderAtPath("/plans/7");

    const heading = await screen.findByText("Optimization Details");
    const section = heading.closest("section") as HTMLElement;
    expect(within(section).getByText("50")).toBeInTheDocument();
    expect(within(section).getByText("100")).toBeInTheDocument();
    expect(within(section).getByText("9")).toBeInTheDocument();
    expect(within(section).getByText("4")).toBeInTheDocument();
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

    const selectButtons = screen.getAllByRole("button", { name: "Highlight on map" });
    expect(selectButtons).toHaveLength(2);

    await user.click(selectButtons[1]);

    expect(await screen.findByRole("button", { name: "Selected" })).toHaveAttribute("aria-pressed", "true");
    const resourceIds = screen.getAllByText(/^engine-[12]$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["engine-1", "engine-2"]);
  });
});
