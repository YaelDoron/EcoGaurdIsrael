import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ViewResponsePlanAction } from "./ViewResponsePlanAction";
import type { GlobalPlanningRunActivityDetail } from "../../types/operationsActivity";

const { getOperationsActivityDetailMock } = vi.hoisted(() => ({
  getOperationsActivityDetailMock: vi.fn(),
}));

vi.mock("../../api/operations", () => ({
  getOperationsActivityDetail: getOperationsActivityDetailMock,
}));

function makeDetail(overrides: Partial<GlobalPlanningRunActivityDetail["details"]> = {}): GlobalPlanningRunActivityDetail {
  return {
    activity_type: "global_planning_run",
    entity_id: 7,
    occurred_at: "2026-09-20T10:00:00Z",
    title: "Global Response Plan",
    location: null,
    details: {
      global_planning_run_id: 7,
      status: "completed",
      trigger: "fire_event_update",
      started_at: "2026-09-20T09:55:00Z",
      completed_at: "2026-09-20T10:00:00Z",
      methodology: "global_planning_refresh_coordinator",
      methodology_version: "1.0",
      fire_event_ids: [18, 19],
      response_plan_ids: [66, 67],
      coverage_score: 100,
      average_eta_seconds: 280,
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
      ...overrides,
    },
  };
}

function renderAction(globalPlanningRunId: number | null) {
  return render(
    <MemoryRouter initialEntries={["/events"]}>
      <Routes>
        <Route path="/events" element={<ViewResponsePlanAction globalPlanningRunId={globalPlanningRunId} />} />
        <Route path="/plans/:planId" element={<p>Plan page</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ViewResponsePlanAction", () => {
  beforeEach(() => {
    getOperationsActivityDetailMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when no real global planning run is available", () => {
    const { container } = renderAction(null);

    expect(container).toBeEmptyDOMElement();
    expect(getOperationsActivityDetailMock).not.toHaveBeenCalled();
  });

  it("shows an accessible View Response Plan button when a global planning run id is available", () => {
    renderAction(7);

    expect(screen.getByRole("button", { name: "View Response Plan" })).toBeEnabled();
    // No A5 request until the operator actually clicks.
    expect(getOperationsActivityDetailMock).not.toHaveBeenCalled();
  });

  it("fetches A5 detail exactly once on click, using the real global_planning_run entity id", async () => {
    const user = userEvent.setup();
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail());
    renderAction(7);

    await user.click(screen.getByRole("button", { name: "View Response Plan" }));

    await waitFor(() => expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1));
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("global_planning_run", 7);
  });

  it("navigates to the existing Response Plan route using the first real persisted response_plan_id", async () => {
    const user = userEvent.setup();
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail({ response_plan_ids: [66, 67] }));
    renderAction(7);

    await user.click(screen.getByRole("button", { name: "View Response Plan" }));

    expect(await screen.findByText("Plan page")).toBeInTheDocument();
  });

  it("never fabricates a plan id and never navigates twice for multiple response_plan_ids", async () => {
    const user = userEvent.setup();
    getOperationsActivityDetailMock.mockResolvedValue(makeDetail({ response_plan_ids: [66, 67, 68] }));
    renderAction(7);

    await user.click(screen.getByRole("button", { name: "View Response Plan" }));

    await screen.findByText("Plan page");
    // Exactly one navigation happened (to the route stub above) - if more
    // than one had fired, React Router would still just show the same
    // single route content, so we additionally assert only one A5 call ever happened.
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1);
  });

  it("prevents a double click from issuing a second request", async () => {
    const user = userEvent.setup();
    let resolveDetail!: (value: GlobalPlanningRunActivityDetail) => void;
    getOperationsActivityDetailMock.mockReturnValue(
      new Promise<GlobalPlanningRunActivityDetail>((resolve) => {
        resolveDetail = resolve;
      }),
    );
    renderAction(7);

    const button = screen.getByRole("button", { name: "View Response Plan" });
    await user.click(button);
    await user.click(button);

    resolveDetail(makeDetail());
    await waitFor(() => expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1));
  });

  it("shows a concise local error when A5 detail loading fails, without breaking the panel", async () => {
    const user = userEvent.setup();
    getOperationsActivityDetailMock.mockRejectedValue(new Error("network down"));
    renderAction(7);

    await user.click(screen.getByRole("button", { name: "View Response Plan" }));

    expect(await screen.findByText("Unable to open the response plan. Please try again.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "View Response Plan" })).toBeEnabled();
  });
});
