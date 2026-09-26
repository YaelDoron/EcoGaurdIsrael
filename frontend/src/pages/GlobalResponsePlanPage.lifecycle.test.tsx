import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GlobalPlanCoverage, GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { GlobalResponsePlanPage } from "./GlobalResponsePlanPage";

const { getCurrentGlobalResponsePlanMock, getCurrentSimulationRunMock } = vi.hoisted(() => ({
  getCurrentGlobalResponsePlanMock: vi.fn(),
  getCurrentSimulationRunMock: vi.fn(),
}));

vi.mock("../api/simulation", () => ({ getCurrentSimulationRun: getCurrentSimulationRunMock }));
vi.mock("../api/globalResponsePlan", () => ({ getCurrentGlobalResponsePlan: getCurrentGlobalResponsePlanMock }));
vi.mock("../api/eventDetails", () => ({ getEventDetails: vi.fn().mockReturnValue(new Promise(() => {})) }));
vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: vi.fn().mockResolvedValue(null) }));
vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

const PLAN = {
  run_id: 77,
  started_at: "2026-09-20T10:00:00Z",
  completed_at: "2026-09-20T10:00:05Z",
  status: "completed",
  metrics: { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0 },
  shortage: { total_required: 0, total_desired: 0, total_assigned: 0, unmet_required: 0, unmet_desired: 0 },
  optimization_config: null,
  events: [],
};

function coverage(state: GlobalPlanCoverage["state"], eligible: number[], covered: number[]): GlobalPlanCoverage {
  const pending = eligible.filter((id) => !covered.includes(id));
  return {
    state,
    eligible_fire_event_ids: eligible,
    covered_fire_event_ids: covered,
    pending_fire_event_ids: pending,
    unplannable_fire_event_ids: [],
    eligible_count: eligible.length,
    covered_count: covered.length,
  };
}

function response(plan: unknown, cov: GlobalPlanCoverage): GlobalResponsePlanResponse {
  return { as_of: "2026-09-20T10:00:10Z", plan, coverage: cov } as unknown as GlobalResponsePlanResponse;
}

function renderPage() {
  return render(
    <MemoryRouter>
      <GlobalResponsePlanPage />
    </MemoryRouter>,
  );
}

describe("GlobalResponsePlanPage plan lifecycle", () => {
  beforeEach(() => {
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-live", state: "running" });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("no confirmed fire: the only case that says No response plan available", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(response(null, coverage("none", [], [])));

    renderPage();

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(screen.queryByText(/Generating response plan/)).not.toBeInTheDocument();
  });

  it("a confirmed fire without a plan yet shows Generating, never No response plan", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(response(null, coverage("generating", [11], [])));

    renderPage();

    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();
    expect(screen.getByText(/1 confirmed fire is being processed: Event #11\./)).toBeInTheDocument();
    expect(screen.queryByText("No response plan available")).not.toBeInTheDocument();
  });

  it("a plan covering 1 of 2 confirmed fires stays visible with an Updating notice", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(response(PLAN, coverage("updating", [11, 12], [11])));

    renderPage();

    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
    expect(screen.getByText("Updating response plan...")).toBeInTheDocument();
    expect(screen.getByText(/Currently covers 1 of 2 confirmed fires\. Still being planned: Event #12\./)).toBeInTheDocument();
  });

  it("the Updating notice disappears on its own once the plan covers every confirmed fire", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(response(PLAN, coverage("updating", [11, 12], [11])))
      .mockResolvedValue(response(PLAN, coverage("current", [11, 12], [11, 12])));

    renderPage();
    expect(await screen.findByText("Updating response plan...")).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(5000);

    await vi.waitFor(() => expect(screen.queryByText("Updating response plan...")).not.toBeInTheDocument());
    expect(screen.getByText(/Last updated/)).toBeInTheDocument();
  });

  it("Generating turns into the plan automatically, without leaving the page", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(response(null, coverage("generating", [11], [])))
      .mockResolvedValue(response(PLAN, coverage("current", [11], [11])));

    renderPage();
    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(5000);

    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
    expect(screen.queryByText(/Generating response plan/)).not.toBeInTheDocument();
  });

  it("SUSPECTED-only fires (not response-eligible) never suggest a plan is being generated", async () => {
    getCurrentGlobalResponsePlanMock.mockResolvedValue(response(null, coverage("none", [], [])));

    renderPage();

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(screen.queryByText(/Generating|Updating/)).not.toBeInTheDocument();
  });
});
