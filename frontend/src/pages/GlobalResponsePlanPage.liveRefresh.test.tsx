import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GlobalPlanCoverage, GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { GlobalResponsePlanPage } from "./GlobalResponsePlanPage";

/**
 * One Global Response Plan page stays mounted for the whole test (no
 * navigation, no remount) while the backend state changes underneath it.
 */
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

function state(
  lifecycle: GlobalPlanCoverage["state"],
  eligible: number[],
  covered: number[],
  monitoring: number[] = [],
  withPlan = covered.length > 0,
): GlobalResponsePlanResponse {
  return {
    as_of: "2026-09-20T10:00:10Z",
    plan: withPlan ? PLAN : null,
    coverage: {
      state: lifecycle,
      eligible_fire_event_ids: eligible,
      covered_fire_event_ids: covered,
      pending_fire_event_ids: eligible.filter((id) => !covered.includes(id)),
      unplannable_fire_event_ids: [],
      monitoring_fire_event_ids: monitoring,
      eligible_count: eligible.length,
      covered_count: covered.length,
    },
  } as unknown as GlobalResponsePlanResponse;
}

const RUNNING = { run_id: "run-1", state: "running" };

function renderPage() {
  return render(
    <MemoryRouter>
      <GlobalResponsePlanPage />
    </MemoryRouter>,
  );
}

async function nextPoll() {
  await vi.advanceTimersByTimeAsync(5000);
}

describe("GlobalResponsePlanPage live refresh while the page stays open", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    getCurrentGlobalResponsePlanMock.mockReset();
    getCurrentSimulationRunMock.mockReset();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    window.sessionStorage.clear();
  });

  it("walks none -> generating -> current 1/1 -> updating 1/2 -> current 2/2 without remounting", async () => {
    getCurrentSimulationRunMock.mockResolvedValue(RUNNING);
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(state("none", [], [], [1, 2])) // A and B SUSPECTED
      .mockResolvedValueOnce(state("generating", [1], [], [2])) // A CONFIRMED
      .mockResolvedValueOnce(state("current", [1], [1], [2])) // plan A saved
      .mockResolvedValueOnce(state("updating", [1, 2], [1])) // B CONFIRMED
      .mockResolvedValue(state("current", [1, 2], [1, 2])); // plan A+B

    renderPage();

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(screen.getByText(/2 suspected fires are being monitored/)).toBeInTheDocument();

    await nextPoll();
    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();
    expect(screen.getByText(/1 confirmed fire is being processed: Event #1\./)).toBeInTheDocument();

    await nextPoll();
    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
    expect(screen.queryByText(/Generating|Updating/)).not.toBeInTheDocument();

    await nextPoll();
    expect(await screen.findByText("Updating response plan...")).toBeInTheDocument();
    expect(screen.getByText(/Currently covers 1 of 2 confirmed fires\. Still being planned: Event #2\./)).toBeInTheDocument();

    await nextPoll();
    await vi.waitFor(() => expect(screen.queryByText("Updating response plan...")).not.toBeInTheDocument());
    expect(screen.getByText(/Last updated/)).toBeInTheDocument();
  });

  it("a page opened before any run starts discovers the run and its confirmed fire later", async () => {
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: null, state: "idle" });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("generating", [1], []));

    renderPage();
    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(getCurrentGlobalResponsePlanMock).not.toHaveBeenCalled(); // gated: nothing to show yet

    getCurrentSimulationRunMock.mockResolvedValue(RUNNING); // Start Simulation (e.g. from another tab)
    await nextPoll();

    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();
  });

  it("never reads or shows the previous run's data while the new run is PREPARING (reset not committed)", async () => {
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-2", state: "preparing" });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("current", [9], [9])); // the OLD run's plan

    renderPage();
    expect(await screen.findByText("Preparing simulation...")).toBeInTheDocument();
    await nextPoll();
    expect(getCurrentGlobalResponsePlanMock).not.toHaveBeenCalled();
    expect(screen.queryByText(/Last updated/)).not.toBeInTheDocument();

    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-2", state: "running" });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("none", [], [])); // after the reset
    await nextPoll();

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(screen.queryByText(/Last updated/)).not.toBeInTheDocument();
  });

  it("keeps checking in the none state while a run is live (no fire confirmed yet)", async () => {
    getCurrentSimulationRunMock.mockResolvedValue(RUNNING);
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("none", [], [], [1]));

    renderPage();
    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    await nextPoll();
    await nextPoll();

    expect(getCurrentGlobalResponsePlanMock.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it("stops polling once the run has ended and the plan is current", async () => {
    window.sessionStorage.setItem("ecoguard.demoSession.runId", "run-1");
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-1", state: "stopped" });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("current", [1], [1]));

    renderPage();
    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
    const callsAfterLoad = getCurrentGlobalResponsePlanMock.mock.calls.length;

    await nextPoll();
    await nextPoll();
    await nextPoll();

    expect(getCurrentGlobalResponsePlanMock.mock.calls.length).toBe(callsAfterLoad);
  });

  it("after the run ended, an unfinished plan is re-checked for a bounded time only", async () => {
    window.sessionStorage.setItem("ecoguard.demoSession.runId", "run-1");
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-1", state: "stopped" });
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("updating", [1, 2], [1]));

    renderPage();
    expect(await screen.findByText("Updating response plan...")).toBeInTheDocument();
    for (let i = 0; i < 20; i += 1) {
      await nextPoll();
    }
    const calls = getCurrentGlobalResponsePlanMock.mock.calls.length;
    await nextPoll();
    await nextPoll();

    expect(calls).toBeGreaterThan(3); // it did keep checking for a while
    expect(calls).toBeLessThanOrEqual(14); // ... but not forever
    expect(getCurrentGlobalResponsePlanMock.mock.calls.length).toBe(calls);
  });

  it("SUSPECTED-only fires say why there is no plan, and never say Generating", async () => {
    getCurrentSimulationRunMock.mockResolvedValue(RUNNING);
    getCurrentGlobalResponsePlanMock.mockResolvedValue(state("none", [], [], [5, 6]));

    renderPage();

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(
      screen.getByText(/No fire is confirmed yet\. 2 suspected fires are being monitored - response plans are generated only for confirmed fires\./),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Generating/)).not.toBeInTheDocument();
  });
});
