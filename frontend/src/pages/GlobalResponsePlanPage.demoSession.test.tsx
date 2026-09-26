import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";
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

const STALE_PLAN: GlobalResponsePlanResponse = {
  as_of: "2026-09-20T10:00:00Z",
  plan: {
    run_id: 77,
    started_at: "2026-09-20T10:00:00Z",
    completed_at: "2026-09-20T10:00:05Z",
    status: "completed",
    metrics: { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0 },
    shortage: { total_required: 0, total_desired: 0, total_assigned: 0, unmet_required: 0, unmet_desired: 0 },
    optimization_config: null,
    events: [],
  },
} as unknown as GlobalResponsePlanResponse;

function renderPage() {
  return render(
    <MemoryRouter>
      <GlobalResponsePlanPage />
    </MemoryRouter>,
  );
}

describe("GlobalResponsePlanPage demo-session gate", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    getCurrentGlobalResponsePlanMock.mockResolvedValue(STALE_PLAN);
  });

  afterEach(() => {
    window.sessionStorage.clear();
    vi.clearAllMocks();
  });

  it("does not show a previous demo run's plan in a fresh browser session", async () => {
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-old", state: "completed" });

    renderPage();

    expect(await screen.findByText("No materialized generation yet")).toBeInTheDocument();
  });

  it("shows the plan while a simulation is running", async () => {
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-live", state: "running" });

    renderPage();

    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
    expect(screen.queryByText("No materialized generation yet")).not.toBeInTheDocument();
  });

  it("shows the plan of a completed run this session started", async () => {
    window.sessionStorage.setItem("ecoguard.demoSession.runId", "run-mine");
    getCurrentSimulationRunMock.mockResolvedValue({ run_id: "run-mine", state: "completed" });

    renderPage();

    expect(await screen.findByText(/Last updated/)).toBeInTheDocument();
  });
});
