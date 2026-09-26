import { act, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LatLngPoint } from "../components/map/mapTypes";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { GlobalResponsePlanPage } from "./GlobalResponsePlanPage";

const { getCurrentGlobalResponsePlanMock, getEventDetailsMock, reverseGeocodeMock, fitBoundsPointsSeen } = vi.hoisted(
  () => ({
    getCurrentGlobalResponsePlanMock: vi.fn(),
    getEventDetailsMock: vi.fn(),
    reverseGeocodeMock: vi.fn(),
    fitBoundsPointsSeen: [] as unknown[],
  }),
);

// A live demo run, so the demo-session gate (hooks/demoSession.ts) shows the plan.
vi.mock("../api/simulation", () => ({
  getCurrentSimulationRun: vi.fn().mockResolvedValue({ run_id: "run-live", state: "running" }),
}));
vi.mock("../api/globalResponsePlan", () => ({ getCurrentGlobalResponsePlan: getCurrentGlobalResponsePlanMock }));
vi.mock("../api/eventDetails", () => ({ getEventDetails: getEventDetailsMock }));
vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));
vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));
// FitBoundsToPoints re-frames the map whenever its `points` identity changes;
// recording what it receives is the direct way to observe that.
vi.mock("../components/map/FitBoundsToPoints", () => ({
  FitBoundsToPoints: ({ points }: { points: LatLngPoint[] }) => {
    fitBoundsPointsSeen.push(points);
    return null;
  },
}));

function makeResponse(completedAt: string, asOf = "2026-09-20T10:00:00Z"): GlobalResponsePlanResponse {
  return {
    as_of: asOf,
    plan: {
      run_id: 77,
      started_at: "2026-09-20T10:00:00Z",
      completed_at: completedAt,
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
      ],
    },
  };
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("GlobalResponsePlanPage polling", () => {
  beforeEach(() => {
    getCurrentGlobalResponsePlanMock.mockReset();
    getEventDetailsMock.mockReset();
    getEventDetailsMock.mockRejectedValue(new Error("no news evidence"));
    reverseGeocodeMock.mockReset();
    reverseGeocodeMock.mockResolvedValue(null);
    fitBoundsPointsSeen.length = 0;
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function renderPage() {
    return render(
      <MemoryRouter initialEntries={["/global-response-plan"]}>
        <Routes>
          <Route path="/global-response-plan" element={<GlobalResponsePlanPage />} />
        </Routes>
      </MemoryRouter>,
    );
  }

  it("shows the newer plan after a poll, without a reload and without flashing the loading state", async () => {
    let resolvePoll: (value: GlobalResponsePlanResponse) => void = () => {};
    getCurrentGlobalResponsePlanMock
      .mockResolvedValueOnce(makeResponse("2026-09-20T10:03:00Z"))
      .mockReturnValueOnce(new Promise<GlobalResponsePlanResponse>((resolve) => (resolvePoll = resolve)));

    const { container } = renderPage();
    await advance(0);
    expect(container.querySelector("time")?.getAttribute("datetime")).toBe("2026-09-20T10:03:00Z");

    await advance(5000);
    // Poll in flight: the plan stays on screen, no full-screen loading state.
    expect(screen.queryByText(/loading global response plan/i)).not.toBeInTheDocument();
    expect(container.querySelector("time")?.getAttribute("datetime")).toBe("2026-09-20T10:03:00Z");

    resolvePoll(makeResponse("2026-09-20T10:07:00Z"));
    await advance(0);

    expect(container.querySelector("time")?.getAttribute("datetime")).toBe("2026-09-20T10:07:00Z");
  });

  it("does not re-fit the map or repeat location lookups when a poll returns an unchanged plan", async () => {
    let call = 0;
    getCurrentGlobalResponsePlanMock.mockImplementation(async () =>
      makeResponse("2026-09-20T10:03:00Z", `2026-09-20T10:00:0${call++}Z`),
    );

    renderPage();
    await advance(0);
    const firstPoints = fitBoundsPointsSeen.at(-1);
    expect(firstPoints).toBeDefined();
    const geocodeCallsAfterFirstLoad = reverseGeocodeMock.mock.calls.length;
    const eventDetailsCallsAfterFirstLoad = getEventDetailsMock.mock.calls.length;

    await advance(5000);
    await advance(5000);

    expect(getCurrentGlobalResponsePlanMock).toHaveBeenCalledTimes(3);
    expect(new Set(fitBoundsPointsSeen).size).toBe(1);
    expect(fitBoundsPointsSeen.at(-1)).toBe(firstPoints);
    expect(reverseGeocodeMock.mock.calls.length).toBe(geocodeCallsAfterFirstLoad);
    expect(getEventDetailsMock.mock.calls.length).toBe(eventDetailsCallsAfterFirstLoad);
  });
});
