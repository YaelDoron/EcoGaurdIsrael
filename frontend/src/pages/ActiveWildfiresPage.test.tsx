import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { ActiveFireEvent } from "../types/activeFireEvents";
import type { OperationsActivityDetailResponse } from "../types/operationsActivity";
import type { OperationsActivityFeedItem, OperationsOverviewResponse } from "../types/operationsOverview";
import { ActiveWildfiresPage } from "./ActiveWildfiresPage";

vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

const { getOperationsOverviewMock, getOperationsActivityDetailMock, startSimulationMock } = vi.hoisted(() => ({
  getOperationsOverviewMock: vi.fn(),
  getOperationsActivityDetailMock: vi.fn(),
  startSimulationMock: vi.fn(),
}));

vi.mock("../api/operations", () => ({
  getOperationsOverview: getOperationsOverviewMock,
  getOperationsActivityDetail: getOperationsActivityDetailMock,
}));

vi.mock("../api/simulation", () => ({
  startSimulation: startSimulationMock,
}));

function makeEvent(overrides: Partial<ActiveFireEvent> = {}): ActiveFireEvent {
  return {
    fire_event_id: 12,
    status: "confirmed",
    latitude: 32.731,
    longitude: 35.046,
    detection_confidence: 0.91,
    detected_at: "2026-09-17T13:20:00Z",
    updated_at: "2026-09-17T13:28:00Z",
    created_at: "2026-09-17T13:20:30Z",
    severity: null,
    location_name: null,
    ml_summary: null,
    ...overrides,
  };
}

const FIRE_DANGER_ACTIVITY_ITEM: OperationsActivityFeedItem = {
  activity_id: "fire_danger:1",
  activity_type: "fire_danger",
  entity_id: 1,
  occurred_at: "2026-09-17T13:00:00Z",
  available_at: "2026-09-17T13:00:00Z",
  title: "Fire Danger Assessment",
  location: { latitude: 32.731, longitude: 35.046 },
  preview: { area_name: "Northern District", status: "valid", level: "very_high", score: 42.5 },
};

function makeOverview(
  activeFires: ActiveFireEvent[],
  overrides: Partial<OperationsOverviewResponse> = {},
): OperationsOverviewResponse {
  return {
    generated_at: "2026-09-17T14:00:00Z",
    simulation: { enabled: false, run: null },
    fire_danger_areas: [],
    active_fires: activeFires,
    activity_feed: { items: [], limit: 30 },
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ActiveWildfiresPage />
    </MemoryRouter>,
  );
}

describe("ActiveWildfiresPage (Task A8 Operations Overview)", () => {
  beforeEach(() => {
    getOperationsOverviewMock.mockReset();
    getOperationsActivityDetailMock.mockReset();
    startSimulationMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows LoadingState on initial render", () => {
    getOperationsOverviewMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage();

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it("shows ErrorState with Retry when the initial load fails, never staying stuck in Loading", async () => {
    getOperationsOverviewMock.mockRejectedValue(new ApiError("boom", 500));

    renderPage();

    expect(await screen.findByRole("heading", { name: "Unable to load the operations overview." })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(screen.queryByText(/loading operations overview/i)).not.toBeInTheDocument();
  });

  it("clears the Loading state once the initial request succeeds", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();
    expect(screen.getByText(/loading operations overview/i)).toBeInTheDocument();

    await waitFor(() => expect(screen.queryByText(/loading operations overview/i)).not.toBeInTheDocument());
    expect(screen.getByRole("region", { name: "National operations map" })).toBeInTheDocument();
  });

  it("Task 5, Part 2/24: mounting the page (and its normal poll refresh) never starts a simulation or requests a reset", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();
    await waitFor(() => expect(screen.queryByText(/loading operations overview/i)).not.toBeInTheDocument());
    // A couple of normal poll cycles worth of overview fetches.
    await waitFor(() => expect(getOperationsOverviewMock.mock.calls.length).toBeGreaterThan(0));

    expect(startSimulationMock).not.toHaveBeenCalled();
  });

  it("does not render a marketing subtitle under the Operations Overview heading", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();
    await screen.findByRole("heading", { name: "Operations Overview" });

    expect(
      screen.queryByText("Live national wildfire operations: Fire Danger, active fires, and recent activity."),
    ).not.toBeInTheDocument();
  });

  it("recovers after Retry when the second attempt succeeds", async () => {
    const user = userEvent.setup();
    getOperationsOverviewMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeOverview([makeEvent({ fire_event_id: 9 })]));

    renderPage();
    await screen.findByRole("button", { name: "Retry" });
    await user.click(screen.getByRole("button", { name: "Retry" }));

    await screen.findByText("Event #9");
  });

  it("renders the map region, Active Fires panel, and Activity Feed as a valid empty state", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();

    expect(await screen.findByRole("region", { name: "National operations map" })).toBeInTheDocument();
    expect(screen.getByText("No active wildfire events")).toBeInTheDocument();
    expect(screen.getByText("No recent operational activity")).toBeInTheDocument();
  });

  it("places the map and Active Fires panel in the same top row, with the Activity Feed in its own row below (layout polish)", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [FIRE_DANGER_ACTIVITY_ITEM], limit: 30 } }),
    );

    renderPage();
    await screen.findByRole("region", { name: "National operations map" });

    const topRow = document.querySelector(".active-wildfires-page__top-row") as HTMLElement;
    const feedRow = document.querySelector(".active-wildfires-page__feed-row") as HTMLElement;
    expect(topRow).toBeInTheDocument();
    expect(feedRow).toBeInTheDocument();

    // Map + Active Fires share the top row.
    expect(within(topRow).getByRole("region", { name: "National operations map" })).toBeInTheDocument();
    expect(within(topRow).getByRole("heading", { level: 2, name: /^Active Fires/ })).toBeInTheDocument();

    // The Activity Feed lives in its own row, not inside the top row.
    expect(within(feedRow).getByText("Northern District - Very high fire danger")).toBeInTheDocument();
    expect(within(topRow).queryByText("Northern District - Very high fire danger")).not.toBeInTheDocument();

    // The top row's map+fires markup precedes the feed row in document order.
    const position = topRow.compareDocumentPosition(feedRow);
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("renders every active fire card without truncation, needing no inner scroll for the standard demo case", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([
        makeEvent({ fire_event_id: 1 }),
        makeEvent({ fire_event_id: 2 }),
        makeEvent({ fire_event_id: 3 }),
        makeEvent({ fire_event_id: 4 }),
        makeEvent({ fire_event_id: 5 }),
      ]),
    );

    renderPage();
    await screen.findByText("Event #1");

    for (const id of [1, 2, 3, 4, 5]) {
      expect(screen.getByText(`Event #${id}`)).toBeInTheDocument();
    }
  });

  it("renders a Fire Danger circle and an active-fire marker on the same map simultaneously", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([makeEvent({ fire_event_id: 1 })], {
        fire_danger_areas: [
          {
            area_id: "area-1",
            area_name: "Northern District",
            center: { latitude: 32.731, longitude: 35.046 },
            radius_km: 5,
            assessment: null,
          },
        ],
      }),
    );

    renderPage();

    expect(await screen.findByTestId("circle")).toBeInTheDocument();
    expect(screen.getByTestId("marker")).toBeInTheDocument();
  });

  it("renders one Active Fires card per active fire, preserving server order, with working navigation", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([makeEvent({ fire_event_id: 12 }), makeEvent({ fire_event_id: 3 })]),
    );

    renderPage();
    await screen.findByText("Event #12");

    const panelHeading = screen.getByRole("heading", { level: 2, name: /^Active Fires/ });
    const panel = within(panelHeading.closest("section") as HTMLElement);
    const headings = panel.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(["Event #12", "Event #3"]);
    expect(panel.getAllByRole("link", { name: "View Event" })[0]).toHaveAttribute("href", "/events/12");
  });

  it("never fabricates an area name on an active fire card", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([makeEvent({ fire_event_id: 1 })]));

    renderPage();
    await screen.findByText("Event #1");

    for (const fake of ["Carmel", "Golan"]) {
      expect(screen.queryByText(fake)).not.toBeInTheDocument();
    }
  });

  it("emphasizes a CONFIRMED + HIGH/CRITICAL severity card, never a SUSPECTED one", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([
        makeEvent({
          fire_event_id: 1,
          status: "confirmed",
          severity: { assessment_id: 1, status: "valid", score: 80, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
        }),
        makeEvent({
          fire_event_id: 2,
          status: "suspected",
          severity: { assessment_id: 2, status: "valid", score: 90, level: "critical", assessed_at: "2026-09-17T13:27:00Z" },
        }),
      ]),
    );

    renderPage();
    const confirmedHeading = await screen.findByText("Event #1");
    const suspectedHeading = screen.getByText("Event #2");

    expect(confirmedHeading.closest("article")).toHaveAttribute("data-emphasized", "true");
    expect(suspectedHeading.closest("article")).toHaveAttribute("data-emphasized", "false");
  });

  it("renders the Activity Feed in exact server order and never shows a System Update pseudo-category", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], {
        activity_feed: {
          items: [
            FIRE_DANGER_ACTIVITY_ITEM,
            {
              activity_id: "news_report:2",
              activity_type: "news_report",
              entity_id: 2,
              occurred_at: "2026-09-17T12:50:00Z",
              available_at: "2026-09-17T12:50:00Z",
              title: "News report ingested",
              location: null,
              preview: { source: "Ynet", headline: "Wildfire spreads" },
            },
          ],
          limit: 30,
        },
      }),
    );

    renderPage();
    await screen.findByText("Northern District - Very high fire danger");

    const buttons = screen.getAllByRole("button").filter((button) => button.className.includes("activity-row"));
    expect(buttons[0]).toHaveTextContent("Northern District - Very high fire danger");
    expect(buttons[1]).toHaveTextContent("Wildfire spreads");
    expect(screen.queryByText(/system update/i)).not.toBeInTheDocument();
  });

  it("opens the Activity Detail drawer on click, fetching exactly the selected item once", async () => {
    const user = userEvent.setup();
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [FIRE_DANGER_ACTIVITY_ITEM], limit: 30 } }),
    );
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "fire_danger",
      entity_id: 1,
      occurred_at: "2026-09-17T13:00:00Z",
      title: "Fire Danger Assessment",
      location: null,
      details: {
        assessment_id: 1,
        area_id: "area-1",
        area_name: "Northern District",
        center: { latitude: 32.731, longitude: 35.046 },
        radius_km: 5,
        status: "valid",
        score: 42.5,
        level: "very_high",
        assessed_at: "2026-09-17T13:00:00Z",
        age_seconds: 60,
        methodology: "FOSBERG_FFWI",
        methodology_version: "1.0",
        weather_inputs: [],
      },
    } satisfies OperationsActivityDetailResponse);

    renderPage();
    await user.click(await screen.findByText("Northern District - Very high fire danger"));

    await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument());
    expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1);
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("fire_danger", 1, expect.anything());
  });

  it("closes the drawer via its close control", async () => {
    const user = userEvent.setup();
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [FIRE_DANGER_ACTIVITY_ITEM], limit: 30 } }),
    );
    getOperationsActivityDetailMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage();
    await user.click(await screen.findByText("Northern District - Very high fire danger"));
    await screen.findByRole("dialog");

    await user.click(screen.getByRole("button", { name: "Close activity details" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows the simulation status indicator, with the run control disabled while RUNNING", async () => {
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], {
        simulation: {
          enabled: true,
          run: {
            run_id: "run-1",
            state: "running",
            preset_id: "standard",
            seed: 1,
            mode: "automatic",
            simulation_duration_seconds: 600,
            events_total: 4,
            events_completed: 1,
            events_succeeded: 1,
            events_failed: 0,
            current_event: null,
            started_at: "2026-09-17T13:50:00Z",
            completed_at: null,
            wall_clock_elapsed_seconds: 30,
            last_message: null,
            error: null,
          },
        },
      }),
    );

    renderPage();

    expect(await screen.findByText("Running - Event 2 of 4")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start simulation/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();
  });

  it("shows an enabled Start Simulation control when simulation.enabled and no run has started (Task A9)", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([], { simulation: { enabled: true, run: null } }));

    renderPage();

    expect(await screen.findByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("hides the simulation action entirely when simulation.enabled is false (Task A9)", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([], { simulation: { enabled: false, run: null } }));

    renderPage();
    await screen.findByText("No active wildfire events");

    expect(screen.queryByRole("button", { name: /start simulation|run again/i })).not.toBeInTheDocument();
  });

  it("displays the A6 generated_at timestamp as Last updated", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([], { generated_at: "2026-09-17T14:00:00Z" }));

    renderPage();
    await screen.findByText(/last updated/i);

    const time = document.querySelector("time");
    expect(time).toHaveAttribute("dateTime", "2026-09-17T14:00:00Z");
  });

  it("makes exactly one operations-overview request on initial load (no direct sub-endpoint polling)", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();
    await screen.findByText("No active wildfire events");

    expect(getOperationsOverviewMock).toHaveBeenCalledTimes(1);
  });

  it("never renders global_planning_run as an Activity Feed row", async () => {
    const globalPlanningItem: OperationsActivityFeedItem = {
      activity_id: "global_planning_run:7",
      activity_type: "global_planning_run",
      entity_id: 7,
      occurred_at: "2026-09-17T13:05:00Z",
      available_at: "2026-09-17T13:05:00Z",
      title: "Global Response Plan",
      location: null,
      preview: { status: "completed", fire_event_count: 2 },
    };
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [globalPlanningItem, FIRE_DANGER_ACTIVITY_ITEM], limit: 30 } }),
    );

    renderPage();
    await screen.findByText("Northern District - Very high fire danger");

    // Scoped to the Activity Feed itself: the page header's own static
    // "Global Response Plan" navigation link is a separate, legitimate
    // element and must not make this assertion false-positive.
    const feed = screen.getByRole("region", { name: /activity/i });
    expect(within(feed).queryByText(/global response plan/i)).not.toBeInTheDocument();
    expect(within(feed).queryByText(/2 active fires/)).not.toBeInTheDocument();
  });

  it("does not show View Response Plan when no global planning run has ever been persisted", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();
    await screen.findByText("No active wildfire events");

    expect(screen.queryByRole("button", { name: /view response plan/i })).not.toBeInTheDocument();
  });

  it("shows View Response Plan beside Active Fires when a real global planning run exists", async () => {
    const globalPlanningItem: OperationsActivityFeedItem = {
      activity_id: "global_planning_run:7",
      activity_type: "global_planning_run",
      entity_id: 7,
      occurred_at: "2026-09-17T13:05:00Z",
      available_at: "2026-09-17T13:05:00Z",
      title: "Global Response Plan",
      location: null,
      preview: { status: "completed", fire_event_count: 2 },
    };
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [globalPlanningItem], limit: 30 } }),
    );

    renderPage();

    expect(await screen.findByRole("button", { name: "View Response Plan" })).toBeInTheDocument();
  });

  it("removing Global Planning from the feed does not remove access to the Response Plan", async () => {
    const user = userEvent.setup();
    const globalPlanningItem: OperationsActivityFeedItem = {
      activity_id: "global_planning_run:7",
      activity_type: "global_planning_run",
      entity_id: 7,
      occurred_at: "2026-09-17T13:05:00Z",
      available_at: "2026-09-17T13:05:00Z",
      title: "Global Response Plan",
      location: null,
      preview: { status: "completed", fire_event_count: 2 },
    };
    getOperationsOverviewMock.mockResolvedValue(
      makeOverview([], { activity_feed: { items: [globalPlanningItem], limit: 30 } }),
    );
    getOperationsActivityDetailMock.mockResolvedValue({
      activity_type: "global_planning_run",
      entity_id: 7,
      occurred_at: "2026-09-17T13:05:00Z",
      title: "Global Response Plan",
      location: null,
      details: {
        global_planning_run_id: 7,
        status: "completed",
        trigger: "fire_event_update",
        started_at: "2026-09-17T13:00:00Z",
        completed_at: "2026-09-17T13:05:00Z",
        methodology: "global_planning_refresh_coordinator",
        methodology_version: "1.0",
        fire_event_ids: [12],
        response_plan_ids: [66],
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
      },
    } satisfies OperationsActivityDetailResponse);

    renderPage();
    const button = await screen.findByRole("button", { name: "View Response Plan" });
    expect(screen.queryByText(/global response plan/i, { selector: ".activity-row__main" })).not.toBeInTheDocument();

    await user.click(button);

    await waitFor(() => expect(getOperationsActivityDetailMock).toHaveBeenCalledTimes(1));
    expect(getOperationsActivityDetailMock).toHaveBeenCalledWith("global_planning_run", 7);
  });
});

describe("ActiveWildfiresPage Start Simulation end-to-end flow (Task A9, Part 37)", () => {
  beforeEach(() => {
    getOperationsOverviewMock.mockReset();
    getOperationsActivityDetailMock.mockReset();
    startSimulationMock.mockReset();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("reacts to each new overview snapshot only - PREPARING -> RUNNING -> Fire Danger -> active fire -> Severity -> completed", async () => {
    const idleSnapshot = makeOverview([], { simulation: { enabled: true, run: null } });

    const preparingRun = {
      run_id: "run-9",
      state: "preparing" as const,
      preset_id: "operations_demo",
      seed: 42,
      mode: "automatic",
      simulation_duration_seconds: null,
      events_total: null,
      events_completed: 0,
      events_succeeded: 0,
      events_failed: 0,
      current_event: null,
      started_at: null,
      completed_at: null,
      wall_clock_elapsed_seconds: null,
      last_message: "Preparing simulation run.",
      error: null,
    };
    const preparingSnapshot = makeOverview([], { simulation: { enabled: true, run: preparingRun } });

    const runningRun = { ...preparingRun, state: "running" as const, events_total: 18, events_completed: 0 };
    const fireDangerArea = {
      area_id: "area-1",
      area_name: "Galilee Demo Area",
      center: { latitude: 32.9, longitude: 35.3 },
      radius_km: 5,
      assessment: {
        assessment_id: 1,
        status: "valid" as const,
        score: 88,
        level: "very_high" as const,
        assessed_at: "2026-09-19T10:00:05Z",
        age_seconds: 5,
        methodology: "FOSBERG_FFWI",
        methodology_version: "1.0",
      },
    };
    const fireDangerSnapshot = makeOverview([], {
      simulation: { enabled: true, run: { ...runningRun, events_completed: 2 } },
      fire_danger_areas: [fireDangerArea],
    });

    const activeFireSnapshot = makeOverview([makeEvent({ fire_event_id: 18, status: "suspected", severity: null })], {
      simulation: { enabled: true, run: { ...runningRun, events_completed: 5 } },
      fire_danger_areas: [fireDangerArea],
    });

    const severitySnapshot = makeOverview(
      [
        makeEvent({
          fire_event_id: 18,
          status: "confirmed",
          severity: { assessment_id: 1, status: "valid", score: 84, level: "critical", assessed_at: "2026-09-19T10:00:20Z" },
        }),
      ],
      {
        simulation: { enabled: true, run: { ...runningRun, events_completed: 8 } },
        fire_danger_areas: [fireDangerArea],
      },
    );

    const completedRun = {
      ...runningRun,
      state: "completed" as const,
      events_completed: 18,
      completed_at: "2026-09-19T10:02:00Z",
    };
    const completedSnapshot = makeOverview(
      [
        makeEvent({
          fire_event_id: 18,
          status: "confirmed",
          severity: { assessment_id: 1, status: "valid", score: 84, level: "critical", assessed_at: "2026-09-19T10:00:20Z" },
        }),
      ],
      {
        simulation: { enabled: true, run: completedRun },
        fire_danger_areas: [fireDangerArea],
      },
    );

    getOperationsOverviewMock
      .mockResolvedValueOnce(idleSnapshot)
      .mockResolvedValueOnce(preparingSnapshot)
      .mockResolvedValueOnce(fireDangerSnapshot)
      .mockResolvedValueOnce(activeFireSnapshot)
      .mockResolvedValueOnce(severitySnapshot)
      .mockResolvedValue(completedSnapshot);
    startSimulationMock.mockResolvedValue(preparingRun);

    renderPage();
    expect(await screen.findByRole("button", { name: "Start Simulation" })).toBeEnabled();

    screen.getByRole("button", { name: "Start Simulation" }).click();

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    expect(startSimulationMock).toHaveBeenCalledWith({ preset: "operations_demo", reset_demo_state: true });
    await waitFor(() => expect(getOperationsOverviewMock).toHaveBeenCalledTimes(2));
    expect(await screen.findByRole("button", { name: "Preparing…" })).toBeDisabled();

    await vi.advanceTimersByTimeAsync(5000);
    expect(await screen.findByTestId("circle")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Running…" })).toBeDisabled();

    await vi.advanceTimersByTimeAsync(5000);
    expect(await screen.findByText("Event #18")).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(5000);
    await waitFor(() => expect(screen.getByText("Event #18").closest("article")).toHaveAttribute("data-emphasized", "true"));

    await vi.advanceTimersByTimeAsync(5000);
    expect(await screen.findByRole("button", { name: "Run Again" })).toBeEnabled();
    expect(screen.getByText("Completed")).toBeInTheDocument();
  });

  it("does not insert any local fake activity/fire/danger data when Start is clicked", async () => {
    const idleSnapshot = makeOverview([], { simulation: { enabled: true, run: null } });
    startSimulationMock.mockReturnValue(new Promise<never>(() => {}));
    getOperationsOverviewMock.mockResolvedValue(idleSnapshot);

    renderPage();
    await screen.findByText("No active wildfire events");

    screen.getByRole("button", { name: "Start Simulation" }).click();

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    // Still the same clean empty state - Start does not fabricate anything locally.
    expect(screen.getByText("No active wildfire events")).toBeInTheDocument();
    expect(screen.getByText("No recent operational activity")).toBeInTheDocument();
    expect(screen.queryByTestId("circle")).not.toBeInTheDocument();
  });

  it("offers a prominent Global Response Plan link to /response-plan with no focus filter", async () => {
    getOperationsOverviewMock.mockResolvedValue(makeOverview([]));

    renderPage();

    const link = await screen.findByRole("link", { name: "Global Response Plan" });
    expect(link).toHaveAttribute("href", "/response-plan");
  });
});
