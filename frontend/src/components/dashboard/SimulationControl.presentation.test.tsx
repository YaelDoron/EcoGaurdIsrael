import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SimulationControl } from "./SimulationControl";
import { ApiError } from "../../api/errors";
import type { SimulationRunStatus } from "../../types/simulation";

const { startSimulationMock, stopSimulationMock } = vi.hoisted(() => ({
  startSimulationMock: vi.fn(),
  stopSimulationMock: vi.fn(),
}));

vi.mock("../../api/simulation", () => ({
  startSimulation: startSimulationMock,
  stopSimulation: stopSimulationMock,
}));

function makeRun(overrides: Partial<SimulationRunStatus> = {}): SimulationRunStatus {
  return {
    run_id: "run-1",
    state: "running",
    preset_id: "presentation_demo",
    seed: 290,
    mode: "automatic",
    simulation_duration_seconds: 1800,
    events_total: 63,
    events_completed: 6,
    events_succeeded: 6,
    events_failed: 0,
    current_event: null,
    started_at: "2026-09-19T10:00:00Z",
    completed_at: null,
    wall_clock_elapsed_seconds: 42,
    last_message: null,
    error: null,
    ...overrides,
  };
}

describe("SimulationControl - single Start / Stop control", () => {
  beforeEach(() => {
    startSimulationMock.mockReset();
    stopSimulationMock.mockReset();
  });

  it("Start Simulation starts the paced presentation run with the backend's pinned seed and its reset", async () => {
    const user = userEvent.setup();
    const refresh = vi.fn();
    startSimulationMock.mockResolvedValue(makeRun({ state: "preparing", run_id: "run-p" }));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={refresh} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    expect(startSimulationMock).toHaveBeenCalledWith({ preset: "presentation_demo", reset_demo_state: true });
    await waitFor(() => expect(refresh).toHaveBeenCalled());
    // The tab that started it keeps showing its results (demo-session gate).
    expect(window.sessionStorage.getItem("ecoguard.demoSession.runId")).toBe("run-p");
  });

  it.each([null, "idle", "completed", "completed_with_errors", "stopped", "failed"] as const)(
    "shows exactly one button - Start Simulation - when the run is %s",
    (state) => {
      render(<SimulationControl run={state === null ? null : makeRun({ state })} enabled onRequestOverviewRefresh={vi.fn()} />);

      expect(screen.getAllByRole("button")).toHaveLength(1);
      expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
    },
  );

  it.each(["preparing", "running"] as const)("shows exactly one button - Stop Simulation - while %s", (state) => {
    render(<SimulationControl run={makeRun({ state })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Stop Simulation" })).toBeEnabled();
  });

  it("never offers a separate presentation button", () => {
    for (const run of [null, makeRun({ state: "running" }), makeRun({ state: "stopped" })]) {
      const { unmount } = render(<SimulationControl run={run} enabled onRequestOverviewRefresh={vi.fn()} />);
      expect(screen.queryByRole("button", { name: /presentation/i })).not.toBeInTheDocument();
      unmount();
    }
  });

  it("while the backend reports STOPPING, the button is disabled and says the run is stopping", () => {
    render(<SimulationControl run={makeRun({ state: "stopping" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getAllByRole("button")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Stopping…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("Stopping simulation...");
  });

  it("RUNNING -> STOPPING -> STOPPED is reflected from the polled run state", () => {
    const { rerender } = render(
      <SimulationControl run={makeRun({ state: "running" })} enabled onRequestOverviewRefresh={vi.fn()} />,
    );
    expect(screen.getByRole("button", { name: "Stop Simulation" })).toBeEnabled();

    rerender(<SimulationControl run={makeRun({ state: "stopping" })} enabled onRequestOverviewRefresh={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Stopping…" })).toBeDisabled();

    rerender(<SimulationControl run={makeRun({ state: "stopped" })} enabled onRequestOverviewRefresh={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
    expect(screen.getByRole("status")).toHaveTextContent("Simulation stopped");
  });

  it("clicking Stop sends exactly one stop request, shows Stopping and refreshes the overview", async () => {
    const user = userEvent.setup();
    const refresh = vi.fn();
    stopSimulationMock.mockResolvedValue(makeRun({ last_message: "Stopping after the current event." }));
    render(<SimulationControl run={makeRun()} enabled onRequestOverviewRefresh={refresh} />);

    await user.click(screen.getByRole("button", { name: "Stop Simulation" }));

    expect(stopSimulationMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Stopping…" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("Stopping simulation...");
    await waitFor(() => expect(refresh).toHaveBeenCalled());
  });

  it("a 409 from Stop (the run already ended) just refreshes to the real state", async () => {
    const user = userEvent.setup();
    const refresh = vi.fn();
    stopSimulationMock.mockRejectedValue(new ApiError("No simulation run is active.", 409));
    render(<SimulationControl run={makeRun()} enabled onRequestOverviewRefresh={refresh} />);

    await user.click(screen.getByRole("button", { name: "Stop Simulation" }));

    await waitFor(() => expect(refresh).toHaveBeenCalled());
    expect(screen.queryByText(/unable to stop/i)).not.toBeInTheDocument();
  });

  it("a failed Stop shows an error and lets the operator retry", async () => {
    const user = userEvent.setup();
    stopSimulationMock.mockRejectedValue(new ApiError("boom", 500));
    render(<SimulationControl run={makeRun()} enabled onRequestOverviewRefresh={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Stop Simulation" }));

    expect(await screen.findByText("Unable to stop the simulation. Please try again.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop Simulation" })).toBeEnabled();
  });

  it("clearly shows that a run was stopped, and offers Start Simulation again", () => {
    render(<SimulationControl run={makeRun({ state: "stopped" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getByRole("status")).toHaveTextContent("Simulation stopped");
    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("shows a completed run as completed", () => {
    render(<SimulationControl run={makeRun({ state: "completed" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getByRole("status")).toHaveTextContent("Simulation completed");
  });
});
