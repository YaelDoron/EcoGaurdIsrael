import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SimulationControl } from "./SimulationControl";
import { ApiError } from "../../api/errors";
import type { SimulationRunStatus } from "../../types/simulation";

const { startSimulationMock } = vi.hoisted(() => ({
  startSimulationMock: vi.fn(),
}));

vi.mock("../../api/simulation", () => ({
  startSimulation: startSimulationMock,
}));

function makeRun(overrides: Partial<SimulationRunStatus> = {}): SimulationRunStatus {
  return {
    run_id: "run-1",
    state: "running",
    preset_id: "operations_demo",
    seed: 42,
    mode: "automatic",
    simulation_duration_seconds: 120,
    events_total: 18,
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

describe("SimulationControl", () => {
  beforeEach(() => {
    startSimulationMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when simulation control is disabled", () => {
    const { container } = render(<SimulationControl run={null} enabled={false} onRequestOverviewRefresh={vi.fn()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("shows an enabled Start Simulation button when enabled and no run has started", () => {
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />);

    const button = screen.getByRole("button", { name: "Start Simulation" });
    expect(button).toBeEnabled();
  });

  it("shows an enabled Start Simulation button for state idle", () => {
    render(<SimulationControl run={makeRun({ state: "idle" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("disables the action while PREPARING", () => {
    render(<SimulationControl run={makeRun({ state: "preparing" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    const button = screen.getByRole("button", { name: "Preparing…" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
  });

  it("disables the action while RUNNING", () => {
    render(<SimulationControl run={makeRun({ state: "running" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    const button = screen.getByRole("button", { name: "Running…" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("aria-busy", "true");
  });

  it("offers Run Again after COMPLETED", () => {
    render(<SimulationControl run={makeRun({ state: "completed" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Run Again" })).toBeEnabled();
  });

  it("offers Run Again after COMPLETED_WITH_ERRORS", () => {
    render(
      <SimulationControl run={makeRun({ state: "completed_with_errors" })} enabled onRequestOverviewRefresh={vi.fn()} />,
    );

    expect(screen.getByRole("button", { name: "Run Again" })).toBeEnabled();
  });

  it("offers Run Again after FAILED", () => {
    render(<SimulationControl run={makeRun({ state: "failed" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Run Again" })).toBeEnabled();
  });

  it("never renders a Stop/Cancel/Pause control", () => {
    render(<SimulationControl run={makeRun({ state: "running" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    for (const forbidden of [/stop/i, /cancel/i, /pause/i]) {
      expect(screen.queryByRole("button", { name: forbidden })).not.toBeInTheDocument();
    }
  });

  it("clicking Start invokes the mutation exactly once with the fixed demo request", async () => {
    const user = userEvent.setup();
    startSimulationMock.mockResolvedValue(makeRun({ state: "preparing" }));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    expect(startSimulationMock).toHaveBeenCalledWith({ preset: "operations_demo", reset_demo_state: true });
  });

  it("a rapid double click issues only one POST", async () => {
    const user = userEvent.setup();
    let resolveStart: (value: SimulationRunStatus) => void = () => {};
    startSimulationMock.mockReturnValue(
      new Promise<SimulationRunStatus>((resolve) => {
        resolveStart = resolve;
      }),
    );
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />);

    const button = screen.getByRole("button", { name: "Start Simulation" });
    await user.click(button);
    await user.click(button);

    resolveStart(makeRun({ state: "preparing" }));
    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
  });

  it("triggers exactly one overview refresh after a successful 202, without waiting for completion", async () => {
    const user = userEvent.setup();
    const onRequestOverviewRefresh = vi.fn();
    startSimulationMock.mockResolvedValue(makeRun({ state: "preparing" }));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={onRequestOverviewRefresh} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(onRequestOverviewRefresh).toHaveBeenCalledTimes(1));
  });

  it("Run Again sends the same fixed demo request", async () => {
    const user = userEvent.setup();
    startSimulationMock.mockResolvedValue(makeRun({ state: "preparing" }));
    render(<SimulationControl run={makeRun({ state: "completed" })} enabled onRequestOverviewRefresh={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Run Again" }));

    await waitFor(() =>
      expect(startSimulationMock).toHaveBeenCalledWith({ preset: "operations_demo", reset_demo_state: true }),
    );
    expect(startSimulationMock).toHaveBeenCalledTimes(1);
  });

  it("shows a concise message and refreshes the overview on 409 SIMULATION_ALREADY_RUNNING, without retrying", async () => {
    const user = userEvent.setup();
    const onRequestOverviewRefresh = vi.fn();
    startSimulationMock.mockRejectedValue(new ApiError("A simulation run is already running.", 409, "SIMULATION_ALREADY_RUNNING"));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={onRequestOverviewRefresh} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    expect(await screen.findByText("A simulation run is already running.")).toBeInTheDocument();
    expect(onRequestOverviewRefresh).toHaveBeenCalledTimes(1);
    expect(startSimulationMock).toHaveBeenCalledTimes(1);
  });

  it("shows a safe message and refreshes the overview on 403 SIMULATION_CONTROL_DISABLED", async () => {
    const user = userEvent.setup();
    const onRequestOverviewRefresh = vi.fn();
    startSimulationMock.mockRejectedValue(
      new ApiError("The Simulation Control API is not enabled on this deployment.", 403, "SIMULATION_CONTROL_DISABLED"),
    );
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={onRequestOverviewRefresh} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    expect(await screen.findByText("The Simulation Control API is not enabled on this deployment.")).toBeInTheDocument();
    expect(onRequestOverviewRefresh).toHaveBeenCalledTimes(1);
  });

  it("shows a safe message on 403 SIMULATION_RESET_DISABLED without starting a fake run", async () => {
    const user = userEvent.setup();
    startSimulationMock.mockRejectedValue(
      new ApiError("reset_demo_state was requested but ENABLE_DEMO_DATA_RESET is not enabled.", 403, "SIMULATION_RESET_DISABLED"),
    );
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    expect(
      await screen.findByText("reset_demo_state was requested but ENABLE_DEMO_DATA_RESET is not enabled."),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("shows a local action error for an unexpected failure, keeping the control usable for retry", async () => {
    const user = userEvent.setup();
    const onRequestOverviewRefresh = vi.fn();
    startSimulationMock.mockRejectedValue(new ApiError("Failed to start the simulation run.", 500, "SIMULATION_START_FAILED"));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={onRequestOverviewRefresh} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    expect(await screen.findByText("Failed to start the simulation run.")).toBeInTheDocument();
    expect(onRequestOverviewRefresh).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("never uses the mutation's own response as a local completed state - the run prop remains authoritative", async () => {
    const user = userEvent.setup();
    startSimulationMock.mockResolvedValue(makeRun({ state: "preparing" }));
    render(<SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    // No "Run Again"/"Completed" label appears merely from the 202 response -
    // only a change to the `run` prop (a fresh A6 poll) would produce that.
    expect(screen.queryByRole("button", { name: "Run Again" })).not.toBeInTheDocument();
  });
});
