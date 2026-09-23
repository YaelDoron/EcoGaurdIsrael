import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { OperationsStatusHeader } from "./OperationsStatusHeader";
import type { OperationsSimulationSummary } from "../../types/operationsOverview";
import type { SimulationRunStatus } from "../../types/simulation";

function makeRun(overrides: Partial<SimulationRunStatus> = {}): SimulationRunStatus {
  return {
    run_id: "run-1",
    state: "running",
    preset_id: "standard",
    seed: 42,
    mode: "automatic",
    simulation_duration_seconds: 600,
    events_total: 5,
    events_completed: 2,
    events_succeeded: 2,
    events_failed: 0,
    current_event: null,
    started_at: "2026-09-20T10:00:00Z",
    completed_at: null,
    wall_clock_elapsed_seconds: 120,
    last_message: null,
    error: null,
    ...overrides,
  };
}

function renderHeader(simulation: OperationsSimulationSummary, onRequestOverviewRefresh = vi.fn()) {
  return render(
    <OperationsStatusHeader
      simulation={simulation}
      generatedAt="2026-09-20T11:00:00Z"
      onRequestOverviewRefresh={onRequestOverviewRefresh}
    />,
  );
}

describe("OperationsStatusHeader", () => {
  it("never renders a simulation-state badge/banner at all - production polish pass", () => {
    for (const simulation of [
      { enabled: false, run: null },
      { enabled: true, run: null },
      { enabled: true, run: makeRun({ state: "preparing" }) },
      { enabled: true, run: makeRun({ state: "running" }) },
      { enabled: true, run: makeRun({ state: "completed" }) },
      { enabled: true, run: makeRun({ state: "completed_with_errors" }) },
      { enabled: true, run: makeRun({ state: "failed" }) },
    ] as OperationsSimulationSummary[]) {
      const { unmount } = renderHeader(simulation);
      for (const forbidden of [
        "No Simulation Running",
        "Running",
        "Preparing Simulation",
        "Completed",
        "Completed with Errors",
        "Failed",
      ]) {
        expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
      }
      expect(document.querySelector(".badge")).not.toBeInTheDocument();
      unmount();
    }
  });

  it("still shows the run's own safe error message on a failed run, even without a status badge", () => {
    renderHeader({
      enabled: true,
      run: makeRun({ state: "failed", error: { code: "EVENT_FAILED", message: "One event could not be processed." } }),
    });

    expect(screen.getByText("One event could not be processed.")).toBeInTheDocument();
  });

  it("shows the A6 generated_at timestamp, never Date.now()", () => {
    renderHeader({ enabled: false, run: null });

    const time = screen.getByText((_, element) => element?.tagName.toLowerCase() === "time");
    expect(time).toHaveAttribute("datetime", "2026-09-20T11:00:00Z");
  });

  it("shows Start Simulation when enabled and no run has started (Task A9)", () => {
    renderHeader({ enabled: true, run: null });

    expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled();
  });

  it("hides the simulation action entirely when simulation.enabled is false (Task A9)", () => {
    renderHeader({ enabled: false, run: null });

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("disables the action while PREPARING/RUNNING", () => {
    renderHeader({ enabled: true, run: makeRun({ state: "preparing" }) });
    expect(screen.getByRole("button", { name: "Preparing…" })).toBeDisabled();
  });

  it("offers Run Again in every terminal state", () => {
    for (const state of ["completed", "completed_with_errors", "failed"] as const) {
      const { unmount } = renderHeader({ enabled: true, run: makeRun({ state }) });
      expect(screen.getByRole("button", { name: "Run Again" })).toBeEnabled();
      unmount();
    }
  });

  it("never shows simulation-engine telemetry (current event, incident id, elapsed seconds) - production polish pass", () => {
    renderHeader({
      enabled: true,
      run: makeRun({
        state: "running",
        current_event: { event_index: 6, incident_id: "incident-carmel-01", event_type: "NEWS", timestamp_offset_sec: 40 },
        wall_clock_elapsed_seconds: 87,
      }),
    });

    for (const forbidden of [/Current event/i, /Current incident/i, /Elapsed:/i, /incident-carmel-01/i, /87s/]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });

  it("never renders a Stop/Cancel/Pause/Reset control", () => {
    renderHeader({ enabled: true, run: makeRun({ state: "running" }) });

    for (const forbidden of [/stop/i, /cancel/i, /pause/i, /reset demo/i]) {
      expect(screen.queryByRole("button", { name: forbidden })).not.toBeInTheDocument();
    }
  });
});
