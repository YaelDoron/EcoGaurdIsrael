import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { SimulationRunStatus } from "../types/simulation";
import { useStartSimulation } from "./useStartSimulation";

const { startSimulationMock } = vi.hoisted(() => ({
  startSimulationMock: vi.fn(),
}));

vi.mock("../api/simulation", () => ({
  startSimulation: startSimulationMock,
}));

function preparingStatus(): SimulationRunStatus {
  return {
    run_id: "run-1",
    state: "preparing",
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
}

describe("useStartSimulation", () => {
  beforeEach(() => {
    startSimulationMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("sends the POST exactly once per call", async () => {
    startSimulationMock.mockResolvedValue(preparingStatus());
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      await result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
    });

    expect(startSimulationMock).toHaveBeenCalledTimes(1);
  });

  it("sends the exact operations_demo/seed/reset_demo_state request payload", async () => {
    startSimulationMock.mockResolvedValue(preparingStatus());
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      await result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
    });

    expect(startSimulationMock).toHaveBeenCalledWith(
      { preset: "operations_demo", seed: 42, reset_demo_state: true },
    );
  });

  it("resolves as soon as the 202 response arrives, without waiting for run completion", async () => {
    startSimulationMock.mockResolvedValue(preparingStatus());
    const { result } = renderHook(() => useStartSimulation());

    let resolved: SimulationRunStatus | undefined;
    await act(async () => {
      resolved = await result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
    });

    expect(resolved?.state).toBe("preparing");
    expect(resolved?.run_id).toBe("run-1");
  });

  it("does not internally poll for the final run status", async () => {
    startSimulationMock.mockResolvedValue(preparingStatus());
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      await result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
    });

    // Only the single startSimulation call - nothing resembling a status-polling call.
    expect(startSimulationMock).toHaveBeenCalledTimes(1);
  });

  it("lets the caller trigger an Operations Overview refetch after success (does not do it itself)", async () => {
    startSimulationMock.mockResolvedValue(preparingStatus());
    const onSuccess = vi.fn();
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      const status = await result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
      onSuccess(status);
    });

    expect(onSuccess).toHaveBeenCalledWith(preparingStatus());
  });

  it("preserves a 409 SIMULATION_ALREADY_RUNNING conflict as a typed, recognizable error", async () => {
    startSimulationMock.mockRejectedValue(
      new ApiError("A simulation run is already running.", 409, "SIMULATION_ALREADY_RUNNING"),
    );
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      await expect(
        result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true }),
      ).rejects.toBeInstanceOf(ApiError);
    });

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.status).toBe(409);
    expect(result.current.error?.code).toBe("SIMULATION_ALREADY_RUNNING");
    expect(result.current.isStarting).toBe(false);
  });

  it("preserves a 403 SIMULATION_CONTROL_DISABLED error as a typed, recognizable error", async () => {
    startSimulationMock.mockRejectedValue(
      new ApiError("The Simulation Control API is not enabled.", 403, "SIMULATION_CONTROL_DISABLED"),
    );
    const { result } = renderHook(() => useStartSimulation());

    await act(async () => {
      await expect(
        result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true }),
      ).rejects.toBeInstanceOf(ApiError);
    });

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.status).toBe(403);
    expect(result.current.error?.code).toBe("SIMULATION_CONTROL_DISABLED");
  });

  it("isStarting reflects the in-flight request", async () => {
    let resolveStart!: (value: SimulationRunStatus) => void;
    startSimulationMock.mockReturnValue(
      new Promise<SimulationRunStatus>((resolve) => {
        resolveStart = resolve;
      }),
    );
    const { result } = renderHook(() => useStartSimulation());

    let startPromise!: Promise<SimulationRunStatus>;
    act(() => {
      startPromise = result.current.start({ preset: "operations_demo", seed: 42, reset_demo_state: true });
    });

    await waitFor(() => expect(result.current.isStarting).toBe(true));

    await act(async () => {
      resolveStart(preparingStatus());
      await startPromise;
    });

    expect(result.current.isStarting).toBe(false);
  });
});
