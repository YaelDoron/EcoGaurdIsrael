import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getCurrentSimulationRun, getSimulationPresets, startSimulation } from "./simulation";
import { ApiError } from "./errors";
import type { SimulationRunStatus } from "../types/simulation";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

function idleRunStatus(): SimulationRunStatus {
  return {
    run_id: null,
    state: "idle",
    preset_id: null,
    seed: null,
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
    last_message: null,
    error: null,
  };
}

describe("simulation API client", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("getSimulationPresets requests the correct path", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ presets: [] }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getSimulationPresets();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/simulation/presets",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("getCurrentSimulationRun requests the correct path and returns idle state cleanly", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(idleRunStatus()));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const result = await getCurrentSimulationRun();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/simulation/runs/current",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result.state).toBe("idle");
    expect(result.run_id).toBeNull();
  });

  it("startSimulation serializes preset/seed/reset_demo_state exactly", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({ ...idleRunStatus(), run_id: "run-1", state: "preparing" }, { status: 202 }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await startSimulation({ preset: "operations_demo", seed: 42, reset_demo_state: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/simulation/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ preset: "operations_demo", seed: 42, reset_demo_state: true }),
      }),
    );
  });

  it("treats HTTP 202 as success and resolves with the reserved run's status", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse({ ...idleRunStatus(), run_id: "run-1", state: "preparing" }, { status: 202 }),
    ) as unknown as typeof fetch;

    const result = await startSimulation({ preset: "operations_demo", seed: 42, reset_demo_state: false });

    expect(result.state).toBe("preparing");
    expect(result.run_id).toBe("run-1");
  });

  it("preserves a 409 SIMULATION_ALREADY_RUNNING conflict as a recognizable ApiError", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(
        { error: { code: "SIMULATION_ALREADY_RUNNING", message: "A simulation run is already running." } },
        { status: 409 },
      ),
    ) as unknown as typeof fetch;

    let caught: unknown;
    try {
      await startSimulation({ preset: "operations_demo", seed: 42, reset_demo_state: false });
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(409);
    expect((caught as ApiError).code).toBe("SIMULATION_ALREADY_RUNNING");
  });

  it("preserves a 403 SIMULATION_CONTROL_DISABLED error as a recognizable ApiError", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(
        { error: { code: "SIMULATION_CONTROL_DISABLED", message: "The Simulation Control API is not enabled." } },
        { status: 403 },
      ),
    ) as unknown as typeof fetch;

    await expect(startSimulation({ preset: "operations_demo", seed: 42, reset_demo_state: false })).rejects.toMatchObject({
      status: 403,
      code: "SIMULATION_CONTROL_DISABLED",
    });
  });

  it("propagates an AbortSignal on startSimulation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(idleRunStatus(), { status: 202 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await startSimulation({ preset: "operations_demo", seed: 42, reset_demo_state: false }, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    );
  });
});
