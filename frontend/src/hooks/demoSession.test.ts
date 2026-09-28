import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { SimulationRunStatus } from "../types/simulation";
import { beginDemoRunStart, confirmDemoRunStart, endDemoRunStart } from "./demoRunStart";
import { getRememberedDemoRunId, isDemoDataVisible, rememberDemoRun } from "./demoSession";

function run(run_id: string | null, state: SimulationRunStatus["state"]): SimulationRunStatus {
  return { run_id, state } as SimulationRunStatus;
}

describe("demoSession", () => {
  beforeEach(() => window.sessionStorage.clear());
  afterEach(() => window.sessionStorage.clear());

  it("hides persisted demo data when no run exists in this browser session", () => {
    expect(isDemoDataVisible({ enabled: true, run: null })).toBe(false);
    expect(isDemoDataVisible({ enabled: true, run: run(null, "idle") })).toBe(false);
    expect(isDemoDataVisible({ enabled: true, run: run("old", "completed") })).toBe(false);
    expect(isDemoDataVisible({ enabled: true, run: run("old", "failed") })).toBe(false);
  });

  it("always shows a PREPARING/RUNNING run", () => {
    expect(isDemoDataVisible({ enabled: true, run: run("live", "preparing") })).toBe(true);
    expect(isDemoDataVisible({ enabled: true, run: run("live", "running") })).toBe(true);
  });

  it("shows a completed run only when this session remembered that exact run", () => {
    rememberDemoRun("mine");
    expect(getRememberedDemoRunId()).toBe("mine");
    expect(isDemoDataVisible({ enabled: true, run: run("mine", "completed") })).toBe(true);
    expect(isDemoDataVisible({ enabled: true, run: run("mine", "completed_with_errors") })).toBe(true);
    expect(isDemoDataVisible({ enabled: true, run: run("someone-else", "completed") })).toBe(false);
  });

  it("hides the remembered previous run from the Start click until the new run leaves PREPARING", () => {
    rememberDemoRun("old");
    expect(isDemoDataVisible({ enabled: true, run: run("old", "completed") })).toBe(true);

    beginDemoRunStart();
    expect(isDemoDataVisible({ enabled: true, run: run("old", "completed") })).toBe(false);

    confirmDemoRunStart("new");
    expect(isDemoDataVisible({ enabled: true, run: run("old", "completed") })).toBe(false);
    expect(isDemoDataVisible({ enabled: true, run: run("new", "preparing") })).toBe(false);
    expect(isDemoDataVisible({ enabled: true, run: run("new", "running") })).toBe(true);
    expect(isDemoDataVisible({ enabled: false, run: null })).toBe(true);
  });

  it("shows the previous run again when the start request is rejected", () => {
    rememberDemoRun("old");
    beginDemoRunStart();
    endDemoRunStart();
    expect(isDemoDataVisible({ enabled: true, run: run("old", "completed") })).toBe(true);
  });

  it("never hides anything when simulation control is disabled", () => {
    expect(isDemoDataVisible({ enabled: false, run: null })).toBe(true);
  });

  it("ignores empty run ids", () => {
    rememberDemoRun(null);
    rememberDemoRun("");
    expect(getRememberedDemoRunId()).toBeNull();
  });
});
