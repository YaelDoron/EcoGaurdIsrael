import { useEffect, useState } from "react";
import { getCurrentSimulationRun } from "../api/simulation";
import { ApiError } from "../api/errors";
import type { SimulationRunStatus } from "../types/simulation";

/**
 * Browser-session presentation gate for demo runtime data (Task: final demo
 * clean-up, Part A).
 *
 * Demo runtime rows (fire events, evidence, plans) stay in Neon after a run
 * finishes - the mandatory reset only happens when the NEXT run starts. So a
 * freshly opened dashboard would otherwise show the previous demo's events
 * before anyone pressed Start Simulation. This module decides, purely on the
 * client, whether that persisted data belongs to "this session's" run:
 *
 *   backend run is PREPARING/RUNNING           -> show (and remember its run_id)
 *   run_id matches the one this tab remembered -> show (inspecting its results)
 *   otherwise                                  -> initial "No simulation started" state
 *
 * Presentation only: nothing here resets or deletes backend data, so a
 * refresh during a running simulation can never lose events. `sessionStorage`
 * survives refresh/navigation in the same tab and starts empty in a new
 * browser session/tab. When simulation control is disabled (not a demo
 * deployment) nothing is ever hidden.
 */
const DEMO_RUN_STORAGE_KEY = "ecoguard.demoSession.runId";

const LIVE_STATES: ReadonlySet<SimulationRunStatus["state"]> = new Set(["preparing", "running"]);

export interface DemoSimulationSummary {
  enabled: boolean;
  run: SimulationRunStatus | null;
}

export function rememberDemoRun(runId: string | null | undefined): void {
  if (!runId) {
    return;
  }
  try {
    window.sessionStorage.setItem(DEMO_RUN_STORAGE_KEY, runId);
  } catch {
    // Storage unavailable (private mode / blocked) - live runs still show via LIVE_STATES.
  }
}

export function getRememberedDemoRunId(): string | null {
  try {
    return window.sessionStorage.getItem(DEMO_RUN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function isLiveRun(run: SimulationRunStatus | null): boolean {
  return run !== null && run.run_id !== null && LIVE_STATES.has(run.state);
}

/** Whether persisted demo runtime data should be shown in this browser session. */
export function isDemoDataVisible(simulation: DemoSimulationSummary): boolean {
  if (!simulation.enabled) {
    return true;
  }
  const run = simulation.run;
  if (run === null || run.run_id === null) {
    return false;
  }
  return isLiveRun(run) || getRememberedDemoRunId() === run.run_id;
}

/**
 * `isDemoDataVisible` plus the one side effect: a run observed while live is
 * remembered, so this tab keeps showing its results after it completes.
 */
export function useDemoDataVisibility(simulation: DemoSimulationSummary | null): boolean {
  const run = simulation?.run ?? null;
  const liveRunId = isLiveRun(run) ? run?.run_id : null;

  useEffect(() => {
    rememberDemoRun(liveRunId);
  }, [liveRunId]);

  return simulation === null ? true : isDemoDataVisible(simulation);
}

/**
 * One-shot read of the current simulation run for pages that do not get it
 * from the Operations Overview (e.g. the Global Response Plan page). A 403
 * means simulation control is disabled (never hide data); any other failure,
 * or still loading, yields `null`, which `useDemoDataVisibility` treats as
 * "show" - the gate fails open rather than hiding real data.
 */
export function useCurrentDemoSimulation(): DemoSimulationSummary | null {
  const [summary, setSummary] = useState<DemoSimulationSummary | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getCurrentSimulationRun(controller.signal)
      .then((run) => setSummary({ enabled: true, run }))
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === 403) {
          setSummary({ enabled: false, run: null });
        }
      });
    return () => controller.abort();
  }, []);

  return summary;
}
