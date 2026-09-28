import { useSyncExternalStore } from "react";

/**
 * A Start Simulation click is "pending" from the click until the NEW run has
 * left PREPARING. The backend only wipes the previous run's rows inside
 * PREPARING (the mandatory reset), so for that whole window the overview and
 * plan reads still return the previous run's fires - and this tab still
 * remembers that run, which would keep showing them. While pending,
 * hooks/demoSession.ts shows nothing persisted: the pages fall back to their
 * empty/loading state instantly instead of flashing the old data.
 *
 * `pendingRunId` is null until the start request has been accepted (the new
 * run's id is not known before that). Import-free on purpose (the test setup
 * resets it without loading the API layer).
 */
let startPending = false;
let pendingRunId: string | null = null;
let version = 0;
const listeners = new Set<() => void>();

function setState(pending: boolean, runId: string | null): void {
  if (startPending === pending && pendingRunId === runId) {
    return;
  }
  startPending = pending;
  pendingRunId = runId;
  version += 1;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getVersion(): number {
  return version;
}

/** Call synchronously on the Start Simulation click, before the request is sent. */
export function beginDemoRunStart(): void {
  setState(true, null);
}

/** The start request was accepted: keep hiding the old data until this run leaves PREPARING. */
export function confirmDemoRunStart(runId: string | null | undefined): void {
  if (startPending) {
    setState(true, runId ?? null);
  }
}

/** The start request was rejected, or the new run is ready: stop hiding. */
export function endDemoRunStart(): void {
  setState(false, null);
}

export function isDemoRunStartPending(): boolean {
  return startPending;
}

export function getPendingDemoRunId(): string | null {
  return pendingRunId;
}

/** Re-renders the caller whenever the pending state changes. */
export function useDemoRunStartVersion(): number {
  return useSyncExternalStore(subscribe, getVersion, getVersion);
}
