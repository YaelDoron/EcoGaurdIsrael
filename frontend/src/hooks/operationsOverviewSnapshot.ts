import type { OperationsOverviewResponse } from "../types/operationsOverview";

/**
 * The most recent Operations Overview response of this browser session, so
 * navigating back to the dashboard (e.g. "Back to Active Fires" from a plan
 * page) renders the last known state at once while useOperationsOverview
 * refetches in the background, instead of blanking the whole page behind
 * "Loading..." for a full round of remote reads. Bounded staleness: a
 * snapshot older than MAX_SNAPSHOT_AGE_MS is not reused, and a reused one is
 * replaced by the immediate refetch. Import-free on purpose (the test setup
 * clears it without loading the API layer).
 */
export const MAX_SNAPSHOT_AGE_MS = 120_000;

interface Snapshot {
  activityLimit: number | undefined;
  data: OperationsOverviewResponse;
  savedAt: number;
}

let snapshot: Snapshot | null = null;

export function readOverviewSnapshot(activityLimit: number | undefined): OperationsOverviewResponse | null {
  if (snapshot === null || snapshot.activityLimit !== activityLimit) {
    return null;
  }
  return Date.now() - snapshot.savedAt <= MAX_SNAPSHOT_AGE_MS ? snapshot.data : null;
}

export function saveOverviewSnapshot(activityLimit: number | undefined, data: OperationsOverviewResponse): void {
  snapshot = { activityLimit, data, savedAt: Date.now() };
}

/** Test-only: forget the snapshot. */
export function clearOverviewSnapshot(): void {
  snapshot = null;
}
