/**
 * Session-lifetime store of FireEvent location names already found (the
 * first news-evidence `location_name` in an event's Event Details), shared
 * by the response-plan pages' label hooks (useEventLocationName/s).
 *
 * A found name never changes for an event, so returning to a plan page does
 * not re-issue a full Event Details read (several seconds against the
 * remote database) just to relabel it. Only real names are stored: "no news
 * evidence yet" or a failed lookup is not, so a later visit can still pick
 * the name up. Deliberately free of imports (the loader lives in
 * useEventLocationName.ts) so the test setup can clear it without loading
 * the API layer before a test's own module mocks.
 */
const namesByEventId = new Map<number, string>();

export function getCachedEventLocationName(fireEventId: number): string | undefined {
  return namesByEventId.get(fireEventId);
}

export function cacheEventLocationName(fireEventId: number, name: string | null): void {
  if (name) {
    namesByEventId.set(fireEventId, name);
  }
}

/** Test-only: forget every cached name. */
export function clearEventLocationNameCache(): void {
  namesByEventId.clear();
}
