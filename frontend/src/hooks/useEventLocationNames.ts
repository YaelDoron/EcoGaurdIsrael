import { useEffect, useState } from "react";
import { getCachedEventLocationName } from "./eventLocationNameCache";
import { loadEventLocationName } from "./useEventLocationName";

/**
 * Persisted location names for several FireEvents at once (event id -> the
 * first news-evidence `location_name` in its Event Details). An id is absent
 * while its lookup is in flight and `null` when none is on record or the
 * lookup failed - a name is decoration, so a failure is silent and callers
 * fall back to "Event #id". Names already found this session are returned
 * immediately (eventLocationNameCache); lookups still in flight are aborted
 * when the ids change or the page unmounts.
 */
export function useEventLocationNames(fireEventIds: number[]): Record<number, string | null> {
  const [names, setNames] = useState<Record<number, string | null>>(() => cachedNames(fireEventIds));
  const signature = fireEventIds.join(",");

  useEffect(() => {
    const controller = new AbortController();
    for (const id of fireEventIds) {
      loadEventLocationName(id, controller.signal)
        .catch(() => null)
        .then((name) => {
          if (!controller.signal.aborted) {
            setNames((previous) => (previous[id] === name ? previous : { ...previous, [id]: name }));
          }
        });
    }
    return () => controller.abort();
    // `signature` captures the identity of `fireEventIds`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  return names;
}

function cachedNames(fireEventIds: number[]): Record<number, string | null> {
  const names: Record<number, string | null> = {};
  for (const id of fireEventIds) {
    const cached = getCachedEventLocationName(id);
    if (cached !== undefined) {
      names[id] = cached;
    }
  }
  return names;
}
