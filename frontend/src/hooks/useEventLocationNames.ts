import { useEffect, useState } from "react";
import { getEventDetails } from "../api/eventDetails";

/**
 * Persisted location names for several FireEvents at once (event id -> the
 * first news-evidence `location_name` in its Event Details). An id is absent
 * while its lookup is in flight and `null` when none is on record or the
 * lookup failed - a name is decoration, so a failure is silent and callers
 * fall back to "Event #id".
 */
export function useEventLocationNames(fireEventIds: number[]): Record<number, string | null> {
  const [names, setNames] = useState<Record<number, string | null>>({});
  const signature = fireEventIds.join(",");

  useEffect(() => {
    let cancelled = false;
    for (const id of fireEventIds) {
      getEventDetails(id)
        .then((details) => details.detection_evidence.news.find((item) => item.location_name)?.location_name ?? null)
        .catch(() => null)
        .then((name) => {
          if (!cancelled) {
            setNames((previous) => (previous[id] === name ? previous : { ...previous, [id]: name }));
          }
        });
    }
    return () => {
      cancelled = true;
    };
    // `signature` captures the identity of `fireEventIds`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  return names;
}
