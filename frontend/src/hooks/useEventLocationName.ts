import { useEffect, useState } from "react";
import { getEventDetails } from "../api/eventDetails";

/**
 * The persisted location name for a FireEvent (the first news-evidence
 * `location_name` in its Event Details), or `null` while loading, when none
 * is on record, or when the lookup fails. A location name is decoration for
 * the Response Plan screen, so a failure is silent - never an error state.
 */
export function useEventLocationName(fireEventId: number | null): string | null {
  const [state, setState] = useState<{ id: number; name: string | null } | null>(null);

  useEffect(() => {
    if (fireEventId === null) {
      return;
    }
    let cancelled = false;
    getEventDetails(fireEventId)
      .then((details) => {
        if (cancelled) return;
        const name = details.detection_evidence.news.find((item) => item.location_name)?.location_name ?? null;
        setState({ id: fireEventId, name });
      })
      .catch(() => {
        if (!cancelled) setState({ id: fireEventId, name: null });
      });
    return () => {
      cancelled = true;
    };
  }, [fireEventId]);

  return state !== null && state.id === fireEventId ? state.name : null;
}
