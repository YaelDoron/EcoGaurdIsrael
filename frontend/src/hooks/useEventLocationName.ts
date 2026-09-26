import { useEffect, useState } from "react";
import { getEventDetails } from "../api/eventDetails";
import { translateIfUntranslated } from "../components/map/stationTranslations";
import { cacheEventLocationName, getCachedEventLocationName } from "./eventLocationNameCache";

/** A FireEvent's persisted location name, from the session cache or its Event Details. */
export function loadEventLocationName(fireEventId: number, signal?: AbortSignal): Promise<string | null> {
  const cached = getCachedEventLocationName(fireEventId);
  if (cached !== undefined) {
    return Promise.resolve(cached);
  }
  return getEventDetails(fireEventId, signal).then((details) => {
    const name = translateIfUntranslated(
      details.detection_evidence.news.find((item) => item.location_name)?.location_name ?? null,
    );
    cacheEventLocationName(fireEventId, name);
    return name;
  });
}

/**
 * The persisted location name for a FireEvent (the first news-evidence
 * `location_name` in its Event Details), or `null` while loading, when none
 * is on record, or when the lookup fails. A location name is decoration for
 * the Response Plan screen, so a failure is silent - never an error state.
 * A name already found this session is returned without a request, and an
 * in-flight lookup is aborted on unmount/id change.
 */
export function useEventLocationName(fireEventId: number | null): string | null {
  const [state, setState] = useState<{ id: number; name: string | null } | null>(null);

  useEffect(() => {
    if (fireEventId === null) {
      return;
    }
    const controller = new AbortController();
    loadEventLocationName(fireEventId, controller.signal)
      .then((name) => {
        if (!controller.signal.aborted) setState({ id: fireEventId, name });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState({ id: fireEventId, name: null });
      });
    return () => controller.abort();
  }, [fireEventId]);

  if (fireEventId === null) {
    return null;
  }
  if (state !== null && state.id === fireEventId) {
    return state.name;
  }
  return getCachedEventLocationName(fireEventId) ?? null;
}
