import { useCallback, useEffect, useRef, useState } from "react";
import { getEventDetails } from "../api/eventDetails";
import { ApiError } from "../api/errors";
import { EVENT_DETAILS_POLL_INTERVAL_MS } from "../config/polling";
import type { EventDetailsResult } from "../types/eventDetails";
import type { FireEventStatus } from "../types/fireEvent";

const LOAD_ERROR_MESSAGE = "Unable to load this event's details. Please try again.";
const REFRESH_ERROR_MESSAGE = "Unable to refresh this event's details. Showing previously loaded data.";
const NOT_FOUND_STATUS = 404;

/** An event still worth polling: it can still change status or gain a response plan. */
function isActiveStatus(status: FireEventStatus): boolean {
  return status === "suspected" || status === "confirmed";
}

export interface UseEventDetailsResult {
  data: EventDetailsResult | null;
  /** True only for the very first fetch of the current fireEventId, before any data has loaded. */
  isLoading: boolean;
  /** True while a refresh/poll is in flight; previous `data` remains available/unchanged. */
  isRefreshing: boolean;
  /** Set only when there is no data to show at all (initial load failed, and it was not a 404). */
  loadError: string | null;
  /** Set when a refresh fails but previously loaded `data` is still valid and shown. */
  refreshError: string | null;
  /** True when the backend reported no FireEvent exists for this id (a 404). */
  notFound: boolean;
  refresh: () => void;
}

/**
 * Owns fetch/loading/error/refresh state for the Event Details page.
 * Performs no wildfire business logic itself - it only fetches and stores
 * exactly what getEventDetails() returns.
 *
 * A missing FireEvent (404) is tracked separately via `notFound` rather than
 * folded into `loadError`, so the page can render a distinct "Event Not
 * Found" state instead of a generic retryable error - retrying an unknown
 * id would only ever 404 again.
 *
 * A refresh failure never discards already-visible valid data: `data` only
 * ever changes on a successful fetch, so a failed refresh surfaces through
 * `refreshError` while the previous `data` stays exactly as it was.
 *
 * Polling: while the loaded event is active (suspected or confirmed), the
 * next fetch is scheduled EVENT_DETAILS_POLL_INTERVAL_MS after the current
 * request settles - not a fixed `setInterval`, so at most one request is in
 * flight at a time even when the backend is slow (same strategy and
 * rationale as useOperationsOverview). Polling stops once the event is
 * resolved/dismissed or gone (404), continues through a transient failed
 * poll, and is cancelled on unmount / fireEventId change: the timer is
 * cleared and a `generationRef` makes any still-in-flight request's
 * eventual result a no-op, so nothing updates state or reschedules after
 * that.
 */
export function useEventDetails(fireEventId: number): UseEventDetailsResult {
  const [data, setData] = useState<EventDetailsResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  const hasDataRef = useRef(false);
  const isFetchingRef = useRef(false);
  // Whether the last successfully loaded event was still active - gates
  // scheduling the next poll. A failed poll leaves it untouched, so polling
  // keeps going through a transient error.
  const isActiveRef = useRef(false);
  const timeoutIdRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bumped on unmount and whenever fireEventId changes, so a request started
  // for a previous event (or before unmount) can tell it has been superseded.
  const generationRef = useRef(0);

  const clearScheduled = useCallback(() => {
    if (timeoutIdRef.current !== null) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
  }, []);

  const load = useCallback(function runLoad() {
    if (isFetchingRef.current) {
      // Avoid duplicate simultaneous requests (e.g. a fast double click on Refresh).
      return;
    }
    isFetchingRef.current = true;
    const generation = generationRef.current;

    const isRefresh = hasDataRef.current;
    if (isRefresh) {
      setIsRefreshing(true);
      setRefreshError(null);
    } else {
      setIsLoading(true);
      setLoadError(null);
      setNotFound(false);
    }

    getEventDetails(fireEventId)
      .then((response) => {
        if (generation !== generationRef.current) {
          return;
        }
        setData(response);
        hasDataRef.current = true;
        isActiveRef.current = isActiveStatus(response.fire_event.status);
        setLoadError(null);
        setRefreshError(null);
        setNotFound(false);
      })
      .catch((error: unknown) => {
        if (generation !== generationRef.current) {
          return;
        }
        if (error instanceof ApiError && error.status === NOT_FOUND_STATUS) {
          setNotFound(true);
          setData(null);
          hasDataRef.current = false;
          isActiveRef.current = false;
          return;
        }
        if (isRefresh) {
          setRefreshError(REFRESH_ERROR_MESSAGE);
        } else {
          setLoadError(LOAD_ERROR_MESSAGE);
        }
      })
      .finally(() => {
        if (generation !== generationRef.current) {
          // Superseded - the effect that started the new generation already
          // reset isFetchingRef and owns loading state and scheduling.
          return;
        }
        isFetchingRef.current = false;
        setIsLoading(false);
        setIsRefreshing(false);
        if (isActiveRef.current) {
          timeoutIdRef.current = setTimeout(runLoad, EVENT_DETAILS_POLL_INTERVAL_MS);
        }
      });
  }, [fireEventId]);

  useEffect(() => {
    // `load` already changes identity whenever fireEventId changes, so this
    // effect re-runs (and re-fetches from scratch) on navigation between two
    // different events' detail pages. Resetting hasDataRef here (a ref, not
    // state) makes that fetch report as a fresh load - not a refresh - so
    // `isLoading` becomes true immediately; `data` itself is left as-is
    // rather than cleared, since every consumer (see EventDetailsPage) is
    // expected to gate on `isLoading` before reading `data` anyway.
    generationRef.current += 1;
    clearScheduled();
    isFetchingRef.current = false;
    hasDataRef.current = false;
    isActiveRef.current = false;
    load();

    return () => {
      generationRef.current += 1;
      clearScheduled();
      isFetchingRef.current = false;
    };
  }, [load, clearScheduled]);

  const refresh = useCallback(() => {
    clearScheduled();
    load();
  }, [load, clearScheduled]);

  return { data, isLoading, isRefreshing, loadError, refreshError, notFound, refresh };
}
