import { useCallback, useEffect, useRef, useState } from "react";
import { getEventDetails } from "../api/eventDetails";
import { ApiError } from "../api/errors";
import type { EventDetailsResult } from "../types/eventDetails";

const LOAD_ERROR_MESSAGE = "Unable to load this event's details. Please try again.";
const REFRESH_ERROR_MESSAGE = "Unable to refresh this event's details. Showing previously loaded data.";
const NOT_FOUND_STATUS = 404;

export interface UseEventDetailsResult {
  data: EventDetailsResult | null;
  /** True only for the very first fetch of the current fireEventId, before any data has loaded. */
  isLoading: boolean;
  /** True while a refresh is in flight; previous `data` remains available/unchanged. */
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

  const load = useCallback(() => {
    if (isFetchingRef.current) {
      // Avoid duplicate simultaneous requests (e.g. a fast double click on Refresh).
      return;
    }
    isFetchingRef.current = true;

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
        setData(response);
        hasDataRef.current = true;
        setLoadError(null);
        setRefreshError(null);
        setNotFound(false);
      })
      .catch((error: unknown) => {
        if (error instanceof ApiError && error.status === NOT_FOUND_STATUS) {
          setNotFound(true);
          setData(null);
          hasDataRef.current = false;
          return;
        }
        if (isRefresh) {
          setRefreshError(REFRESH_ERROR_MESSAGE);
        } else {
          setLoadError(LOAD_ERROR_MESSAGE);
        }
      })
      .finally(() => {
        isFetchingRef.current = false;
        setIsLoading(false);
        setIsRefreshing(false);
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
    hasDataRef.current = false;
    load();
  }, [load]);

  return { data, isLoading, isRefreshing, loadError, refreshError, notFound, refresh: load };
}
