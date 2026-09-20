import { useCallback, useEffect, useRef, useState } from "react";
import { getOperationsOverview } from "../api/operations";
import { OPERATIONS_OVERVIEW_POLL_INTERVAL_MS } from "../config/polling";
import type { OperationsOverviewResponse } from "../types/operationsOverview";

const LOAD_ERROR_MESSAGE = "Unable to load the operations overview. Please try again.";
const REFRESH_ERROR_MESSAGE = "Unable to refresh the operations overview. Showing previously loaded data.";

export interface UseOperationsOverviewOptions {
  activityLimit?: number;
  /** Set false to fetch once and never poll. Defaults to true. */
  pollingEnabled?: boolean;
}

export interface UseOperationsOverviewResult {
  data: OperationsOverviewResponse | null;
  /** True only for the very first fetch, before any data has ever loaded. */
  isLoading: boolean;
  /** True while a poll/refresh is in flight after the first load; previous `data` remains visible/unchanged. */
  isRefreshing: boolean;
  /** Set only when there is no data to show at all (initial load failed). */
  loadError: string | null;
  /** Set when a background poll/refresh fails but previously loaded `data` is still valid and shown. */
  refreshError: string | null;
  refresh: () => void;
}

/**
 * Owns fetch/poll/loading/error state for the Operations Overview
 * dashboard (Task A6/A7). Performs no wildfire business logic itself - it
 * only fetches and stores exactly what `getOperationsOverview()` returns.
 *
 * Polling strategy (Task A7, Part 9): the next poll is scheduled with
 * `setTimeout` only AFTER the current request fully settles (success or
 * failure) - never a fixed `setInterval`, which would fire on schedule
 * regardless of whether the previous request is still in flight. This
 * guarantees at most one Operations Overview request in flight at a time,
 * by construction, even though a real request currently takes
 * 3.1-4.6 seconds against the demo database (see
 * OPERATIONS_OVERVIEW_POLL_INTERVAL_MS's own docstring). A manual
 * `refresh()` call is likewise guarded (`isFetchingRef`) so a fast double
 * click cannot start a second overlapping request.
 *
 * A failed poll never discards already-visible valid data: `data` only
 * ever changes on a successful fetch, so a transient failure surfaces
 * through `refreshError` while `data` stays exactly as it was, and polling
 * continues automatically on the next scheduled tick (no special retry
 * logic needed - the loop itself is the recovery mechanism).
 *
 * Every in-flight request carries an `AbortController` signal, aborted on
 * unmount and whenever `activityLimit`/`pollingEnabled` changes (a stale
 * generation's eventual response/rejection is also ignored via
 * `generationRef`, so no state update - and therefore no poll rescheduling
 * - ever happens from a superseded request).
 */
export function useOperationsOverview(options: UseOperationsOverviewOptions = {}): UseOperationsOverviewResult {
  const { activityLimit, pollingEnabled = true } = options;

  const [data, setData] = useState<OperationsOverviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const hasDataRef = useRef(false);
  const isFetchingRef = useRef(false);
  const timeoutIdRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  // Bumped on unmount and on every activityLimit/pollingEnabled change, so
  // an in-flight request started under a previous configuration can detect
  // it has been superseded and skip both its state updates and its own
  // poll-rescheduling.
  const generationRef = useRef(0);

  const clearScheduled = useCallback(() => {
    if (timeoutIdRef.current !== null) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
  }, []);

  /**
   * Aborts any in-flight request AND releases `isFetchingRef` in the same
   * step. Without releasing it here, React StrictMode's development-only
   * mount -> cleanup -> remount cycle (or any other case where the effect
   * re-runs while a request is still in flight) leaves `isFetchingRef`
   * stuck `true` from the just-abandoned request, so the very next
   * `runFetch()` call - the real one - is silently dropped by its own
   * `if (isFetchingRef.current) return;` guard, and the abandoned request's
   * own `.finally()` later bails out on the generation check before ever
   * clearing `isLoading` or scheduling the next poll. That combination is
   * exactly what produced an indefinite "Loading operations overview..."
   * in `npm run dev` despite the backend responding normally.
   */
  const abortInFlight = useCallback(() => {
    abortControllerRef.current?.abort();
    isFetchingRef.current = false;
  }, []);

  const runFetch = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;
    const generation = generationRef.current;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const isRefresh = hasDataRef.current;
    if (isRefresh) {
      setIsRefreshing(true);
      setRefreshError(null);
    } else {
      setIsLoading(true);
      setLoadError(null);
    }

    getOperationsOverview({ activityLimit, signal: controller.signal })
      .then((response) => {
        if (generation !== generationRef.current) {
          return;
        }
        setData(response);
        hasDataRef.current = true;
        setLoadError(null);
        setRefreshError(null);
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          // Intentional cancellation (unmount, or a config change starting
          // a fresh generation) - not a failure, never surfaced as one.
          return;
        }
        if (generation !== generationRef.current) {
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
        if (generation !== generationRef.current) {
          // Superseded - the effect that started the new generation already
          // owns scheduling from here.
          return;
        }
        setIsLoading(false);
        setIsRefreshing(false);
        if (pollingEnabled) {
          timeoutIdRef.current = setTimeout(runFetch, OPERATIONS_OVERVIEW_POLL_INTERVAL_MS);
        }
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activityLimit, pollingEnabled]);

  useEffect(() => {
    generationRef.current += 1;
    clearScheduled();
    abortInFlight();
    runFetch();

    return () => {
      generationRef.current += 1;
      clearScheduled();
      abortInFlight();
    };
  }, [runFetch, clearScheduled, abortInFlight]);

  const refresh = useCallback(() => {
    clearScheduled();
    runFetch();
  }, [runFetch, clearScheduled]);

  return { data, isLoading, isRefreshing, loadError, refreshError, refresh };
}
