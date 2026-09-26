import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentGlobalResponsePlan } from "../api/globalResponsePlan";
import { GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS } from "../config/polling";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";

const LOAD_ERROR_MESSAGE = "Unable to load the global response plan. Please try again.";
const REFRESH_ERROR_MESSAGE = "Unable to refresh the global response plan. Showing the last loaded plan.";

export interface UseGlobalResponsePlanResult {
  /** The backend's response, unchanged. `null` only before the first
   * successful fetch, or after a failed first fetch. A successful `plan: null`
   * (no materialized generation yet) is exposed via `response.plan`, not
   * treated as an error. A failed poll never clears it. */
  response: GlobalResponsePlanResponse | null;
  /** True only for the very first fetch, before any response has loaded. */
  isLoading: boolean;
  /** True while a poll/refresh is in flight; the previous `response` stays visible/unchanged. */
  isRefreshing: boolean;
  /** Set only when there is nothing to show (the first fetch failed). A safe,
   * static message - never the raw ApiError/backend detail. */
  error: string | null;
  /** Set when a poll fails but the previously loaded `response` is still shown. */
  refreshError: string | null;
  retry: () => void;
}

/**
 * Keeps the previous `plan` object (same identity) when a poll returned an
 * identical one. Consumers memoize map bounds, geocoding targets and layers
 * on `plan.events`, so a new-but-equal object every poll would re-frame the
 * map and re-trigger location lookups every few seconds for no reason.
 * `as_of` still updates.
 */
function keepPlanIdentityIfUnchanged(
  previous: GlobalResponsePlanResponse | null,
  next: GlobalResponsePlanResponse,
): GlobalResponsePlanResponse {
  if (previous === null || previous.plan === null || next.plan === null) {
    return next;
  }
  if (JSON.stringify(previous.plan) !== JSON.stringify(next.plan)) {
    return next;
  }
  return { ...next, plan: previous.plan };
}

/**
 * Owns fetch/poll/loading/error state for the single "current" Global
 * Response Plan generation. Performs no business computation of its own (no
 * shortage/metrics aggregation, no route/target enrichment, no per-event
 * grouping) - it only stores exactly what `getCurrentGlobalResponsePlan()`
 * returns.
 *
 * Deliberately takes no arguments: `focusEventId` (Task B-FE-5/B-FE-8) is a
 * purely frontend highlighting/filtering concern owned by the page (via the
 * URL query string) and never triggers a new backend request - this hook
 * has no awareness of which event, if any, is focused.
 *
 * Polling: once a response has loaded, the next fetch is scheduled
 * GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS after the current request settles -
 * not a fixed `setInterval`, so at most one request is in flight at a time
 * (same strategy as useEventDetails/useOperationsOverview). A failed poll
 * keeps the previous `response` visible (`refreshError`) and polling
 * continues. If the very first fetch fails there is nothing to show and no
 * poll is scheduled; `retry()` starts a fresh load. Unmount clears the
 * timer, and a `generationRef` makes any request still in flight a no-op,
 * so nothing updates state or reschedules afterwards.
 */
export interface UseGlobalResponsePlanOptions {
  /** When false nothing is fetched or polled (e.g. the demo-session gate
   * already knows the plan will not be shown). Defaults to true. */
  enabled?: boolean;
  /** Whether to keep re-fetching every GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS
   * after the first load. Defaults to true. Switching it back on re-fetches at
   * once and resumes polling. */
  polling?: boolean;
}

export function useGlobalResponsePlan(options: UseGlobalResponsePlanOptions = {}): UseGlobalResponsePlanResult {
  const { enabled = true, polling = true } = options;
  const pollingRef = useRef(polling);
  pollingRef.current = polling;
  const [response, setResponse] = useState<GlobalResponsePlanResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const hasDataRef = useRef(false);
  const isFetchingRef = useRef(false);
  const timeoutIdRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bumped on unmount, so a request started before it can tell it has been
  // superseded and skip both its state updates and its poll rescheduling.
  const generationRef = useRef(0);
  // The in-flight request, aborted on unmount so navigating away frees the
  // browser connection instead of waiting out a slow read in the background.
  const abortRef = useRef<AbortController | null>(null);

  const clearScheduled = useCallback(() => {
    if (timeoutIdRef.current !== null) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
  }, []);

  const runFetch = useCallback(
    function runFetch() {
      if (isFetchingRef.current) {
        // Avoid duplicate simultaneous requests (e.g. a fast double click on Retry).
        return;
      }
      isFetchingRef.current = true;
      timeoutIdRef.current = null; // a fired (or cleared) poll timer is no longer pending
      const generation = generationRef.current;

      const isRefresh = hasDataRef.current;
      if (isRefresh) {
        setIsRefreshing(true);
        setRefreshError(null);
      } else {
        setIsLoading(true);
        setError(null);
      }

      const controller = new AbortController();
      abortRef.current = controller;
      getCurrentGlobalResponsePlan(controller.signal)
        .then((result) => {
          if (generation !== generationRef.current) {
            return;
          }
          setResponse((previous) => keepPlanIdentityIfUnchanged(previous, result));
          hasDataRef.current = true;
          setError(null);
          setRefreshError(null);
        })
        .catch(() => {
          if (generation !== generationRef.current) {
            return;
          }
          if (isRefresh) {
            setRefreshError(REFRESH_ERROR_MESSAGE);
          } else {
            setResponse(null);
            setError(LOAD_ERROR_MESSAGE);
          }
        })
        .finally(() => {
          if (generation !== generationRef.current) {
            // Superseded (unmounted) - nothing to update or schedule.
            return;
          }
          isFetchingRef.current = false;
          setIsLoading(false);
          setIsRefreshing(false);
          if (hasDataRef.current && pollingRef.current) {
            timeoutIdRef.current = setTimeout(runFetch, GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS);
          }
        });
    },
    [],
  );

  useEffect(() => {
    generationRef.current += 1;
    clearScheduled();
    isFetchingRef.current = false;
    hasDataRef.current = false;
    if (!enabled) {
      setIsLoading(false);
      return undefined;
    }
    runFetch();

    return () => {
      generationRef.current += 1;
      clearScheduled();
      isFetchingRef.current = false;
      abortRef.current?.abort();
    };
  }, [runFetch, clearScheduled, enabled]);

  // Polling switched OFF: cancel the pending re-check. Switched back ON (e.g. a
  // new run went live): refresh now and resume. Only real off->on transitions
  // re-fetch - never the initial mount or an `enabled` change.
  const previousPollingRef = useRef(polling);
  useEffect(() => {
    const wasPolling = previousPollingRef.current;
    previousPollingRef.current = polling;
    if (!polling) {
      clearScheduled();
      return;
    }
    if (!wasPolling && enabled && hasDataRef.current && timeoutIdRef.current === null && !isFetchingRef.current) {
      runFetch();
    }
  }, [enabled, polling, runFetch, clearScheduled]);

  const retry = useCallback(() => {
    clearScheduled();
    runFetch();
  }, [runFetch, clearScheduled]);

  return { response, isLoading, isRefreshing, error, refreshError, retry };
}
