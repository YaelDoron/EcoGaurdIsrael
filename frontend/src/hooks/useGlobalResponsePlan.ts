import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentGlobalResponsePlan } from "../api/globalResponsePlan";
import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";

const LOAD_ERROR_MESSAGE = "Unable to load the global response plan. Please try again.";

export interface UseGlobalResponsePlanResult {
  /** The backend's response, unchanged. `null` only before the first
   * successful fetch, or after a failed one. A successful `plan: null`
   * (no materialized generation yet) is exposed via `response.plan`, not
   * treated as an error. */
  response: GlobalResponsePlanResponse | null;
  isLoading: boolean;
  /** A safe, static message - never the raw ApiError/backend detail. */
  error: string | null;
  retry: () => void;
}

/**
 * Owns fetch/loading/error state for the single "current" Global Response
 * Plan generation. Performs no business computation of its own (no
 * shortage/metrics aggregation, no route/target enrichment, no per-event
 * grouping) - it only stores exactly what `getCurrentGlobalResponsePlan()`
 * returns.
 *
 * Deliberately takes no arguments: `focusEventId` (Task B-FE-5/B-FE-8) is a
 * purely frontend highlighting/filtering concern owned by the page (via the
 * URL query string) and never triggers a new backend request - this hook
 * fetches the one "current" generation exactly once (plus manual retry) and
 * has no awareness of which event, if any, is focused.
 */
export function useGlobalResponsePlan(): UseGlobalResponsePlanResult {
  const [response, setResponse] = useState<GlobalResponsePlanResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isFetchingRef = useRef(false);
  const requestIdRef = useRef(0);

  const runFetch = useCallback(() => {
    const requestId = ++requestIdRef.current;
    isFetchingRef.current = true;
    setIsLoading(true);
    setError(null);

    getCurrentGlobalResponsePlan()
      .then((result) => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setResponse(result);
      })
      .catch(() => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setResponse(null);
        setError(LOAD_ERROR_MESSAGE);
      })
      .finally(() => {
        if (requestIdRef.current === requestId) {
          setIsLoading(false);
        }
        isFetchingRef.current = false;
      });
  }, []);

  useEffect(() => {
    runFetch();
  }, [runFetch]);

  const retry = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    runFetch();
  }, [runFetch]);

  return { response, isLoading, error, retry };
}
