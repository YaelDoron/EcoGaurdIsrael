import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentResponsePlan, getResponsePlanById } from "../api/responsePlans";
import type { ResponsePlan } from "../types/responsePlan";

const LOAD_ERROR_MESSAGE = "Unable to load the response plan. Please try again.";

/**
 * Which of US 6.3's two data sources to fetch. The page decides which
 * variant applies based only on which route parameter is present - this
 * hook has no routing awareness of its own.
 */
export type ResponsePlanSource = { kind: "current"; fireEventId: number } | { kind: "by-id"; planId: number };

export interface UseResponsePlanResult {
  /** The backend's `plan` field, unchanged. `null` is a valid, successful
   * "no current plan" result for the `current` source - never an error. */
  plan: ResponsePlan | null;
  isLoading: boolean;
  /** A safe, static message - never the raw ApiError/backend detail. */
  error: string | null;
  retry: () => void;
}

function requestKeyOf(source: ResponsePlanSource): string {
  return source.kind === "current" ? `current:${source.fireEventId}` : `by-id:${source.planId}`;
}

/**
 * Owns fetch/loading/error state for one Response Plan, sourced either by
 * FireEvent (current plan) or by plan id. Performs no business computation
 * of its own (no status/currency determination, no metric/ETA/baseline
 * calculation, no action/target filtering) - it only stores exactly what
 * `getCurrentResponsePlan`/`getResponsePlanById` returns.
 *
 * A source/id change is picked up automatically (no manual refetch needed);
 * a failed request for a new source never leaves the previous source's
 * `plan` visible - `plan` is reset to `null` alongside the error.
 */
export function useResponsePlan(source: ResponsePlanSource): UseResponsePlanResult {
  const [plan, setPlan] = useState<ResponsePlan | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isFetchingRef = useRef(false);
  // Guards a stale response (an in-flight request for a source/id that has
  // since changed, or an overtaken manual retry) from overwriting the
  // latest one's state.
  const requestIdRef = useRef(0);

  const requestKey = requestKeyOf(source);

  const runFetch = useCallback(() => {
    const requestId = ++requestIdRef.current;
    isFetchingRef.current = true;
    setIsLoading(true);
    setError(null);

    const request =
      source.kind === "current" ? getCurrentResponsePlan(source.fireEventId) : getResponsePlanById(source.planId);

    request
      .then((response) => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setPlan(response.plan);
      })
      .catch(() => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setPlan(null);
        setError(LOAD_ERROR_MESSAGE);
      })
      .finally(() => {
        if (requestIdRef.current === requestId) {
          setIsLoading(false);
        }
        isFetchingRef.current = false;
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey]);

  useEffect(() => {
    // Always runs for a genuinely new source/id - never skipped by the
    // duplicate-request guard below, which only protects the manual retry()
    // path from a fast double click.
    runFetch();
  }, [runFetch]);

  const retry = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    runFetch();
  }, [runFetch]);

  return { plan, isLoading, error, retry };
}
