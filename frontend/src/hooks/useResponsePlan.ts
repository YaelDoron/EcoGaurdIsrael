import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentResponsePlan, getResponsePlanById } from "../api/responsePlans";
import { RESPONSE_PLAN_PENDING_POLL_INTERVAL_MS } from "../config/polling";
import type { ResponsePlan, ResponsePlanEnvelopeResponse } from "../types/responsePlan";

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
  /** The backend's `plan_status` (current source only): `generating` means the
   * event is CONFIRMED and its plan is still being produced - the hook then
   * keeps re-fetching quietly until the plan appears. */
  planStatus: NonNullable<ResponsePlanEnvelopeResponse["plan_status"]> | null;
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
  const [planStatus, setPlanStatus] = useState<UseResponsePlanResult["planStatus"]>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isFetchingRef = useRef(false);
  // Guards a stale response (an in-flight request for a source/id that has
  // since changed, or an overtaken manual retry) from overwriting the
  // latest one's state.
  const requestIdRef = useRef(0);
  // The in-flight request, aborted when superseded or on unmount (frees the
  // browser connection when the user navigates away mid-load).
  const abortRef = useRef<AbortController | null>(null);
  // Quiet re-fetch while the plan is still being generated (no loading flash).
  const pendingPollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const requestKey = requestKeyOf(source);

  const runFetch = useCallback(function runFetch(quiet = false) {
    const requestId = ++requestIdRef.current;
    isFetchingRef.current = true;
    if (!quiet) {
      setIsLoading(true);
      setError(null);
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const request =
      source.kind === "current"
        ? getCurrentResponsePlan(source.fireEventId, controller.signal)
        : getResponsePlanById(source.planId, controller.signal);

    request
      .then((response) => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setPlan(response.plan);
        setPlanStatus(response.plan_status ?? null);
        setError(null);
        if (response.plan === null && response.plan_status === "generating") {
          pendingPollRef.current = setTimeout(() => runFetch(true), RESPONSE_PLAN_PENDING_POLL_INTERVAL_MS);
        }
      })
      .catch(() => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        if (quiet) {
          // A failed background re-check keeps showing "generating" and tries again.
          pendingPollRef.current = setTimeout(() => runFetch(true), RESPONSE_PLAN_PENDING_POLL_INTERVAL_MS);
          return;
        }
        setPlan(null);
        setPlanStatus(null);
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
    return () => {
      requestIdRef.current += 1;
      isFetchingRef.current = false;
      abortRef.current?.abort();
      if (pendingPollRef.current !== null) {
        clearTimeout(pendingPollRef.current);
        pendingPollRef.current = null;
      }
    };
  }, [runFetch]);

  const retry = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    if (pendingPollRef.current !== null) {
      clearTimeout(pendingPollRef.current);
      pendingPollRef.current = null;
    }
    runFetch();
  }, [runFetch]);

  return { plan, planStatus, isLoading, error, retry };
}
