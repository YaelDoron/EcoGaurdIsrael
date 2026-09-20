import { useCallback, useEffect, useRef, useState } from "react";
import { getOperationsActivityDetail } from "../api/operations";
import { ApiError } from "../api/errors";
import type { OperationsActivityDetailResponse, OperationsActivityType } from "../types/operationsActivity";

const LOAD_ERROR_MESSAGE = "Unable to load this activity's details. Please try again.";
const NOT_FOUND_STATUS = 404;

export interface UseOperationsActivityDetailOptions {
  /** `null` means "nothing selected" - no request is made. */
  activityType: OperationsActivityType | null;
  entityId: number | null;
  /** Set false to suppress fetching even when a selection is present (e.g. the drawer is closed). Defaults to true. */
  enabled?: boolean;
}

export interface UseOperationsActivityDetailResult {
  data: OperationsActivityDetailResponse | null;
  isLoading: boolean;
  /** A safe, static message - never the raw ApiError/backend detail. */
  error: string | null;
  /** True when the backend reported no activity exists for this type+id (a 404). */
  notFound: boolean;
  retry: () => void;
}

/**
 * Owns fetch/loading/error state for ONE selected Activity Feed item's full
 * detail (Task A5, the future drawer). This hook is meant to be used once
 * per "currently selected activity" - never once per rendered feed item;
 * the Activity Feed's own lightweight `preview` (Task A6) is what the list
 * renders without any network request. Requests only when both
 * `activityType` and `entityId` are non-null and `enabled` is true;
 * clearing the selection (either to `null`) cancels any in-flight request
 * and resets to the idle "nothing selected" state without fetching.
 *
 * Selecting a new activity supersedes an in-flight request for the
 * previous one (both via `AbortController` and a request-id guard, mirroring
 * useResponsePlan.ts's own stale-response precedent) - a slow response for
 * an item the user already navigated away from can never overwrite the
 * newer selection's state.
 */
export function useOperationsActivityDetail(
  options: UseOperationsActivityDetailOptions,
): UseOperationsActivityDetailResult {
  const { activityType, entityId, enabled = true } = options;

  const [data, setData] = useState<OperationsActivityDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);

  const isFetchingRef = useRef(false);
  const requestIdRef = useRef(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  const shouldFetch = enabled && activityType !== null && entityId !== null;

  const runFetch = useCallback(() => {
    if (!shouldFetch || activityType === null || entityId === null) {
      return;
    }
    const requestId = ++requestIdRef.current;
    isFetchingRef.current = true;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);
    setNotFound(false);

    getOperationsActivityDetail(activityType, entityId, controller.signal)
      .then((response) => {
        if (requestIdRef.current !== requestId) {
          return;
        }
        setData(response);
        setNotFound(false);
        setError(null);
      })
      .catch((caught: unknown) => {
        if (caught instanceof DOMException && caught.name === "AbortError") {
          return;
        }
        if (requestIdRef.current !== requestId) {
          return;
        }
        if (caught instanceof ApiError && caught.status === NOT_FOUND_STATUS) {
          setNotFound(true);
          setData(null);
          return;
        }
        setData(null);
        setError(LOAD_ERROR_MESSAGE);
      })
      .finally(() => {
        isFetchingRef.current = false;
        if (requestIdRef.current === requestId) {
          setIsLoading(false);
        }
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activityType, entityId, shouldFetch]);

  useEffect(() => {
    if (!shouldFetch) {
      requestIdRef.current += 1;
      abortControllerRef.current?.abort();
      setData(null);
      setIsLoading(false);
      setError(null);
      setNotFound(false);
      return;
    }

    runFetch();

    return () => {
      requestIdRef.current += 1;
      abortControllerRef.current?.abort();
    };
  }, [runFetch, shouldFetch]);

  const retry = useCallback(() => {
    if (isFetchingRef.current) {
      return;
    }
    runFetch();
  }, [runFetch]);

  return { data, isLoading, error, notFound, retry };
}
