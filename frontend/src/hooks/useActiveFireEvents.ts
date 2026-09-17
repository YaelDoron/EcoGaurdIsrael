import { useCallback, useEffect, useRef, useState } from "react";
import { getActiveFireEvents } from "../api/activeFireEvents";
import type { ActiveFireEventsResponse } from "../types/activeFireEvents";

const LOAD_ERROR_MESSAGE = "Unable to load active wildfire events. Please try again.";
const REFRESH_ERROR_MESSAGE = "Unable to refresh active wildfire events. Showing previously loaded data.";

export interface UseActiveFireEventsResult {
  data: ActiveFireEventsResponse | null;
  /** True only for the very first fetch, before any data has ever loaded. */
  isLoading: boolean;
  /** True while a refresh is in flight; previous `data` remains available/unchanged. */
  isRefreshing: boolean;
  /** Set only when there is no data to show at all (initial load failed). */
  loadError: string | null;
  /** Set when a refresh fails but previously loaded `data` is still valid and shown. */
  refreshError: string | null;
  refresh: () => void;
}

/**
 * Owns fetch/loading/error/refresh state for the Active Wildfires
 * dashboard. Performs no wildfire business logic itself (no counting, no
 * severity derivation) - it only fetches and stores exactly what
 * getActiveFireEvents() returns.
 *
 * A refresh failure never discards already-visible valid data: `data`
 * only ever changes on a successful fetch, so a failed refresh surfaces
 * through `refreshError` while the previous `data` stays exactly as it
 * was (see Task 12).
 */
export function useActiveFireEvents(): UseActiveFireEventsResult {
  const [data, setData] = useState<ActiveFireEventsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

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
    }

    getActiveFireEvents()
      .then((response) => {
        setData(response);
        hasDataRef.current = true;
        setLoadError(null);
        setRefreshError(null);
      })
      .catch(() => {
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
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return { data, isLoading, isRefreshing, loadError, refreshError, refresh: load };
}
