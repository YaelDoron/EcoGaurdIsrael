import { useCallback, useRef, useState } from "react";
import { startSimulation } from "../api/simulation";
import { ApiError } from "../api/errors";
import type { SimulationRunStatus, StartSimulationRequest } from "../types/simulation";

const GENERIC_ERROR_MESSAGE = "Something went wrong while talking to the server.";

export interface UseStartSimulationResult {
  /** Sends the POST exactly once per call (a second call while one is in flight rejects immediately, never sending a duplicate request). Resolves on HTTP 202 - never waits for the simulation to finish. */
  start: (request: StartSimulationRequest) => Promise<SimulationRunStatus>;
  isStarting: boolean;
  /**
   * The raw `ApiError` (never collapsed to a generic string) so a caller
   * can branch on `error.status`/`error.code` - e.g. 409/
   * "SIMULATION_ALREADY_RUNNING" or 403/"SIMULATION_CONTROL_DISABLED"
   * (Task A7, Part 16). Presentation of that error is left to the caller.
   */
  error: ApiError | null;
  reset: () => void;
}

/**
 * A reusable mutation for `POST /api/v1/simulation/runs` (Task A3/A7). Owns
 * only request-in-flight/error state - it never polls for the simulation's
 * eventual outcome (that is `useOperationsOverview`'s/`getCurrentSimulationRun`'s
 * job) and never fabricates progress locally.
 *
 * After a successful start, the caller is expected to refresh/refetch the
 * Operations Overview itself (e.g. by calling `useOperationsOverview`'s own
 * `refresh()`) so the dashboard picks up the new `preparing`/`running`
 * state - this hook does not reach into another hook's state to do that
 * itself, keeping the two independently testable.
 */
export function useStartSimulation(): UseStartSimulationResult {
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const isStartingRef = useRef(false);

  const start = useCallback((request: StartSimulationRequest): Promise<SimulationRunStatus> => {
    if (isStartingRef.current) {
      return Promise.reject(new ApiError("A simulation start request is already in flight.", 0));
    }
    isStartingRef.current = true;
    setIsStarting(true);
    setError(null);

    return startSimulation(request)
      .then((status) => {
        setError(null);
        return status;
      })
      .catch((caught: unknown) => {
        const apiError = caught instanceof ApiError ? caught : new ApiError(GENERIC_ERROR_MESSAGE, 0);
        setError(apiError);
        throw apiError;
      })
      .finally(() => {
        isStartingRef.current = false;
        setIsStarting(false);
      });
  }, []);

  const reset = useCallback(() => {
    setError(null);
  }, []);

  return { start, isStarting, error, reset };
}
