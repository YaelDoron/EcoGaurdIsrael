import type { OperationsActivityDetailResponse, OperationsActivityType } from "../types/operationsActivity";
import type { OperationsOverviewResponse } from "../types/operationsOverview";
import { apiGet } from "./client";

const OPERATIONS_OVERVIEW_PATH = "/api/v1/operations/overview";

export interface GetOperationsOverviewParams {
  /** Forwarded as `?activity_limit=`; omitted entirely (not defaulted here) when unset, letting the backend apply its own default (Task A6). */
  activityLimit?: number;
  signal?: AbortSignal;
}

/**
 * Fetch the Operations Overview dashboard snapshot (Task A6). Thin wrapper
 * over the generic `apiGet` client - no fetch call, base-URL handling,
 * retry, polling, or caching logic lives here (see hooks/useOperationsOverview.ts
 * for that). This is the ONLY endpoint the main dashboard should call for
 * its operational state - do not separately poll
 * /fire-danger/areas/latest, /fire-events/active, or /simulation/runs/current.
 */
export function getOperationsOverview(params: GetOperationsOverviewParams = {}): Promise<OperationsOverviewResponse> {
  const { activityLimit, signal } = params;
  const query = activityLimit !== undefined ? `?activity_limit=${encodeURIComponent(String(activityLimit))}` : "";
  return apiGet<OperationsOverviewResponse>(`${OPERATIONS_OVERVIEW_PATH}${query}`, { signal });
}

/**
 * Fetch one Activity Feed item's full detail (Task A5) - the drawer/click
 * detail, never prefetched for every feed item. A missing entity (404)
 * surfaces as a rejected promise carrying an `ApiError` with
 * `status === 404`; an unsupported `activityType` is rejected by the
 * backend's own path-enum validation (422) before it can be misread as a
 * different resource.
 */
export function getOperationsActivityDetail(
  activityType: OperationsActivityType,
  entityId: number,
  signal?: AbortSignal,
): Promise<OperationsActivityDetailResponse> {
  return apiGet<OperationsActivityDetailResponse>(`/api/v1/operations/activity/${activityType}/${entityId}`, {
    signal,
  });
}
