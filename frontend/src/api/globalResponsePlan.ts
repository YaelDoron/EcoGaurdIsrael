import type { GlobalResponsePlanResponse } from "../types/globalResponsePlan";
import { apiGet } from "./client";

/**
 * Global Response Plan data access. A thin wrapper over the generic
 * `apiGet` client - no fetch call, base-URL handling, retry, or caching
 * logic lives here (see client.ts). Returns the backend response exactly as
 * received: no shortage/metrics aggregation, no route/target enrichment,
 * and no per-event grouping happens on this side - see
 * backend/src/services/global_planning/global_response_plan_read_service.py
 * for where that already happened.
 */
export function getCurrentGlobalResponsePlan(signal?: AbortSignal): Promise<GlobalResponsePlanResponse> {
  return apiGet<GlobalResponsePlanResponse>("/api/v1/global-response-plan/current", { signal });
}
