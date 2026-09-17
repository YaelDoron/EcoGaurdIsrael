import type { ResponsePlanEnvelopeResponse } from "../types/responsePlan";
import { apiGet } from "./client";

/**
 * Response Plan data access (Epic 6, US 6.3). Thin wrappers over the
 * generic `apiGet` client - no fetch call, base-URL handling, retry, or
 * caching logic lives here (see client.ts). Both functions return the
 * backend envelope exactly as received: no metric/ETA/distance/priority
 * calculation, no CURRENT-vs-SUPERSEDED or COMPLETE/PARTIAL determination,
 * no coordinate derivation, and no filtering of uncovered targets happens
 * on this side - see backend/src/api/response_plan_presenter.py for where
 * that enrichment already happened.
 */

export function getCurrentResponsePlan(fireEventId: number): Promise<ResponsePlanEnvelopeResponse> {
  return apiGet<ResponsePlanEnvelopeResponse>(`/api/v1/fire-events/${fireEventId}/response-plan`);
}

export function getResponsePlanById(planId: number): Promise<ResponsePlanEnvelopeResponse> {
  return apiGet<ResponsePlanEnvelopeResponse>(`/api/v1/response-plans/${planId}`);
}
