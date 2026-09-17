import type { ResponseTargetType, RouteStatus } from "../../types/responsePlan";

/**
 * Centralized `target_type`/route `status` -> human-readable label mapping
 * for the Response Plan UI (Epic 6, US 6.3). Purely a presentation table,
 * matching the convention in `src/components/status/presentation.ts`: no
 * value here is derived, recalculated, or reinterpreted - callers only
 * look up the exact backend-serialized value they already have.
 */
export const TARGET_TYPE_LABELS: Record<ResponseTargetType, string> = {
  active_fire: "Active fire",
  predicted_risk: "Predicted risk",
};

export const ROUTE_STATUS_LABELS: Record<RouteStatus, string> = {
  reachable: "Reachable",
  unreachable: "Unreachable",
  unmappable: "Unmappable",
};
