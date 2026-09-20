import type { FireSeverityLevel } from "./fireEvent";
import type { ResponsePlanAction, ResponsePlanTarget } from "./responsePlan";

/**
 * Frontend mirror of the Global Response Plan API's exact JSON shape (Epic 6
 * UI/API Rework, Tasks B-BE-5/B-FE-5) -
 * backend/src/api/schemas/global_response_plan.py, populated by
 * `GlobalResponsePlanReadService`.
 *
 * `actions`/`uncovered_targets` on `GlobalEventPlan` reuse
 * `ResponsePlanAction`/`ResponsePlanTarget` from `./responsePlan` verbatim -
 * the backend itself reuses `ResponsePlanActionResponse`/
 * `ResponsePlanTargetResponse` (the same DTOs `ResponsePlanPresenter`
 * already produces for one child plan) rather than a parallel shape, so this
 * frontend layer mirrors that reuse instead of duplicating the type.
 *
 * These are pure data shapes - no field here is calculated on the frontend.
 * In particular, `GlobalPlanShortage`/`GlobalPlanMetrics` are already the
 * backend's own aggregation across every FireEvent in the generation;
 * nothing here re-sums, re-derives, or recalculates them.
 *
 * `GlobalPlanningRunStatus` values must match the backend's serialized enum
 * `.value` exactly - backend/src/models/global_planning_run_status.py.
 */
export type GlobalPlanningRunStatus = "running" | "completed" | "partial" | "failed" | "no_active_events";

export interface GlobalPlanMetrics {
  fitness_score: number | null;
  coverage_score: number | null;
  average_eta_seconds: number | null;
}

export interface GlobalPlanShortage {
  total_required: number;
  total_desired: number;
  total_assigned: number;
  unmet_required: number;
  unmet_desired: number;
}

export interface GlobalOptimizationConfig {
  random_seed: number;
  population_size: number;
  generation_count: number;
  mutation_rate: number;
  crossover_rate: number;
}

/**
 * One FireEvent's materialized child plan within the global generation.
 * Only present for a membership row that actually has a `response_plan_id`
 * - a NO_OP/FAILED/INSUFFICIENT_DATA/SKIPPED_INACTIVE member never appears
 * here. `severity_level`/`severity_score`/`minimum_resources`/
 * `desired_resources`/`assigned_resources`/`coverage_score`/
 * `average_eta_seconds` are `null` only when this member's cycle never
 * reached demand computation - never fabricated.
 */
export interface GlobalEventPlan {
  fire_event_id: number;
  response_plan_id: number;
  severity_level: FireSeverityLevel | null;
  severity_score: number | null;
  minimum_resources: number | null;
  desired_resources: number | null;
  assigned_resources: number | null;
  coverage_score: number | null;
  average_eta_seconds: number | null;
  actions: ResponsePlanAction[];
  uncovered_targets: ResponsePlanTarget[];
}

export interface GlobalPlanResponse {
  run_id: number;
  started_at: string;
  completed_at: string | null;
  status: GlobalPlanningRunStatus;
  metrics: GlobalPlanMetrics;
  shortage: GlobalPlanShortage;
  optimization_config: GlobalOptimizationConfig | null;
  events: GlobalEventPlan[];
}

/**
 * The response body for `GET /api/v1/global-response-plan/current`. `plan`
 * is `null` only when no GlobalPlanningRun has ever materialized a
 * ResponsePlan yet - never an error.
 */
export interface GlobalResponsePlanResponse {
  as_of: string;
  plan: GlobalPlanResponse | null;
}
