/**
 * Frontend mirror of the Response Plan API's exact JSON shape (Epic 6,
 * US 6.3) - backend/src/api/schemas/response_plans.py
 * (`ResponsePlanDetailResponse`, `ResponsePlanEnvelopeResponse`, etc.),
 * populated by `ResponsePlanPresenter` (backend/src/api/response_plan_presenter.py).
 *
 * These are pure data shapes - no field here is calculated on the
 * frontend. In particular:
 * - `status` (COMPLETE/PARTIAL/NO_FEASIBLE_ASSIGNMENTS) and `is_current`
 *   (CURRENT/SUPERSEDED) are two independent, backend-resolved concepts -
 *   never derived from action counts or freshness on this side.
 * - `priority_score`, `eta_seconds`, `distance_meters`, and the baseline
 *   comparison fields are always the exact persisted values; nothing here
 *   recalculates them.
 * - `path_coordinates` is only ever what the backend already resolved from
 *   `node_path` - never derived from `resource.origin`/`target` coordinates
 *   on the frontend.
 *
 * String union values below must match the backend's serialized enum
 * `.value` exactly:
 * - ResponsePlanStatus <-> backend/src/models/response_plan_status.py
 * - RouteStatus        <-> backend/src/models/routing.py (`RouteStatus`)
 * - ResponseTargetType <-> backend/src/models/response_target_type.py
 */
export type ResponsePlanStatus = "complete" | "partial" | "no_feasible_assignments";

export type RouteStatus = "reachable" | "unreachable" | "unmappable";

export type ResponseTargetType = "active_fire" | "predicted_risk";

export interface Coordinate {
  latitude: number;
  longitude: number;
}

/**
 * `station_name`/`origin` are `null` only when the action's `station_id`
 * no longer resolves against persisted FireStation data - the backend
 * never fabricates a placeholder value for them.
 */
export interface ResponsePlanResource {
  resource_id: string;
  station_id: string;
  station_name: string | null;
  origin: Coordinate | null;
}

/**
 * `target_type`/`priority_score`/`latitude`/`longitude` are `null` only
 * when `response_target_id` no longer resolves against the plan's exact
 * persisted ResponseTargetSet snapshot - never fabricated or recalculated
 * on the frontend.
 */
export interface ResponsePlanTarget {
  response_target_id: number;
  target_type: ResponseTargetType | null;
  priority_score: number | null;
  latitude: number | null;
  longitude: number | null;
}

/**
 * `status` is `null` only when no persisted RouteResult could be found for
 * this action at all. `path_coordinates` is populated only when every node
 * in `node_path` resolved to a persisted GraphNode on the backend -
 * otherwise it is `null`, never a partial or fabricated line.
 */
export interface ResponsePlanRoute {
  status: RouteStatus | null;
  eta_seconds: number | null;
  distance_meters: number | null;
  node_path: number[] | null;
  path_coordinates: Coordinate[] | null;
}

export interface ResponsePlanAction {
  resource: ResponsePlanResource;
  target: ResponsePlanTarget;
  route: ResponsePlanRoute;
}

export interface ResponsePlanMetrics {
  plan_score: number;
  coverage_score: number;
  average_eta_seconds: number | null;
}

export interface BaselineComparison {
  baseline_score: number;
  baseline_coverage_score: number;
  baseline_average_eta_seconds: number | null;
  score_difference: number;
  improvement_percentage: number | null;
}

export interface OptimizationConfig {
  population_size: number;
  generation_count: number;
  mutation_rate: number;
  crossover_rate: number;
  eta_reference_seconds: number;
  initial_assignment_probability: number;
  tournament_size: number;
  elitism_count: number;
}

export interface ResponsePlan {
  plan_id: number;
  fire_event_id: number;
  response_target_set_id: number;
  route_planning_run_id: number;
  generated_at: string;
  methodology: string;
  methodology_version: string;
  random_seed: number;
  status: ResponsePlanStatus;
  is_current: boolean;
  metrics: ResponsePlanMetrics;
  actions: ResponsePlanAction[];
  uncovered_targets: ResponsePlanTarget[];
  baseline_comparison: BaselineComparison | null;
  optimization_config: OptimizationConfig | null;
  no_resources_during_planning: boolean;
}

/** The shared Epic 6 envelope: `{ plan: ... }`, or `{ plan: null }` when none exists. */
export interface ResponsePlanEnvelopeResponse {
  plan: ResponsePlan | null;
  /** Current-plan endpoint only: `generating` = CONFIRMED event whose plan is still being produced. */
  plan_status?: "available" | "generating" | "not_applicable" | null;
}
