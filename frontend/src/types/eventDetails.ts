import type { FireEventStatus, FireSeverityAssessmentStatus, FireSeverityLevel } from "./fireEvent";

/**
 * Frontend mirror of `GET /api/v1/fire-events/{fire_event_id}/details`'s
 * exact JSON shape (backend/src/api/schemas/event_details.py). Field names
 * and nullability match the backend response one-to-one - this layer never
 * adds, renames, or infers a field the backend does not send.
 */
export type FireDangerAssessmentStatus = "valid" | "insufficient_data";

export type FireDangerLevel = "low" | "moderate" | "high" | "very_high" | "extreme";

export type FireSpreadPredictionStatus = "valid" | "insufficient_data" | "inactive_event";

export type ResourceStatus = "available" | "assigned" | "unavailable";

export type ResponseTargetType = "active_fire" | "predicted_risk";

export interface FireEventSummary {
  fire_event_id: number;
  status: FireEventStatus;
  latitude: number;
  longitude: number;
  detection_confidence: number;
  detected_at: string;
  updated_at: string;
  methodology: string;
  methodology_version: string;
}

export interface SeverityAssessment {
  assessment_id: number;
  status: FireSeverityAssessmentStatus;
  score: number | null;
  level: FireSeverityLevel | null;
  assessed_at: string;
}

/**
 * Shaped for a future FireEvent<->area link; always `null` on
 * `EventDetailsResult.danger` today - see the backend schema's docstring.
 */
export interface DangerAssessment {
  assessment_id: number;
  status: FireDangerAssessmentStatus;
  score: number | null;
  level: FireDangerLevel | null;
  assessed_at: string;
}

export interface SpreadPredictionCell {
  latitude: number;
  longitude: number;
  spread_probability: number;
  spread_risk_score: number;
  reached_step: number;
  reached_minutes: number;
}

/** `cells` is `[]` whenever `status` is not `"valid"` - never a fabricated fallback. */
export interface SpreadPrediction {
  horizon_minutes: number;
  status: FireSpreadPredictionStatus;
  predicted_at: string;
  cells: SpreadPredictionCell[];
}

export interface ResponseTarget {
  target_order: number;
  target_type: ResponseTargetType;
  latitude: number;
  longitude: number;
  priority_score: number;
  prediction_horizon_minutes: number | null;
}

export interface FireStation {
  station_id: string;
  name: string;
  latitude: number;
  longitude: number;
  station_type: string | null;
  address: string | null;
}

export interface FirefightingResource {
  resource_id: string;
  station_id: string;
  status: ResourceStatus;
}

export interface SatelliteEvidence {
  id: number;
  detected_at: string;
  latitude: number;
  longitude: number;
  confidence: string | null;
  frp: number | null;
  brightness: number | null;
  satellite: string | null;
  instrument: string | null;
  day_night: string | null;
}

export interface NewsEvidence {
  id: number;
  title: string;
  summary: string;
  source: string;
  observed_at: string;
  location_name: string | null;
  latitude: number | null;
  longitude: number | null;
}

/** The direct evidence (satellite + news) that supports the FireEvent's detection. */
export interface DetectionEvidence {
  satellite: SatelliteEvidence[];
  news: NewsEvidence[];
}

/** One resource from a station allocated to the event's current response plan. */
export interface StationAllocation {
  resource_id: string;
  fire_event_id: number;
  response_plan_id: number;
}

/** Per-station resource counts and current-plan allocations, for the station popup. */
export interface StationSummary {
  station_id: string;
  total_resources: number;
  available: number;
  assigned_status: number;
  unavailable: number;
  current_global_plan_allocations: StationAllocation[];
}

export interface ResponseAction {
  resource_id: string;
  station_id: string;
  response_target_id: number;
  target_type: string;
  target_priority: number;
  eta_seconds: number | null;
  route_distance_meters: number | null;
  node_path: number[] | null;
}

export interface BaselineComparison {
  baseline_score: number;
  baseline_coverage_score: number;
  baseline_average_eta_seconds: number | null;
  score_difference: number;
  improvement_percentage: number | null;
}

export interface CurrentResponsePlan {
  plan_id: number;
  generated_at: string;
  methodology: string;
  methodology_version: string;
  plan_score: number;
  coverage_score: number;
  average_eta_seconds: number | null;
  actions: ResponseAction[];
  uncovered_target_ids: number[];
  baseline_comparison: BaselineComparison | null;
}

export interface EventDetailsResult {
  as_of: string;
  fire_event: FireEventSummary;
  severity: SeverityAssessment | null;
  danger: DangerAssessment | null;
  detection_evidence: DetectionEvidence;
  spread_predictions: SpreadPrediction[];
  targets: ResponseTarget[];
  stations: FireStation[];
  resources: FirefightingResource[];
  station_summaries: StationSummary[];
  current_response_plan: CurrentResponsePlan | null;
}
