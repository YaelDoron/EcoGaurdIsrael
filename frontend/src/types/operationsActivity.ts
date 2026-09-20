/**
 * Frontend mirror of the A5 Activity Detail contract
 * (backend/src/api/schemas/operations_activity.py) and its underlying
 * `OperationsActivityType` enum (backend/src/models/operations_activity.py).
 * `activity_type` values are the exact public strings the backend accepts
 * in the URL path and returns as the discriminator - never a table/model
 * name invented on the frontend.
 */
import type { FireDangerAssessmentStatus, FireDangerLevel } from "./fireDanger";
import type { FireEventStatus, FireSeverityAssessmentStatus, FireSeverityLevel } from "./fireEvent";

export type OperationsActivityType =
  | "fire_danger"
  | "satellite_hotspot"
  | "news_report"
  | "fire_event"
  | "fire_severity"
  | "global_planning_run"
  | "weather_conditions";

export type GlobalPlanningRunStatus = "running" | "completed" | "partial" | "failed" | "no_active_events";

export type GlobalPlanningRunEventStatus = "planned" | "no_op" | "insufficient_data" | "skipped_inactive" | "failed";

export interface OperationsActivityLocation {
  latitude: number;
  longitude: number;
}

// ---------------------------------------------------------------------------
// FIRE_DANGER detail - reuses A4's exact assessment-detail shape.
// ---------------------------------------------------------------------------

export interface FireDangerAreaCenter {
  latitude: number;
  longitude: number;
}

export interface FireDangerAssessmentWeatherInput {
  observation_id: number;
  station_external_id: number;
  station_name: string;
  observed_at: string;
}

export interface FireDangerAssessmentDetail {
  assessment_id: number;
  area_id: string;
  area_name: string;
  center: FireDangerAreaCenter;
  radius_km: number;
  status: FireDangerAssessmentStatus;
  score: number | null;
  level: FireDangerLevel | null;
  assessed_at: string;
  age_seconds: number;
  methodology: string;
  methodology_version: string;
  weather_inputs: FireDangerAssessmentWeatherInput[];
}

export interface FireDangerActivityDetail {
  activity_type: "fire_danger";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: FireDangerAssessmentDetail;
}

// ---------------------------------------------------------------------------
// SATELLITE_HOTSPOT detail
// ---------------------------------------------------------------------------

/** `confidence` is the raw FIRMS confidence string exactly as persisted - never reclassified. */
export interface SatelliteHotspotDetails {
  hotspot_id: number;
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

export interface SatelliteHotspotActivityDetail {
  activity_type: "satellite_hotspot";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: SatelliteHotspotDetails;
}

// ---------------------------------------------------------------------------
// NEWS_REPORT detail
// ---------------------------------------------------------------------------

export interface NewsReportDetails {
  report_id: number;
  source_url: string;
  source_feed: string;
  title: string;
  summary: string;
  location_name: string | null;
  latitude: number | null;
  longitude: number | null;
  published_at: string | null;
  fetched_at: string;
}

export interface NewsReportActivityDetail {
  activity_type: "news_report";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: NewsReportDetails;
}

// ---------------------------------------------------------------------------
// FIRE_EVENT detail - no area/place name; FireEvent persists none.
// ---------------------------------------------------------------------------

export interface FireEventEvidenceRefs {
  satellite_hotspot_ids: number[];
  news_report_ids: number[];
}

export interface FireEventSeverityReference {
  assessment_id: number;
  status: FireSeverityAssessmentStatus;
  score: number | null;
  level: FireSeverityLevel | null;
  assessed_at: string;
}

export interface FireEventActivityDetails {
  fire_event_id: number;
  status: FireEventStatus;
  detection_confidence: number;
  detected_at: string;
  updated_at: string;
  /** DB-insert ("Opened") timestamp - distinct from detected_at (source evidence time, may be earlier). */
  created_at: string;
  latitude: number;
  longitude: number;
  methodology: string;
  methodology_version: string;
  /** Trusted persisted location (or safe read-side fallback for historical events) - null when none exists. */
  location_name: string | null;
  evidence: FireEventEvidenceRefs;
  latest_severity: FireEventSeverityReference | null;
}

export interface FireEventActivityDetail {
  activity_type: "fire_event";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: FireEventActivityDetails;
}

// ---------------------------------------------------------------------------
// FIRE_SEVERITY detail
// ---------------------------------------------------------------------------

export interface FireSeverityDetails {
  assessment_id: number;
  fire_event_id: number;
  status: FireSeverityAssessmentStatus;
  score: number | null;
  level: FireSeverityLevel | null;
  assessed_at: string;
  methodology: string;
  methodology_version: string;
  vegetation_source: string | null;
  vegetation_dataset_year: number | null;
  vegetation_radius_km: number | null;
  vegetation_dominant_land_cover: string | null;
  vegetation_fuel_score: number | null;
  weather_observation_ids: number[];
  satellite_hotspot_ids: number[];
  selected_frp_hotspot_id: number | null;
}

export interface FireSeverityActivityDetail {
  activity_type: "fire_severity";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: FireSeverityDetails;
}

// ---------------------------------------------------------------------------
// GLOBAL_PLANNING_RUN detail - the global multi-fire optimization, never a
// single-event projection.
// ---------------------------------------------------------------------------

export interface GlobalPlanningRunMember {
  fire_event_id: number;
  event_order: number;
  result_status: GlobalPlanningRunEventStatus | null;
  response_plan_id: number | null;
  severity_level: FireSeverityLevel | null;
  assigned_resources: number | null;
  coverage_score: number | null;
  average_eta_seconds: number | null;
}

export interface GlobalPlanningRunDetails {
  global_planning_run_id: number;
  status: GlobalPlanningRunStatus;
  trigger: string;
  started_at: string;
  completed_at: string | null;
  methodology: string;
  methodology_version: string;
  fire_event_ids: number[];
  response_plan_ids: number[];
  coverage_score: number | null;
  average_eta_seconds: number | null;
  shortage_total_required: number | null;
  shortage_total_desired: number | null;
  shortage_total_assigned: number | null;
  shortage_unmet_required: number | null;
  shortage_unmet_desired: number | null;
  ga_population_size: number | null;
  ga_generation_count: number | null;
  ga_mutation_rate: number | null;
  ga_crossover_rate: number | null;
  members: GlobalPlanningRunMember[];
}

export interface GlobalPlanningRunActivityDetail {
  activity_type: "global_planning_run";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: GlobalPlanningRunDetails;
}

// ---------------------------------------------------------------------------
// WEATHER_CONDITIONS detail - the persisted weather inputs already traced
// for a HIGH+ Fire Danger assessment (entity_id IS the assessment id) -
// never a recalculated FFWI score, never a separate WeatherAlert entity.
// ---------------------------------------------------------------------------

export interface WeatherConditionsStationReading {
  station_id: number;
  station_name: string;
  observation_id: number;
  observed_at: string;
  temperature: number | null;
  relative_humidity: number | null;
  wind_speed: number | null;
  wind_gust: number | null;
}

export interface WeatherConditionsDetails {
  fire_danger_assessment_id: number;
  area_name: string;
  fire_danger_level: FireDangerLevel;
  assessed_at: string;
  readings: WeatherConditionsStationReading[];
}

export interface WeatherConditionsActivityDetail {
  activity_type: "weather_conditions";
  entity_id: number;
  occurred_at: string;
  title: string;
  location: OperationsActivityLocation | null;
  details: WeatherConditionsDetails;
}

// ---------------------------------------------------------------------------
// Discriminated union - switch on `activity_type`, matching the backend's
// `Field(discriminator="activity_type")` contract exactly.
// ---------------------------------------------------------------------------

export type OperationsActivityDetailResponse =
  | FireDangerActivityDetail
  | SatelliteHotspotActivityDetail
  | NewsReportActivityDetail
  | FireEventActivityDetail
  | FireSeverityActivityDetail
  | GlobalPlanningRunActivityDetail
  | WeatherConditionsActivityDetail;
