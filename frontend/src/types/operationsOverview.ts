/**
 * Frontend mirror of `GET /api/v1/operations/overview`'s exact JSON shape
 * (backend/src/api/schemas/operations_overview.py, Task A6).
 *
 * Reuses existing contract types wherever the backend reuses its own
 * schema, rather than duplicating a parallel shape:
 * - `fire_danger_areas` items -> `FireDangerArea` (same as A4's
 *   `GET /api/v1/fire-danger/areas/latest`; A6 wraps and reuses that exact
 *   schema server-side).
 * - `active_fires` items -> `ActiveFireEvent` (same as US 6.1's
 *   `GET /api/v1/fire-events/active`; A6 reuses that exact schema
 *   server-side too). `ActiveFireEvent.location_name` is optional,
 *   read/presentation-only metadata resolved server-side the same way as
 *   `SatelliteHotspotActivityPreview.location_name` - never invented here.
 * - `simulation.run` -> `SimulationRunStatus` (same as A3's
 *   `GET /api/v1/simulation/runs/current`).
 *
 * Activity Feed items are a lightweight PREVIEW only (Task A6, Part 11) -
 * never the full A5 detail payload. Each item is discriminated by its own
 * `activity_type` (not by `preview`'s shape alone, since the backend does
 * not tag `preview` itself) - narrowing on `item.activity_type` narrows
 * `item.preview` to the matching variant.
 */
import type { ActiveFireEvent } from "./activeFireEvents";
import type { FireDangerArea, FireDangerAssessmentStatus, FireDangerLevel } from "./fireDanger";
import type { FireEventStatus, FireSeverityLevel } from "./fireEvent";
import type { GlobalPlanningRunStatus, OperationsActivityLocation, OperationsActivityType } from "./operationsActivity";
import type { SimulationRunStatus } from "./simulation";

export interface OperationsSimulationSummary {
  enabled: boolean;
  /** `null` when `enabled` is false, or when enabled but no run has ever started - never an error state. */
  run: SimulationRunStatus | null;
}

// ---------------------------------------------------------------------------
// Activity Feed item previews (small, typed, per-type projections)
// ---------------------------------------------------------------------------

export interface FireDangerActivityPreview {
  area_name: string;
  status: FireDangerAssessmentStatus;
  level: FireDangerLevel | null;
  score: number | null;
}

export interface SatelliteHotspotActivityPreview {
  confidence: string | null;
  frp: number | null;
  /** The real persisted Fire Danger area name whose circle contains this hotspot's coordinates, or `null` when no known area does - never a frontend-derived/invented location. */
  location_name: string | null;
}

export interface NewsReportActivityPreview {
  source: string;
  headline: string;
}

export interface FireEventActivityPreview {
  status: FireEventStatus;
  confidence: number;
}

export interface FireSeverityActivityPreview {
  fire_event_id: number;
  level: FireSeverityLevel | null;
  score: number | null;
}

export interface GlobalPlanningRunActivityPreview {
  status: GlobalPlanningRunStatus;
  fire_event_count: number;
}

export interface WeatherConditionsActivityPreview {
  area_name: string;
  fire_danger_level: FireDangerLevel;
  fire_danger_assessment_id: number;
  temperature_c: number;
  relative_humidity_pct: number;
  wind_speed_kmh: number;
  /** `null` when no contributing observation persisted a gust value. */
  wind_gust_kmh: number | null;
}

interface OperationsActivityFeedItemBase {
  /** Stable composite "{activity_type}:{entity_id}" identity - entity_id alone collides across the six source tables. */
  activity_id: string;
  entity_id: number;
  /** Source/domain event time (observation, publication, assessment, detection) - kept for A5 detail/history purposes. Never render this as the LIVE feed timestamp - see available_at. */
  occurred_at: string;
  /** When EcoGuard actually persisted/could expose this activity - the LIVE feed's display/sort timestamp (Task 4). Never derived on the frontend (no Date.now()); always the server's own value. */
  available_at: string;
  title: string;
  location: OperationsActivityLocation | null;
}

export interface FireDangerActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "fire_danger";
  preview: FireDangerActivityPreview;
}

export interface SatelliteHotspotActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "satellite_hotspot";
  preview: SatelliteHotspotActivityPreview;
}

export interface NewsReportActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "news_report";
  preview: NewsReportActivityPreview;
}

export interface FireEventActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "fire_event";
  preview: FireEventActivityPreview;
}

export interface FireSeverityActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "fire_severity";
  preview: FireSeverityActivityPreview;
}

export interface GlobalPlanningRunActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "global_planning_run";
  preview: GlobalPlanningRunActivityPreview;
}

export interface WeatherConditionsActivityFeedItem extends OperationsActivityFeedItemBase {
  activity_type: "weather_conditions";
  preview: WeatherConditionsActivityPreview;
}

/** Discriminated on `activity_type` - matches `OperationsActivityType` exactly (Task A5). */
export type OperationsActivityFeedItem =
  | FireDangerActivityFeedItem
  | SatelliteHotspotActivityFeedItem
  | NewsReportActivityFeedItem
  | FireEventActivityFeedItem
  | FireSeverityActivityFeedItem
  | GlobalPlanningRunActivityFeedItem
  | WeatherConditionsActivityFeedItem;

export interface OperationsActivityFeed {
  /** Server order is already the final chronological order (available_at DESC) - never resort on the frontend. */
  items: OperationsActivityFeedItem[];
  limit: number;
}

export interface OperationsOverviewResponse {
  /** The server's snapshot-assembly timestamp - never replace with a frontend Date.now(). */
  generated_at: string;
  simulation: OperationsSimulationSummary;
  fire_danger_areas: FireDangerArea[];
  /** Already in the backend's canonical order - never re-sort on the frontend. */
  active_fires: ActiveFireEvent[];
  activity_feed: OperationsActivityFeed;
}

// Re-exported so callers of this module never need to reach into
// operationsActivity.ts merely to name the discriminator's type.
export type { OperationsActivityType };
