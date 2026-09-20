import type { FireEventStatus, FireSeverityAssessmentStatus, FireSeverityLevel } from "./fireEvent";

/**
 * Frontend mirror of `GET /api/v1/fire-events/active`'s exact JSON shape
 * (backend/src/api/schemas/active_fire_events.py).
 *
 * `location_name` reflects the persisted FireEvent's own trustworthy
 * provenance when available (e.g. a simulation's canonical scenario
 * location), falling back to a safe read-side Fire Danger area-containment
 * association only for historical events predating that provenance - never
 * a frontend coordinate-to-name mapping, never geocoding.
 *
 * `detected_at` is the earliest correlated evidence's own (source)
 * observation time - it can be EARLIER than `created_at`, since evidence
 * may already exist before EcoGuard's detection/correlation step actually
 * runs and persists the FireEvent. `created_at` ("Opened") is when EcoGuard
 * itself opened/persisted this FireEvent - the operator-facing dashboard
 * time - and is never changed by later FireEvent updates.
 */
export interface ActiveFireEventSeverity {
  assessment_id: number;
  status: FireSeverityAssessmentStatus;
  score: number | null;
  level: FireSeverityLevel | null;
  assessed_at: string;
}

export interface ActiveFireEvent {
  fire_event_id: number;
  status: FireEventStatus;
  latitude: number;
  longitude: number;
  detection_confidence: number;
  detected_at: string;
  updated_at: string;
  created_at: string;
  severity: ActiveFireEventSeverity | null;
  location_name: string | null;
}

export interface ActiveFireEventsResponse {
  as_of: string;
  items: ActiveFireEvent[];
}
