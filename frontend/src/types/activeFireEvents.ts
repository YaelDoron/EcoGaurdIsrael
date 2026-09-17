import type { FireEventStatus, FireSeverityAssessmentStatus, FireSeverityLevel } from "./fireEvent";

/**
 * Frontend mirror of `GET /api/v1/fire-events/active`'s exact JSON shape
 * (backend/src/api/schemas/active_fire_events.py). No `area_name` field -
 * the backend does not send one (see that module's docstring); this layer
 * does not fabricate one.
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
  severity: ActiveFireEventSeverity | null;
}

export interface ActiveFireEventsResponse {
  as_of: string;
  items: ActiveFireEvent[];
}
