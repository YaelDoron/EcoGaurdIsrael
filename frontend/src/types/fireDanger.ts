/**
 * Frontend mirror of the Fire Danger area/assessment shape shared by
 * `GET /api/v1/fire-danger/areas/latest` (Task A4) and
 * `GET /api/v1/operations/overview`'s `fire_danger_areas` field (Task A6,
 * backend/src/api/schemas/fire_danger.py's `FireDangerAreaLatestResponse` -
 * A6 reuses that exact schema, so this frontend type is shared too).
 *
 * Fire Danger is pre-fire regional risk - a distinct domain from FireEvent
 * status or Fire Severity (see src/types/fireEvent.ts). `FireDangerLevel`
 * must never be conflated with `FireSeverityLevel`: they are different
 * backend enums with different value sets.
 */
export type FireDangerLevel = "low" | "moderate" | "high" | "very_high" | "extreme";

export type FireDangerAssessmentStatus = "valid" | "insufficient_data";

export interface FireDangerAreaCenter {
  latitude: number;
  longitude: number;
}

/**
 * `score`/`level` are `null` exactly when `status` is `"insufficient_data"`
 * - a real, persisted assessment attempt, never fabricated as `0`/`"low"`.
 */
export interface FireDangerAssessmentSummary {
  assessment_id: number;
  status: FireDangerAssessmentStatus;
  score: number | null;
  level: FireDangerLevel | null;
  assessed_at: string;
  /** Elapsed wall-clock seconds since `assessed_at`, computed server-side at response time - never recompute this locally. */
  age_seconds: number;
  methodology: string;
  methodology_version: string;
}

/**
 * `assessment` is `null` only when this area has never had a persisted
 * assessment at all - a materially different state from `assessment.status
 * === "insufficient_data"` (see FireDangerAssessmentSummary). Never treat
 * `null` as "low risk".
 */
export interface FireDangerArea {
  area_id: string;
  area_name: string;
  center: FireDangerAreaCenter;
  radius_km: number;
  assessment: FireDangerAssessmentSummary | null;
}
