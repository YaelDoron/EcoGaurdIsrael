/**
 * Minimal shared FireEvent/Severity domain vocabulary needed by shared
 * presentation components (StatusBadge, SeverityBadge) and the Active
 * Wildfires dashboard (Task 6). These are NOT the full Active Events API
 * response contract - see src/types/activeFireEvents.ts for that.
 *
 * Values must match the backend's serialized enum `.value` exactly:
 * - FireEventStatus             <-> backend/src/models/fire_event_status.py
 * - FireSeverityLevel           <-> backend/src/models/fire_severity_level.py
 * - FireSeverityAssessmentStatus <-> backend/src/models/fire_severity_assessment_status.py
 */
export type FireEventStatus = "suspected" | "confirmed" | "resolved" | "dismissed";

export type FireSeverityLevel = "low" | "moderate" | "high" | "critical";

export type FireSeverityAssessmentStatus = "valid" | "insufficient_data" | "inactive_event";
