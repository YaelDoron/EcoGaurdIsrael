import type { FireDangerLevel } from "../../types/fireDanger";

/**
 * Centralized Fire Danger level -> label + map color mapping (Task A8).
 * Mirrors presentation.ts's own precedent (STATUS_PRESENTATION/
 * SEVERITY_PRESENTATION) for FireEvent status/Severity - this is the ONLY
 * place that decides Fire Danger's label/color. Purely a presentation
 * table: the level always comes from the backend's persisted assessment;
 * nothing here recalculates FFWI or re-derives a level from a score.
 *
 * Fire Danger is a distinct domain from Severity (see types/fireDanger.ts) -
 * this intentionally does NOT reuse SEVERITY_PRESENTATION's tone table,
 * since the two enums have different value sets (5 Fire Danger levels vs 4
 * Severity levels) and conflating them would blur a real domain boundary.
 *
 * Colors form one sequential ramp (calm -> strongest), extending the
 * existing success/warning/danger tokens with two additional tiers - see
 * index.css's own comment on --color-fire-danger-very-high/-extreme.
 */
export interface FireDangerPresentationEntry {
  label: string;
  /** A CSS color (custom-property reference) for map fills/legend swatches. */
  color: string;
}

export const FIRE_DANGER_PRESENTATION: Record<FireDangerLevel, FireDangerPresentationEntry> = {
  low: { label: "Low", color: "var(--color-success)" },
  moderate: { label: "Moderate", color: "var(--color-warning)" },
  high: { label: "High", color: "var(--color-danger)" },
  very_high: { label: "Very high", color: "var(--color-fire-danger-very-high)" },
  extreme: { label: "Extreme", color: "var(--color-fire-danger-extreme)" },
};

/** Neutral color for an area with no persisted assessment yet - never implies "low risk". */
export const FIRE_DANGER_UNASSESSED_COLOR = "var(--color-text-muted)";
