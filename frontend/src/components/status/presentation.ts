import type { FireEventStatus, FireSeverityLevel } from "../../types/fireEvent";

/**
 * Centralized FireEvent status / Severity level -> label + visual tone
 * mapping. This is the ONLY place that decides labels/tone for these
 * values - StatusBadge/SeverityBadge just look values up here. Purely a
 * presentation table: no lifecycle rules, no score->level derivation, no
 * business logic of any kind.
 */
export type PresentationTone = "neutral" | "success" | "warning" | "danger" | "critical";

export interface PresentationEntry {
  label: string;
  tone: PresentationTone;
}

export const STATUS_PRESENTATION: Record<FireEventStatus, PresentationEntry> = {
  suspected: { label: "Suspected", tone: "warning" },
  confirmed: { label: "Confirmed", tone: "danger" },
  resolved: { label: "Resolved", tone: "success" },
  dismissed: { label: "Dismissed", tone: "neutral" },
};

export const SEVERITY_PRESENTATION: Record<FireSeverityLevel, PresentationEntry> = {
  low: { label: "Low", tone: "success" },
  moderate: { label: "Moderate", tone: "warning" },
  high: { label: "High", tone: "danger" },
  critical: { label: "Critical", tone: "critical" },
};
