import type { FireSeverityLevel } from "../../types/fireEvent";
import { SEVERITY_PRESENTATION } from "./presentation";
import "./badges.css";

export interface SeverityBadgeProps {
  /** `null` means no severity assessment is available yet - never fabricated as "Low". */
  level: FireSeverityLevel | null;
}

const NOT_AVAILABLE_LABEL = "Not available";

/**
 * Presentation-only mapping of a known Severity level to a readable label +
 * visual tone (see ./presentation.ts). The level always comes from the
 * backend; this component never derives severity from a score or any other
 * calculation.
 */
export function SeverityBadge({ level }: SeverityBadgeProps) {
  if (level === null) {
    return <span className="badge badge--unavailable">{NOT_AVAILABLE_LABEL}</span>;
  }

  const presentation = SEVERITY_PRESENTATION[level];

  if (!presentation) {
    // Defensive fallback only - see StatusBadge's equivalent comment.
    return <span className="badge badge--unavailable">{level}</span>;
  }

  return <span className={`badge badge--${presentation.tone}`}>{presentation.label}</span>;
}
