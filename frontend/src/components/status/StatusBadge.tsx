import type { FireEventStatus } from "../../types/fireEvent";
import { STATUS_PRESENTATION } from "./presentation";
import "./badges.css";

export interface StatusBadgeProps {
  status: FireEventStatus;
}

/**
 * Presentation-only mapping of a known FireEvent status to a readable
 * label + visual tone (see ./presentation.ts). Never derives or changes
 * status - it only displays the value it is given.
 */
export function StatusBadge({ status }: StatusBadgeProps) {
  const presentation = STATUS_PRESENTATION[status];

  if (!presentation) {
    // Defensive fallback only - TypeScript's FireEventStatus union already
    // prevents this at compile time; this guards a value from outside the
    // type system (e.g. an unvalidated runtime response).
    return <span className="badge badge--neutral">{status}</span>;
  }

  return <span className={`badge badge--${presentation.tone}`}>{presentation.label}</span>;
}
