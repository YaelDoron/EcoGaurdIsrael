import type { PresentationTone } from "../status/presentation";

/**
 * Map-marker colors, expressed as CSS custom-property references so they
 * stay derived from the single design-token source (src/index.css) and the
 * same status/severity tone vocabulary StatusBadge/SeverityBadge already
 * use (src/components/status/presentation.ts) - this file adds no new
 * color decisions of its own, only a tone->map-color lookup.
 */
export const TONE_MAP_COLOR: Record<PresentationTone, string> = {
  neutral: "var(--color-text-muted)",
  success: "var(--color-success)",
  warning: "var(--color-warning)",
  danger: "var(--color-danger)",
  critical: "var(--color-danger)",
};

// Default (unallocated) fire stations are green.
export const STATION_MARKER_COLOR = "var(--color-success)";
// A station actively contributing trucks to the current response plan: deep
// blue "action" color. Deliberately NOT red - red is reserved for active fires.
export const STATION_ALLOCATED_COLOR = "#1e3a8a";

// Wildfire flame colors: vibrant orange-red so an active fire stands out.
export const FIRE_ICON_COLOR_ACTIVE = "#ff4500";
export const FIRE_ICON_COLOR_CONFIRMED = "#ff2200";
export const FIRE_ICON_COLOR_INACTIVE = "#9ca3af";

const SPREAD_RISK_LOW_RGB = { r: 253, g: 230, b: 138 }; // matches badge warning background (#fde68a)
const SPREAD_RISK_HIGH_RGB = { r: 185, g: 28, b: 28 }; // matches --color-danger (#b91c1c)

/**
 * Interpolates an already-computed `spread_risk_score` (0-100) into a
 * display color - a pure visual mapping of a persisted number, never a
 * recalculation of risk itself.
 */
export function spreadRiskColor(score: number): string {
  const ratio = Math.max(0, Math.min(100, score)) / 100;
  const r = Math.round(SPREAD_RISK_LOW_RGB.r + (SPREAD_RISK_HIGH_RGB.r - SPREAD_RISK_LOW_RGB.r) * ratio);
  const g = Math.round(SPREAD_RISK_LOW_RGB.g + (SPREAD_RISK_HIGH_RGB.g - SPREAD_RISK_LOW_RGB.g) * ratio);
  const b = Math.round(SPREAD_RISK_LOW_RGB.b + (SPREAD_RISK_HIGH_RGB.b - SPREAD_RISK_LOW_RGB.b) * ratio);
  return `rgb(${r}, ${g}, ${b})`;
}
