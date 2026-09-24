import type { FireEventStatus } from "../../types/fireEvent";
import type { FireEventMLAssessment } from "../../types/eventDetails";

/**
 * Operator-facing wording for the FINAL Fire Detection semantics (Task 9A/9B/9C). Presentation only - no business rule lives here.
 *
 *   SUSPECTED -> monitored, collecting evidence, NOT eligible for response planning (no plan, no resources)
 *   CONFIRMED -> response eligible (a response plan may be generated)
 *
 * The AI value is a model-estimated likelihood learned from synthetic data - never call it "certainty".
 */
export const AI_HYBRID_V5_MODE = "ai_hybrid_v5";

export const SUSPECTED_MONITORING_LABEL = "Monitoring - awaiting additional evidence";

export const SUSPECTED_HELP_TEXT =
  "SUSPECTED means the system detected evidence consistent with a possible wildfire but does not yet have enough corroborating evidence to confirm it.";

export const SUSPECTED_LEGEND_TITLE = "Monitoring - awaiting additional evidence; no response plan";

export const CONFIRMED_AI_HELP_TEXT =
  "CONFIRMED means a high AI likelihood together with corroborating current satellite evidence. The likelihood is a model estimate, not certainty.";

export const CONFIRMED_GENERIC_HELP_TEXT = "CONFIRMED events are eligible for response planning.";

export const AI_LIKELIHOOD_HELP_TEXT =
  "Model-estimated AI likelihood. The model was trained on synthetic data, so this is not real-world certainty.";

export function isAiHybridMode(mode: string | null | undefined): boolean {
  return mode === AI_HYBRID_V5_MODE;
}

/** Short explanation of the event's status, or null when there is nothing useful to add (resolved / dismissed). */
export function statusHelpText(status: FireEventStatus, mode: string | null | undefined): string | null {
  if (status === "suspected") return SUSPECTED_HELP_TEXT;
  if (status === "confirmed") return isAiHybridMode(mode) ? CONFIRMED_AI_HELP_TEXT : CONFIRMED_GENERIC_HELP_TEXT;
  return null;
}

const DETECTION_MODE_LABEL: Record<string, string> = {
  rule_only: "Rule-based",
  shadow: "Rule-based (AI shadow)",
  hybrid: "Rule + AI hybrid",
  [AI_HYBRID_V5_MODE]: "AI Hybrid V5",
};

export function detectionModeLabel(mode: string): string {
  return DETECTION_MODE_LABEL[mode] ?? mode;
}

const POLICY_STATUS_LABEL: Record<string, string> = {
  no_event: "No event",
  suspected: "Suspected",
  confirmed: "Confirmed",
};

export function policyStatusLabel(policyStatus: string): string {
  return POLICY_STATUS_LABEL[policyStatus] ?? policyStatus;
}

/**
 * Operator-facing AI likelihood: a whole percentage (0.87 -> "87%", 0.445 -> "45%"). One rounding rule for the whole UI
 * (round half up, tiny epsilon so 0.445 is not lost to binary floating point). The API value itself is never changed.
 */
export function formatLikelihood(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Unavailable";
  return `${Math.round((value + Number.EPSILON) * 100)}%`;
}

/**
 * The AI Hybrid detection mode is driven by the event's PERSISTED decision mode (`ml_assessment.mode` / `ml_summary.mode`), never by
 * the current global configuration and never by the mere existence of an AI score (shadow events carry one too).
 */
export const AI_HYBRID_FIRE_EVENT_METHODOLOGY = "ECOGUARD_AI_HYBRID_DETECTION";

/**
 * `FireEvent.detection_confidence` is the PEAK AI likelihood only for an event that the AI path itself created. An event that
 * was created by a legacy (rule) decision keeps a rule-based confidence, so its value must not be labelled "peak AI likelihood".
 */
export function peakAiLikelihood(methodology: string | null | undefined, detectionConfidence: number): number | null {
  return methodology === AI_HYBRID_FIRE_EVENT_METHODOLOGY ? detectionConfidence : null;
}

export function modelLabel(modelName: string | null, modelVersion: string | null): string | null {
  if (!modelName) return null;
  const major = modelVersion ? modelVersion.split(".")[0] : null;
  return modelName.includes("hgb") ? `HGB${major ? ` V${major}` : ""}` : modelName;
}

export function policyLabel(policyVersion: string | null): string | null {
  if (!policyVersion) return null;
  const match = /v(\d+(?:\.\d+)?)$/.exec(policyVersion);
  return match ? `AI Hybrid Policy v${match[1]}` : policyVersion;
}

/** The AI audit facts to show (only when the assessment came from the AI Hybrid mode and carries them). */
export interface AiAssessmentFacts {
  detectionMode: string;
  latestLikelihood: string;
  modelLabel: string | null;
  policyLabel: string | null;
  policyVerdict: string | null;
  satellitePassCount: number | null;
  currentSatellitePixelCount: number | null;
  historyUsed: boolean | null;
}

export function aiAssessmentFacts(assessment: FireEventMLAssessment | null): AiAssessmentFacts | null {
  if (!assessment || !isAiHybridMode(assessment.mode)) return null;
  return {
    detectionMode: detectionModeLabel(assessment.mode),
    latestLikelihood: assessment.available ? formatLikelihood(assessment.model_score) : "Unavailable",
    modelLabel: modelLabel(assessment.model_name ?? null, assessment.model_version ?? null),
    policyLabel: policyLabel(assessment.policy_version ?? null),
    policyVerdict: assessment.policy_status ? policyStatusLabel(assessment.policy_status) : null,
    satellitePassCount: assessment.satellite_pass_count ?? null,
    currentSatellitePixelCount: assessment.current_satellite_pixel_count ?? null,
    historyUsed: assessment.history_available ?? null,
  };
}
