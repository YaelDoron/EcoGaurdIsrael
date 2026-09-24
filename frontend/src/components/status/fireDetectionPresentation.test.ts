import { describe, expect, it } from "vitest";
import { formatLikelihood, modelLabel, peakAiLikelihood, policyLabel } from "./fireDetectionPresentation";

describe("fireDetectionPresentation (Task 12)", () => {
  it("formats AI likelihood as a whole percentage with one rounding rule (half up)", () => {
    expect(formatLikelihood(0.87)).toBe("87%");
    expect(formatLikelihood(0.79)).toBe("79%");
    expect(formatLikelihood(0.445)).toBe("45%");
    expect(formatLikelihood(0.4449)).toBe("44%");
    expect(formatLikelihood(0)).toBe("0%");
    expect(formatLikelihood(1)).toBe("100%");
  });

  it("never turns a missing score into 0%", () => {
    expect(formatLikelihood(null)).toBe("Unavailable");
    expect(formatLikelihood(undefined)).toBe("Unavailable");
  });

  it("peak AI likelihood is only meaningful for events the AI path created", () => {
    expect(peakAiLikelihood("ECOGUARD_AI_HYBRID_DETECTION", 0.87)).toBe(0.87);
    expect(peakAiLikelihood("ECOGUARD_MULTI_SOURCE_DETECTION", 0.875)).toBeNull();
    expect(peakAiLikelihood(undefined, 0.5)).toBeNull();
  });

  it("labels the model and policy without exposing internal names", () => {
    expect(modelLabel("fire_detection_hgb_v5", "5.0")).toBe("HGB V5");
    expect(policyLabel("ai_hybrid_policy_v5.0")).toBe("AI Hybrid Policy v5.0");
    expect(modelLabel(null, null)).toBeNull();
  });
});
