import { describe, expect, it } from "vitest";
import { spreadRiskColor, TONE_MAP_COLOR } from "./colors";

describe("TONE_MAP_COLOR", () => {
  it("provides a color for every presentation tone", () => {
    expect(TONE_MAP_COLOR.neutral).toBe("var(--color-text-muted)");
    expect(TONE_MAP_COLOR.success).toBe("var(--color-success)");
    expect(TONE_MAP_COLOR.warning).toBe("var(--color-warning)");
    expect(TONE_MAP_COLOR.danger).toBe("var(--color-danger)");
    expect(TONE_MAP_COLOR.critical).toBe("var(--color-danger)");
  });
});

describe("spreadRiskColor", () => {
  it("returns the low-risk color at score 0", () => {
    expect(spreadRiskColor(0)).toBe("rgb(253, 230, 138)");
  });

  it("returns the high-risk color at score 100", () => {
    expect(spreadRiskColor(100)).toBe("rgb(185, 28, 28)");
  });

  it("interpolates for a mid-range score", () => {
    const color = spreadRiskColor(50);
    expect(color).toMatch(/^rgb\(\d+, \d+, \d+\)$/);
    expect(color).not.toBe(spreadRiskColor(0));
    expect(color).not.toBe(spreadRiskColor(100));
  });

  it("clamps out-of-range scores instead of producing an invalid color", () => {
    expect(spreadRiskColor(-10)).toBe(spreadRiskColor(0));
    expect(spreadRiskColor(150)).toBe(spreadRiskColor(100));
  });
});
