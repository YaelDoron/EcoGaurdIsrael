import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { BaselineComparison } from "../../types/responsePlan";
import { BaselineComparisonSection } from "./BaselineComparison";

function makeComparison(overrides: Partial<BaselineComparison> = {}): BaselineComparison {
  return {
    baseline_score: 50.0,
    baseline_coverage_score: 40.0,
    baseline_average_eta_seconds: 300,
    score_difference: 10.0,
    improvement_percentage: 20.0,
    ...overrides,
  };
}

describe("BaselineComparisonSection", () => {
  it("displays the baseline score exactly from API data", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ baseline_score: 63.4 })} />);

    expect(screen.getByText("Baseline Score")).toBeInTheDocument();
    expect(screen.getByText("63.4")).toBeInTheDocument();
  });

  it("displays the baseline coverage score", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ baseline_coverage_score: 40 })} />);

    expect(screen.getByText("Baseline Coverage")).toBeInTheDocument();
    expect(screen.getByText("40.0%")).toBeInTheDocument();
  });

  it("displays the baseline average ETA", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ baseline_average_eta_seconds: 300 })} />);

    expect(screen.getByText("Baseline Average ETA")).toBeInTheDocument();
    expect(screen.getByText("5m 0s")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null baseline average ETA", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ baseline_average_eta_seconds: null })} />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("displays the score difference", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ score_difference: 12.5 })} />);

    expect(screen.getByText("Score Difference")).toBeInTheDocument();
    expect(screen.getByText("+12.5")).toBeInTheDocument();
  });

  it("displays a negative score difference without a plus sign", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ score_difference: -8.0 })} />);

    expect(screen.getByText("-8.0")).toBeInTheDocument();
  });

  it("displays the improvement percentage", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ improvement_percentage: 20.0 })} />);

    expect(screen.getByText("Improvement")).toBeInTheDocument();
    expect(screen.getByText("+20.0%")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null improvement percentage", () => {
    render(<BaselineComparisonSection comparison={makeComparison({ improvement_percentage: null })} />);

    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
  });

  it('shows "Baseline comparison not available" when comparison is null, without hiding the section', () => {
    render(<BaselineComparisonSection comparison={null} />);

    expect(screen.getByText("Baseline Comparison")).toBeInTheDocument();
    expect(screen.getByText("Baseline comparison not available")).toBeInTheDocument();
    expect(screen.getByText("No baseline comparison has been computed for this response plan.")).toBeInTheDocument();
    expect(screen.queryByText("Baseline Score")).not.toBeInTheDocument();
  });
});
