import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GlobalMetricsPanel } from "./GlobalMetricsPanel";
import type { GlobalPlanMetrics, GlobalPlanShortage } from "../../types/globalResponsePlan";

function makeMetrics(overrides: Partial<GlobalPlanMetrics> = {}): GlobalPlanMetrics {
  return { fitness_score: 91.5, coverage_score: 95.0, average_eta_seconds: 180.0, ...overrides };
}

function makeShortage(overrides: Partial<GlobalPlanShortage> = {}): GlobalPlanShortage {
  return { total_required: 3, total_desired: 6, total_assigned: 4, unmet_required: 0, unmet_desired: 2, ...overrides };
}

describe("GlobalMetricsPanel", () => {
  it("displays coverage and average ETA, and no fitness score", () => {
    render(<GlobalMetricsPanel metrics={makeMetrics()} shortage={makeShortage()} />);

    expect(screen.getByText("95.0%")).toBeInTheDocument();
    expect(screen.getByText("3m 0s")).toBeInTheDocument();
    expect(screen.queryByText("Fitness Score")).not.toBeInTheDocument();
    expect(screen.queryByText("91.5")).not.toBeInTheDocument();
  });

  it("shows Not available for a null metric rather than fabricating a value", () => {
    render(
      <GlobalMetricsPanel
        metrics={{ fitness_score: null, coverage_score: null, average_eta_seconds: null }}
        shortage={makeShortage()}
      />,
    );

    expect(screen.getAllByText("Not available")).toHaveLength(2);
  });

  it("condenses the shortage into one summary line", () => {
    render(
      <GlobalMetricsPanel
        metrics={makeMetrics()}
        shortage={makeShortage({ total_required: 10, total_desired: 20, total_assigned: 15, unmet_required: 1, unmet_desired: 6 })}
      />,
    );

    const summary = screen.getByLabelText("Resource shortage");
    expect(summary).toHaveTextContent("Assigned: 15 / Desired: 20");
    expect(summary).toHaveTextContent("Unmet: 6");
    expect(summary).toHaveTextContent("Required: 10 | Unmet: 1");
    expect(summary).not.toHaveTextContent("(required)");
  });

  it("draws a progress bar of assigned versus desired, capped at 100%", () => {
    const { rerender } = render(
      <GlobalMetricsPanel metrics={makeMetrics()} shortage={makeShortage({ total_desired: 8, total_assigned: 2 })} />,
    );
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");

    rerender(<GlobalMetricsPanel metrics={makeMetrics()} shortage={makeShortage({ total_desired: 4, total_assigned: 6 })} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");

    rerender(<GlobalMetricsPanel metrics={makeMetrics()} shortage={makeShortage({ total_desired: 0, total_assigned: 0 })} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "0");
  });
});
