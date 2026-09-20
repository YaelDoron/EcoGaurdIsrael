import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ResponsePlanMetrics } from "../../types/responsePlan";
import { PlanMetrics } from "./PlanMetrics";

function makeMetrics(overrides: Partial<ResponsePlanMetrics> = {}): ResponsePlanMetrics {
  return {
    plan_score: 87.3,
    coverage_score: 60.0,
    average_eta_seconds: 125,
    ...overrides,
  };
}

describe("PlanMetrics", () => {
  it("does not show the internal algorithmic plan score", () => {
    render(<PlanMetrics metrics={makeMetrics({ plan_score: 87.3 })} />);

    expect(screen.queryByText("Plan Score")).not.toBeInTheDocument();
    expect(screen.queryByText("87.3")).not.toBeInTheDocument();
  });

  it("does not show a Coverage metric card", () => {
    render(<PlanMetrics metrics={makeMetrics({ coverage_score: 60 })} />);

    expect(screen.queryByText("Coverage")).not.toBeInTheDocument();
    expect(screen.queryByText("60.0%")).not.toBeInTheDocument();
  });

  it("displays the persisted average ETA as a human-readable duration", () => {
    render(<PlanMetrics metrics={makeMetrics({ average_eta_seconds: 125 })} />);

    expect(screen.getByText("Average Travel Time")).toBeInTheDocument();
    expect(screen.getByText("2m 5s")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state when average_eta_seconds is null", () => {
    render(<PlanMetrics metrics={makeMetrics({ average_eta_seconds: null })} />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("displays each metric as the exact persisted value, not derived from any action data", () => {
    // PlanMetricsProps only accepts `metrics` - there is no `actions` input
    // for this component to derive a value from. Two different metrics
    // objects must produce two different displays, straight from the
    // fields given.
    const { rerender } = render(<PlanMetrics metrics={makeMetrics({ average_eta_seconds: 60 })} />);
    expect(screen.getByText("1m 0s")).toBeInTheDocument();

    rerender(<PlanMetrics metrics={makeMetrics({ average_eta_seconds: 300 })} />);
    expect(screen.getByText("5m 0s")).toBeInTheDocument();
    expect(screen.queryByText("1m 0s")).not.toBeInTheDocument();
  });

  it("has no visible Plan Metrics heading but stays an accessible labelled section", () => {
    render(<PlanMetrics metrics={makeMetrics()} />);

    expect(screen.queryByText("Plan Metrics")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Plan metrics")).toBeInTheDocument();
  });
});
