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
  it("displays the persisted plan score", () => {
    render(<PlanMetrics metrics={makeMetrics({ plan_score: 87.3 })} />);

    expect(screen.getByText("Plan Score")).toBeInTheDocument();
    expect(screen.getByText("87.3")).toBeInTheDocument();
  });

  it("displays the persisted coverage score as a percentage", () => {
    render(<PlanMetrics metrics={makeMetrics({ coverage_score: 60 })} />);

    expect(screen.getByText("Coverage")).toBeInTheDocument();
    expect(screen.getByText("60.0%")).toBeInTheDocument();
  });

  it("displays the persisted average ETA as a human-readable duration", () => {
    render(<PlanMetrics metrics={makeMetrics({ average_eta_seconds: 125 })} />);

    expect(screen.getByText("Average ETA")).toBeInTheDocument();
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
    const { rerender } = render(<PlanMetrics metrics={makeMetrics({ plan_score: 10, coverage_score: 20 })} />);
    expect(screen.getByText("10.0")).toBeInTheDocument();
    expect(screen.getByText("20.0%")).toBeInTheDocument();

    rerender(<PlanMetrics metrics={makeMetrics({ plan_score: 99.9, coverage_score: 5.5 })} />);
    expect(screen.getByText("99.9")).toBeInTheDocument();
    expect(screen.getByText("5.5%")).toBeInTheDocument();
    expect(screen.queryByText("10.0")).not.toBeInTheDocument();
  });
});
