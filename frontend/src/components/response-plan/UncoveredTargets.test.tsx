import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ResponsePlanTarget } from "../../types/responsePlan";
import { UncoveredTargets } from "./UncoveredTargets";

function makeTarget(overrides: Partial<ResponsePlanTarget> = {}): ResponsePlanTarget {
  return {
    response_target_id: 1,
    target_type: "active_fire",
    priority_score: 0.75,
    latitude: 32.1,
    longitude: 35.1,
    ...overrides,
  };
}

describe("UncoveredTargets", () => {
  it("renders every backend-provided uncovered target", () => {
    const targets = [makeTarget({ response_target_id: 5 }), makeTarget({ response_target_id: 2 }), makeTarget({ response_target_id: 9 })];

    render(<UncoveredTargets targets={targets} />);

    expect(screen.getByText("#5")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
    expect(screen.getByText("#9")).toBeInTheDocument();
  });

  it("preserves backend order", () => {
    const targets = [makeTarget({ response_target_id: 5 }), makeTarget({ response_target_id: 2 }), makeTarget({ response_target_id: 9 })];

    render(<UncoveredTargets targets={targets} />);

    const ids = screen.getAllByText(/^#\d+$/).map((el) => el.textContent);
    expect(ids).toEqual(["#5", "#2", "#9"]);
  });

  it("displays the target ID", () => {
    render(<UncoveredTargets targets={[makeTarget({ response_target_id: 42 })]} />);

    expect(screen.getByText("#42")).toBeInTheDocument();
  });

  it("renders active_fire as a human-readable label", () => {
    render(<UncoveredTargets targets={[makeTarget({ target_type: "active_fire" })]} />);

    expect(screen.getByText("Active fire")).toBeInTheDocument();
  });

  it("renders predicted_risk as a human-readable label", () => {
    render(<UncoveredTargets targets={[makeTarget({ target_type: "predicted_risk" })]} />);

    expect(screen.getByText("Predicted risk")).toBeInTheDocument();
  });

  it("displays priority exactly as returned by the backend", () => {
    render(<UncoveredTargets targets={[makeTarget({ priority_score: 0.42 })]} />);

    expect(screen.getByText("0.42")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null priority", () => {
    render(<UncoveredTargets targets={[makeTarget({ priority_score: null })]} />);

    const priorityRow = screen.getByText("Priority").closest("div") as HTMLElement;
    expect(within(priorityRow).getByText("Not available")).toBeInTheDocument();
  });

  it("displays coordinates when available", () => {
    render(<UncoveredTargets targets={[makeTarget({ latitude: 32.1, longitude: 35.1 })]} />);

    expect(screen.getByText("32.1000, 35.1000")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state when coordinates are missing", () => {
    render(<UncoveredTargets targets={[makeTarget({ latitude: null, longitude: null })]} />);

    const coordinatesRow = screen.getByText("Coordinates").closest("div") as HTMLElement;
    expect(within(coordinatesRow).getByText("Not available")).toBeInTheDocument();
  });

  it("shows the all-covered state when the uncovered target list is empty", () => {
    render(<UncoveredTargets targets={[]} />);

    expect(screen.getByText("All response targets are covered by this plan.")).toBeInTheDocument();
  });

  it("renders only the targets it is given - never one derived from action data", () => {
    // UncoveredTargetsProps only accepts `targets` - there is no `actions`
    // input for this component to derive a target from.
    render(<UncoveredTargets targets={[makeTarget({ response_target_id: 1 })]} />);

    expect(screen.getAllByText(/^#\d+$/)).toHaveLength(1);
  });
});
