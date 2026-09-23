import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ResponseTargetLayer } from "./ResponseTargetLayer";
import type { ResponseTarget } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeTarget(overrides: Partial<ResponseTarget> = {}): ResponseTarget {
  return {
    target_order: 0,
    target_type: "active_fire",
    latitude: 32.731,
    longitude: 35.046,
    priority_score: 1.0,
    prediction_horizon_minutes: null,
    ...overrides,
  };
}

describe("ResponseTargetLayer", () => {
  it("renders nothing when there are no targets", () => {
    const { container } = render(<ResponseTargetLayer targets={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one marker per target", () => {
    render(
      <ResponseTargetLayer
        targets={[makeTarget({ target_order: 0 }), makeTarget({ target_order: 1, latitude: 32.8 })]}
      />,
    );

    expect(screen.getAllByTestId("marker")).toHaveLength(2);
  });

  it("visually distinguishes ACTIVE_FIRE (filled) from PREDICTED_RISK (hollow) markers", () => {
    render(
      <ResponseTargetLayer
        targets={[
          makeTarget({ target_order: 0, target_type: "active_fire" }),
          makeTarget({
            target_order: 1,
            target_type: "predicted_risk",
            latitude: 32.8,
            prediction_horizon_minutes: 30,
          }),
        ]}
      />,
    );

    const icons = screen.getAllByTestId("marker-icon").map((el) => el.querySelector("span") as HTMLElement);
    const [activeFireIcon, predictedRiskIcon] = icons;

    expect(activeFireIcon.style.background).toBe("var(--color-danger)");
    expect(predictedRiskIcon.style.background).toBe("transparent");
    expect(predictedRiskIcon.style.border).toContain("dashed");
  });

  it("shows target type, a human-readable operational term (not the raw priority score), and horizon (when present) in the popup", () => {
    render(
      <ResponseTargetLayer
        targets={[makeTarget({ target_type: "predicted_risk", priority_score: 80, prediction_horizon_minutes: 60 })]}
      />,
    );

    expect(screen.getByText("Predicted risk")).toBeInTheDocument();
    expect(screen.getByText("Risk: Critical")).toBeInTheDocument();
    expect(screen.queryByText(/Priority score/)).not.toBeInTheDocument();
    expect(screen.getByText("Prediction horizon: 60 min")).toBeInTheDocument();
  });

  it("omits the horizon line for an ACTIVE_FIRE target (prediction_horizon_minutes is null)", () => {
    render(<ResponseTargetLayer targets={[makeTarget({ target_type: "active_fire" })]} />);

    expect(screen.queryByText(/Prediction horizon/)).not.toBeInTheDocument();
  });

  it.each([
    [100, "Low"],
    [125, "Moderate"],
    [150, "High"],
    [187.4, "Critical"],
  ] as const)(
    "labels an ACTIVE_FIRE target with priority_score %s as Severity: %s (severity_score = priority_score - 100)",
    (priorityScore, level) => {
      render(<ResponseTargetLayer targets={[makeTarget({ target_type: "active_fire", priority_score: priorityScore })]} />);

      expect(screen.getByText(`Severity: ${level}`)).toBeInTheDocument();
    },
  );

  it("shows the event's location name in every target's popup when known", () => {
    render(
      <ResponseTargetLayer
        targets={[makeTarget({ target_order: 0 }), makeTarget({ target_order: 1, target_type: "predicted_risk" })]}
        eventLocationName="HaGalil"
      />,
    );

    expect(screen.getAllByText("HaGalil")).toHaveLength(2);
  });

  it("omits the location name line when none is known", () => {
    render(<ResponseTargetLayer targets={[makeTarget()]} />);

    expect(screen.queryByText("HaGalil")).not.toBeInTheDocument();
  });
});
