import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SpreadLayer } from "./SpreadLayer";
import type { SpreadPrediction } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeCell(overrides: Partial<SpreadPrediction["cells"][number]> = {}) {
  return {
    latitude: 32.74,
    longitude: 35.05,
    spread_probability: 0.6,
    spread_risk_score: 60,
    reached_step: 1,
    reached_minutes: 5,
    ...overrides,
  };
}

describe("SpreadLayer", () => {
  it("renders nothing when every prediction has an empty cells array", () => {
    const predictions: SpreadPrediction[] = [
      { horizon_minutes: 30, status: "insufficient_data", predicted_at: "2026-09-17T13:25:00Z", cells: [] },
      { horizon_minutes: 60, status: "inactive_event", predicted_at: "2026-09-17T13:25:00Z", cells: [] },
    ];
    const { container } = render(<SpreadLayer predictions={predictions} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one circle marker per cell across all horizons", () => {
    const predictions: SpreadPrediction[] = [
      {
        horizon_minutes: 30,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [makeCell(), makeCell({ latitude: 32.75 })],
      },
      {
        horizon_minutes: 60,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [makeCell({ latitude: 32.76 })],
      },
    ];
    render(<SpreadLayer predictions={predictions} />);

    expect(screen.getAllByTestId("circle-marker")).toHaveLength(3);
  });

  it("skips a horizon with no cells while still rendering another horizon's valid cells", () => {
    const predictions: SpreadPrediction[] = [
      { horizon_minutes: 30, status: "insufficient_data", predicted_at: "2026-09-17T13:25:00Z", cells: [] },
      { horizon_minutes: 60, status: "valid", predicted_at: "2026-09-17T13:25:00Z", cells: [makeCell()] },
    ];
    render(<SpreadLayer predictions={predictions} />);

    expect(screen.getAllByTestId("circle-marker")).toHaveLength(1);
  });

  it("shows the horizon, risk score, and probability in each cell's popup", () => {
    const predictions: SpreadPrediction[] = [
      {
        horizon_minutes: 30,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [makeCell({ spread_risk_score: 72.5, spread_probability: 0.5, reached_minutes: 10 })],
      },
    ];
    render(<SpreadLayer predictions={predictions} />);

    // p = 0.5 is exactly the propagation threshold -> a spreading cell.
    expect(screen.getByText("Predicted spread (30 min horizon)")).toBeInTheDocument();
    expect(screen.getByText("Reached the propagation threshold")).toBeInTheDocument();
    expect(screen.getByText("Risk score: 72.5")).toBeInTheDocument();
    expect(screen.getByText("Probability: 50%")).toBeInTheDocument();
    expect(screen.getByText("Model reach time: 10 min")).toBeInTheDocument();
    expect(screen.queryByText(/Reached at/)).not.toBeInTheDocument();
  });

  it("uses a higher fill opacity for a higher spread probability", () => {
    const predictions: SpreadPrediction[] = [
      {
        horizon_minutes: 30,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [makeCell({ spread_probability: 0.1 }), makeCell({ spread_probability: 0.9, latitude: 32.8 })],
      },
    ];
    render(<SpreadLayer predictions={predictions} />);

    const markers = screen.getAllByTestId("circle-marker");
    const lowOpacity = Number(markers[0].getAttribute("data-fill-opacity"));
    const highOpacity = Number(markers[1].getAttribute("data-fill-opacity"));
    expect(highOpacity).toBeGreaterThan(lowOpacity);
  });

  it("describes a risk-only cell as predicted spread risk, not spread", () => {
    const predictions: SpreadPrediction[] = [
      {
        horizon_minutes: 30,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [makeCell({ spread_probability: 0.38, spread_risk_score: 38, reached_minutes: 5 })],
      },
    ];
    render(<SpreadLayer predictions={predictions} />);

    expect(screen.getByText("Predicted spread risk (30 min horizon)")).toBeInTheDocument();
    expect(screen.getByText("Did not reach the propagation threshold")).toBeInTheDocument();
    expect(screen.getByText("Risk assessed at: 5 min")).toBeInTheDocument();
    expect(screen.getByText("Probability: 38%")).toBeInTheDocument();
    expect(screen.queryByText(/Model reach time|Reached the propagation threshold/)).not.toBeInTheDocument();
  });

  it("gives risk-only cells a dashed, lower-opacity style and keeps the spreading style unchanged", () => {
    const predictions: SpreadPrediction[] = [
      {
        horizon_minutes: 30,
        status: "valid",
        predicted_at: "2026-09-17T13:25:00Z",
        cells: [
          makeCell({ spread_probability: 0.49, latitude: 32.7 }),
          makeCell({ spread_probability: 0.5, latitude: 32.8 }),
        ],
      },
    ];
    render(<SpreadLayer predictions={predictions} />);

    const [riskOnly, spreading] = screen.getAllByTestId("circle-marker");
    expect(riskOnly.getAttribute("data-dash-array")).toBeTruthy();
    expect(spreading.getAttribute("data-dash-array")).toBeNull();
    // Spreading keeps the original opacity formula; risk-only is reduced below its own base value.
    expect(Number(spreading.getAttribute("data-fill-opacity"))).toBeCloseTo(0.35 + 0.5 * 0.4);
    expect(Number(riskOnly.getAttribute("data-fill-opacity"))).toBeLessThan(0.35 + 0.49 * 0.4);
  });
});
