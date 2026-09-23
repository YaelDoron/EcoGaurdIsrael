import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FireDangerLayer } from "./FireDangerLayer";
import type { FireDangerArea } from "../../types/fireDanger";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeArea(overrides: Partial<FireDangerArea> = {}): FireDangerArea {
  return {
    area_id: "area-carmel",
    area_name: "Carmel",
    center: { latitude: 32.731, longitude: 35.046 },
    radius_km: 5,
    assessment: {
      assessment_id: 1,
      status: "valid",
      score: 42.5,
      level: "very_high",
      assessed_at: "2026-09-20T11:00:00Z",
      age_seconds: 60,
      methodology: "FOSBERG_FFWI",
      methodology_version: "1.0",
    },
    ...overrides,
  };
}

describe("FireDangerLayer", () => {
  it("renders nothing for an empty areas array", () => {
    const { container } = render(<FireDangerLayer areas={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one circle per area with the correct center and radius in meters", () => {
    render(<FireDangerLayer areas={[makeArea(), makeArea({ area_id: "area-golan", radius_km: 7 })]} />);

    const circles = screen.getAllByTestId("circle");
    expect(circles).toHaveLength(2);
    expect(circles[0]).toHaveAttribute("data-lat", "32.731");
    expect(circles[0]).toHaveAttribute("data-lng", "35.046");
    expect(circles[0]).toHaveAttribute("data-radius", "5000");
    expect(circles[1]).toHaveAttribute("data-radius", "7000");
  });

  it("uses the persisted level to control fill color - never recalculated", () => {
    render(
      <FireDangerLayer
        areas={[
          makeArea({ area_id: "area-low", assessment: { ...makeArea().assessment!, level: "low" } }),
          makeArea({ area_id: "area-extreme", assessment: { ...makeArea().assessment!, level: "extreme" } }),
        ]}
      />,
    );

    const circles = screen.getAllByTestId("circle");
    const lowColor = circles[0].getAttribute("data-fill-color");
    const extremeColor = circles[1].getAttribute("data-fill-color");
    expect(lowColor).not.toBe(extremeColor);
    expect(lowColor).toBe("var(--color-success)");
    expect(extremeColor).toBe("var(--color-fire-danger-extreme)");
  });

  it("supports all five Fire Danger levels with distinct colors", () => {
    const levels = ["low", "moderate", "high", "very_high", "extreme"] as const;
    render(
      <FireDangerLayer
        areas={levels.map((level, index) => makeArea({ area_id: `area-${index}`, assessment: { ...makeArea().assessment!, level } }))}
      />,
    );

    const circles = screen.getAllByTestId("circle");
    const colors = circles.map((circle) => circle.getAttribute("data-fill-color"));
    expect(new Set(colors).size).toBe(5);
  });

  it("renders a known area with no assessment in a neutral color, never implying low risk", () => {
    render(<FireDangerLayer areas={[makeArea({ assessment: null })]} />);

    const circle = screen.getByTestId("circle");
    expect(circle.getAttribute("data-fill-color")).toBe("var(--color-text-muted)");
    expect(circle.getAttribute("data-fill-color")).not.toBe("var(--color-success)");
  });

  it("shows the real area name, level, score, and assessed time in the popup", () => {
    render(<FireDangerLayer areas={[makeArea({ area_name: "Jerusalem Forest" })]} />);

    expect(screen.getByText("Jerusalem Forest")).toBeInTheDocument();
    expect(screen.getByText("Level: Very High")).toBeInTheDocument();
    expect(screen.getByText("FFWI: 42.5")).toBeInTheDocument();
  });

  it("labels the danger level 'Level', never bare or ambiguous", () => {
    render(<FireDangerLayer areas={[makeArea()]} />);

    expect(screen.getByText("Level: Very High")).toBeInTheDocument();
  });

  it("renders VERY_HIGH human-readably as 'Very High'", () => {
    render(<FireDangerLayer areas={[makeArea({ assessment: { ...makeArea().assessment!, level: "very_high" } })]} />);

    expect(screen.getByText("Level: Very High")).toBeInTheDocument();
  });

  it("labels the persisted score 'FFWI', never a bare/generic 'Score'", () => {
    render(<FireDangerLayer areas={[makeArea()]} />);

    expect(screen.getByText("FFWI: 42.5")).toBeInTheDocument();
    expect(screen.queryByText(/^Score:/)).not.toBeInTheDocument();
    expect(screen.queryByText("Score: 42.5")).not.toBeInTheDocument();
  });

  it("shows the persisted score value unchanged - never recalculated", () => {
    render(<FireDangerLayer areas={[makeArea({ assessment: { ...makeArea().assessment!, score: 87.3 } })]} />);

    expect(screen.getByText("FFWI: 87.3")).toBeInTheDocument();
  });

  it("never labels the Fire Danger level as Severity - these are distinct domains", () => {
    render(<FireDangerLayer areas={[makeArea()]} />);

    expect(screen.queryByText(/Severity/)).not.toBeInTheDocument();
  });

  it("gives the FFWI score a concise help affordance, not a paragraph of explanation", () => {
    render(<FireDangerLayer areas={[makeArea()]} />);

    const scoreElement = screen.getByText("FFWI: 42.5");
    expect(scoreElement).toHaveAttribute(
      "title",
      "Fire-weather danger score used to determine the Fire Danger level.",
    );
  });

  // -------------------------------------------------------------------------
  // Popup identity + FFWI consistency across all levels (Parts 3/5)
  // -------------------------------------------------------------------------

  it('includes an explicit "Fire Danger Assessment" label, distinguishing it from a FireEvent popup', () => {
    render(<FireDangerLayer areas={[makeArea()]} />);

    expect(screen.getByText("Fire Danger Assessment")).toBeInTheDocument();
  });

  it.each([
    ["low", "Low"],
    ["high", "High"],
    ["extreme", "Extreme"],
  ] as const)(
    "shows FFWI for level=%s just like every other level (never hidden by level)",
    (level, expectedLabel) => {
      render(
        <FireDangerLayer
          areas={[makeArea({ assessment: { ...makeArea().assessment!, level, score: 12.5 } })]}
        />,
      );

      expect(screen.getByText(`Level: ${expectedLabel}`)).toBeInTheDocument();
      expect(screen.getByText("FFWI: 12.5")).toBeInTheDocument();
    },
  );

  it("shows 'No assessment yet' for a known area with a null assessment", () => {
    render(<FireDangerLayer areas={[makeArea({ assessment: null })]} />);

    expect(screen.getByText("No assessment yet")).toBeInTheDocument();
  });

  it("shows insufficient-data messaging without a level when status is insufficient_data", () => {
    render(
      <FireDangerLayer
        areas={[
          makeArea({
            assessment: {
              assessment_id: 2,
              status: "insufficient_data",
              score: null,
              level: null,
              assessed_at: "2026-09-20T11:00:00Z",
              age_seconds: 60,
              methodology: "FOSBERG_FFWI",
              methodology_version: "1.0",
            },
          }),
        ]}
      />,
    );

    expect(screen.getByText("Insufficient data for an assessment")).toBeInTheDocument();
  });
});
