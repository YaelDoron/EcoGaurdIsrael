import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { FireSeverityLevel } from "../../types/fireEvent";
import { SeverityBadge } from "./SeverityBadge";

describe("SeverityBadge", () => {
  const cases: Array<[FireSeverityLevel, string]> = [
    ["low", "Low"],
    ["moderate", "Moderate"],
    ["high", "High"],
    ["critical", "Critical"],
  ];

  it.each(cases)("renders the readable label for level=%s", (level, label) => {
    render(<SeverityBadge level={level} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('renders "Not available" for a null level instead of fabricating a value', () => {
    render(<SeverityBadge level={null} />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
    for (const label of ["Low", "Moderate", "High", "Critical"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });
});
