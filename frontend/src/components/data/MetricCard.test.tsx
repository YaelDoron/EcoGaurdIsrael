import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricCard } from "./MetricCard";

describe("MetricCard", () => {
  it("displays the label and value", () => {
    render(<MetricCard label="Active Events" value={3} />);

    expect(screen.getByText("Active Events")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("displays a string value as-is", () => {
    render(<MetricCard label="Status" value="Nominal" />);

    expect(screen.getByText("Nominal")).toBeInTheDocument();
  });

  it("renders optional helper text", () => {
    render(<MetricCard label="Active Events" value={3} helperText="Across all regions" />);

    expect(screen.getByText("Across all regions")).toBeInTheDocument();
  });

  it("omits helper text when none is given", () => {
    render(<MetricCard label="Active Events" value={3} />);

    expect(screen.queryByText(/across all regions/i)).not.toBeInTheDocument();
  });
});
