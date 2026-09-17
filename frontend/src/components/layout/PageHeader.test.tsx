import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PageHeader } from "./PageHeader";

describe("PageHeader", () => {
  it("renders the title as a level-1 heading", () => {
    render(<PageHeader title="Active Wildfires" />);

    expect(screen.getByRole("heading", { level: 1, name: "Active Wildfires" })).toBeInTheDocument();
  });

  it("renders an optional description", () => {
    render(<PageHeader title="Active Wildfires" description="Current suspected and confirmed wildfire events." />);

    expect(screen.getByText("Current suspected and confirmed wildfire events.")).toBeInTheDocument();
  });

  it("omits the description when none is given", () => {
    render(<PageHeader title="Active Wildfires" />);

    expect(screen.queryByText(/current suspected/i)).not.toBeInTheDocument();
  });

  it("renders optional actions", () => {
    render(<PageHeader title="Active Wildfires" actions={<button type="button">Refresh</button>} />);

    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
  });
});
