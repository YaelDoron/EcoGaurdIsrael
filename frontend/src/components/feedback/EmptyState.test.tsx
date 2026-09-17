import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EmptyState } from "./EmptyState";

describe("EmptyState", () => {
  it("renders the title", () => {
    render(<EmptyState title="No active wildfire events" />);

    expect(screen.getByRole("heading", { name: "No active wildfire events" })).toBeInTheDocument();
  });

  it("renders an optional message", () => {
    render(
      <EmptyState
        title="No active wildfire events"
        message="There are currently no suspected or confirmed wildfire events."
      />,
    );

    expect(
      screen.getByText("There are currently no suspected or confirmed wildfire events."),
    ).toBeInTheDocument();
  });

  it("omits the message paragraph when none is given", () => {
    render(<EmptyState title="No active wildfire events" />);

    expect(screen.queryByText(/currently no/i)).not.toBeInTheDocument();
  });

  it("renders an optional action", () => {
    render(<EmptyState title="No active wildfire events" action={<button type="button">Refresh</button>} />);

    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
  });
});
