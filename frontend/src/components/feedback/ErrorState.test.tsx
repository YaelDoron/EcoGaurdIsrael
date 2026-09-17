import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ErrorState } from "./ErrorState";

describe("ErrorState", () => {
  it("renders the safe title and message it is given", () => {
    render(<ErrorState title="Unable to load data" message="Please try again." />);

    expect(screen.getByRole("heading", { name: "Unable to load data" })).toBeInTheDocument();
    expect(screen.getByText("Please try again.")).toBeInTheDocument();
  });

  it("does not render a retry button when onRetry is not provided", () => {
    render(<ErrorState title="Unable to load data" />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders a real button when onRetry is provided", () => {
    render(<ErrorState title="Unable to load data" onRetry={() => {}} />);

    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("invokes onRetry when the retry button is clicked", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    render(<ErrorState title="Unable to load data" onRetry={onRetry} />);

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("supports a custom retry label", () => {
    render(<ErrorState title="Unable to load data" onRetry={() => {}} retryLabel="Try again" />);

    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});
