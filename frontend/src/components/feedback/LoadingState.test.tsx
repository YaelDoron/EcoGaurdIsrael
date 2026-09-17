import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LoadingState } from "./LoadingState";

describe("LoadingState", () => {
  it("renders a default message", () => {
    render(<LoadingState />);

    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("renders a custom message", () => {
    render(<LoadingState message="Loading active wildfire events…" />);

    expect(screen.getByText("Loading active wildfire events…")).toBeInTheDocument();
  });

  it("exposes accessible status semantics", () => {
    render(<LoadingState message="Loading…" />);

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Loading…");
  });
});
