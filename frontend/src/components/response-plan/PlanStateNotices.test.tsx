import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PlanStateNotices } from "./PlanStateNotices";

describe("PlanStateNotices", () => {
  it("shows the partial-plan message when status is partial", () => {
    render(<PlanStateNotices plan={{ status: "partial", no_resources_during_planning: false }} />);

    expect(screen.getByText("This response plan does not cover all response targets.")).toBeInTheDocument();
  });

  it("does not show a partial warning when status is complete", () => {
    render(<PlanStateNotices plan={{ status: "complete", no_resources_during_planning: false }} />);

    expect(screen.queryByText("This response plan does not cover all response targets.")).not.toBeInTheDocument();
    expect(screen.queryByText("This response plan contains no feasible assignments.")).not.toBeInTheDocument();
  });

  it("renders nothing when status is complete and resources were available", () => {
    const { container } = render(
      <PlanStateNotices plan={{ status: "complete", no_resources_during_planning: false }} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("shows the no-feasible-assignments message for that persisted status", () => {
    render(<PlanStateNotices plan={{ status: "no_feasible_assignments", no_resources_during_planning: false }} />);

    expect(screen.getByText("This response plan contains no feasible assignments.")).toBeInTheDocument();
  });

  it("shows the no-resources message when no_resources_during_planning is true", () => {
    render(<PlanStateNotices plan={{ status: "complete", no_resources_during_planning: true }} />);

    expect(
      screen.getByText("No firefighting resources were available in the planning snapshot used for this plan."),
    ).toBeInTheDocument();
  });

  it("does not show the no-resources message when no_resources_during_planning is false", () => {
    render(<PlanStateNotices plan={{ status: "complete", no_resources_during_planning: false }} />);

    expect(
      screen.queryByText("No firefighting resources were available in the planning snapshot used for this plan."),
    ).not.toBeInTheDocument();
  });

  it("shows both partial and no-resources notices together without either overwriting the other", () => {
    render(<PlanStateNotices plan={{ status: "partial", no_resources_during_planning: true }} />);

    expect(screen.getByText("This response plan does not cover all response targets.")).toBeInTheDocument();
    expect(
      screen.getByText("No firefighting resources were available in the planning snapshot used for this plan."),
    ).toBeInTheDocument();
  });
});
