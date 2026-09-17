import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { ResponseActions } from "./ResponseActions";

function makeAction(overrides: Partial<ResponsePlanAction> = {}): ResponsePlanAction {
  return {
    resource: {
      resource_id: "engine-1",
      station_id: "station-1",
      station_name: "Central Station",
      origin: { latitude: 32.0, longitude: 35.0 },
    },
    target: {
      response_target_id: 1,
      target_type: "active_fire",
      priority_score: 0.75,
      latitude: 32.1,
      longitude: 35.1,
    },
    route: {
      status: "reachable",
      eta_seconds: 125,
      distance_meters: 850,
      node_path: [1, 2, 3],
      path_coordinates: null,
    },
    ...overrides,
  };
}

describe("ResponseActions", () => {
  it("renders every action returned by the backend", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-3" } }),
    ];

    render(<ResponseActions actions={actions} />);

    expect(screen.getByText("engine-1")).toBeInTheDocument();
    expect(screen.getByText("engine-2")).toBeInTheDocument();
    expect(screen.getByText("engine-3")).toBeInTheDocument();
  });

  it("preserves backend action order without sorting by ETA or priority", () => {
    const actions = [
      makeAction({
        resource: { ...makeAction().resource, resource_id: "slow-low-priority" },
        route: { ...makeAction().route, eta_seconds: 900 },
        target: { ...makeAction().target, priority_score: 0.1 },
      }),
      makeAction({
        resource: { ...makeAction().resource, resource_id: "fast-high-priority" },
        route: { ...makeAction().route, eta_seconds: 30 },
        target: { ...makeAction().target, priority_score: 0.9 },
      }),
    ];

    render(<ResponseActions actions={actions} />);

    const resourceIds = screen.getAllByText(/^(slow-low-priority|fast-high-priority)$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["slow-low-priority", "fast-high-priority"]);
  });

  it("shows an explicit empty state when there are no actions", () => {
    render(<ResponseActions actions={[]} />);

    expect(screen.getByText("No response actions")).toBeInTheDocument();
    expect(screen.getByText("This response plan has no assigned resources.")).toBeInTheDocument();
  });

  it("does not fabricate an action when the actions array is empty", () => {
    render(<ResponseActions actions={[]} />);

    expect(screen.queryByText("engine-1")).not.toBeInTheDocument();
  });

  it("renders the section heading", () => {
    render(<ResponseActions actions={[makeAction()]} />);

    expect(screen.getByRole("heading", { name: "Response Actions" })).toBeInTheDocument();
  });

  it("highlights only the selected action's card", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="engine-2" onSelectAction={() => {}} />);

    const buttons = screen.getAllByRole("button");
    expect(buttons).toHaveLength(2);
    expect(buttons[0]).toHaveAttribute("aria-pressed", "false");
    expect(buttons[1]).toHaveAttribute("aria-pressed", "true");
  });

  it("calls onSelectAction with the clicked action's key", async () => {
    const user = userEvent.setup();
    const onSelectAction = vi.fn();
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } }),
    ];

    render(<ResponseActions actions={actions} onSelectAction={onSelectAction} />);

    await user.click(screen.getAllByRole("button")[1]);

    expect(onSelectAction).toHaveBeenCalledWith("engine-2");
  });

  it("does not change action order when a selection is made", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="engine-2" onSelectAction={() => {}} />);

    const resourceIds = screen.getAllByText(/^engine-[12]$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["engine-1", "engine-2"]);
  });

  it("keeps an action without a drawable route visible in the list regardless of selection", () => {
    const actions = [
      makeAction({
        resource: { ...makeAction().resource, resource_id: "engine-unreachable" },
        route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
      }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="some-other-action" onSelectAction={() => {}} />);

    expect(screen.getByText("engine-unreachable")).toBeInTheDocument();
    expect(screen.getByText("Unreachable")).toBeInTheDocument();
  });
});
