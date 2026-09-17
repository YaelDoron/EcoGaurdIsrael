import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { ResponseActionCard } from "./ResponseActionCard";

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

describe("ResponseActionCard", () => {
  it("displays the resource ID", () => {
    render(<ResponseActionCard action={makeAction({ resource: { ...makeAction().resource, resource_id: "engine-9" } })} />);

    expect(screen.getByText("engine-9")).toBeInTheDocument();
  });

  it("displays the station ID", () => {
    render(<ResponseActionCard action={makeAction({ resource: { ...makeAction().resource, station_id: "station-42" } })} />);

    expect(screen.getByText("(station-42)")).toBeInTheDocument();
  });

  it("displays the station name when available", () => {
    render(
      <ResponseActionCard
        action={makeAction({ resource: { ...makeAction().resource, station_name: "North Station" } })}
      />,
    );

    expect(screen.getByText(/North Station/)).toBeInTheDocument();
  });

  it("shows an explicit unavailable state, and still shows the station ID, when the station name is missing", () => {
    render(
      <ResponseActionCard
        action={makeAction({
          resource: { ...makeAction().resource, station_name: null, station_id: "station-99" },
        })}
      />,
    );

    expect(screen.getByText("Not available")).toBeInTheDocument();
    expect(screen.getByText("(station-99)")).toBeInTheDocument();
  });

  it("displays the target ID", () => {
    render(<ResponseActionCard action={makeAction({ target: { ...makeAction().target, response_target_id: 77 } })} />);

    expect(screen.getByText("#77")).toBeInTheDocument();
  });

  it("renders active_fire as a human-readable label", () => {
    render(<ResponseActionCard action={makeAction({ target: { ...makeAction().target, target_type: "active_fire" } })} />);

    expect(screen.getByText("Active fire")).toBeInTheDocument();
  });

  it("renders predicted_risk as a human-readable label", () => {
    render(
      <ResponseActionCard action={makeAction({ target: { ...makeAction().target, target_type: "predicted_risk" } })} />,
    );

    expect(screen.getByText("Predicted risk")).toBeInTheDocument();
  });

  it("displays the target priority exactly as returned by the backend", () => {
    render(<ResponseActionCard action={makeAction({ target: { ...makeAction().target, priority_score: 0.63 } })} />);

    expect(screen.getByText("0.63")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null target priority", () => {
    render(<ResponseActionCard action={makeAction({ target: { ...makeAction().target, priority_score: null } })} />);

    const targetSection = screen.getByText("Priority").closest("div") as HTMLElement;
    expect(within(targetSection).getByText("Not available")).toBeInTheDocument();
  });

  it("formats ETA for readability without changing the underlying value", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, eta_seconds: 125 } })} />);

    expect(screen.getByText("2m 5s")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null ETA", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, eta_seconds: null } })} />);

    const etaRow = screen.getByText("ETA").closest("div") as HTMLElement;
    expect(within(etaRow).getByText("Not available")).toBeInTheDocument();
  });

  it("formats a sub-kilometer distance in meters", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, distance_meters: 850 } })} />);

    expect(screen.getByText("850 m")).toBeInTheDocument();
  });

  it("formats a large distance in kilometers", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, distance_meters: 1500 } })} />);

    expect(screen.getByText("1.5 km")).toBeInTheDocument();
  });

  it("shows an explicit unavailable state for a null distance", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, distance_meters: null } })} />);

    const distanceRow = screen.getByText("Distance").closest("div") as HTMLElement;
    expect(within(distanceRow).getByText("Not available")).toBeInTheDocument();
  });

  it("displays a reachable route status", () => {
    render(<ResponseActionCard action={makeAction({ route: { ...makeAction().route, status: "reachable" } })} />);

    expect(screen.getByText("Reachable")).toBeInTheDocument();
  });

  it("keeps an unreachable route explicitly unreachable", () => {
    render(
      <ResponseActionCard
        action={makeAction({
          route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
        })}
      />,
    );

    expect(screen.getByText("Unreachable")).toBeInTheDocument();
    expect(screen.queryByText("Reachable")).not.toBeInTheDocument();
  });

  it("keeps an unmappable route explicitly unmappable", () => {
    render(
      <ResponseActionCard
        action={makeAction({
          route: { status: "unmappable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
        })}
      />,
    );

    expect(screen.getByText("Unmappable")).toBeInTheDocument();
    expect(screen.queryByText("Reachable")).not.toBeInTheDocument();
  });

  it("does not interpret a missing route status as reachable", () => {
    render(
      <ResponseActionCard
        action={makeAction({
          route: { status: null, eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
        })}
      />,
    );

    const statusRow = screen.getByText("Route status").closest("div") as HTMLElement;
    expect(within(statusRow).getByText("Not available")).toBeInTheDocument();
    expect(screen.queryByText("Reachable")).not.toBeInTheDocument();
  });

  it("renders no select control when onSelect is omitted (existing Task 9 usage)", () => {
    render(<ResponseActionCard action={makeAction()} />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("calls onSelect with the action's resource_id when the select control is clicked", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    render(
      <ResponseActionCard
        action={makeAction({ resource: { ...makeAction().resource, resource_id: "engine-42" } })}
        onSelect={onSelect}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Highlight on map" }));

    expect(onSelect).toHaveBeenCalledWith("engine-42");
  });

  it("shows a visually distinct selected state via aria-pressed when isSelected is true", () => {
    render(<ResponseActionCard action={makeAction()} isSelected onSelect={() => {}} />);

    expect(screen.getByRole("button", { name: "Selected" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the unselected select control label when isSelected is false", () => {
    render(<ResponseActionCard action={makeAction()} isSelected={false} onSelect={() => {}} />);

    expect(screen.getByRole("button", { name: "Highlight on map" })).toHaveAttribute("aria-pressed", "false");
  });
});
