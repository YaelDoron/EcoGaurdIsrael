import { render, screen } from "@testing-library/react";
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
      response_target_id: 223,
      target_type: "active_fire",
      priority_score: 179.56,
      latitude: 32.1,
      longitude: 35.1,
    },
    route: {
      status: "reachable",
      eta_seconds: 541,
      distance_meters: 8500,
      node_path: [1, 2, 3],
      path_coordinates: null,
    },
    ...overrides,
  };
}

describe("ResponseActionCard", () => {
  it("reads as one horizontal journey: station (truck) -> target with distance and ETA", () => {
    const { container } = render(<ResponseActionCard action={makeAction()} />);

    expect(container.querySelector(".response-action-card__station")).toHaveTextContent("Station: Central Station");
    expect(container.querySelector(".response-action-card__truck")).toHaveTextContent("(engine-1)");
    expect(screen.getByText("Active fire")).toBeInTheDocument();
    expect(container.querySelector(".response-action-card__stats")).toHaveTextContent("8.5 km • ETA: 9m 1s");
    expect(container.querySelector(".response-action-card__eta")).toHaveTextContent("9m 1s");
  });

  it("does not expose raw algorithm values: target id or priority score", () => {
    const { container } = render(<ResponseActionCard action={makeAction()} />);

    expect(container.textContent).not.toContain("223");
    expect(container.textContent).not.toContain("179.56");
    expect(container.textContent).not.toMatch(/Priority|Target #/);
  });

  it("renders the origin station in English", () => {
    const action = makeAction({ resource: { ...makeAction().resource, station_name: "נשר" } });
    const { container } = render(<ResponseActionCard action={action} />);

    expect(container.querySelector(".response-action-card__station")).toHaveTextContent("Station: Nesher");
    expect(container.textContent).not.toMatch(/[\u0590-\u05FF]/);
  });

  it("falls back to the station id when the station name is missing", () => {
    const action = makeAction({ resource: { ...makeAction().resource, station_id: "station-99", station_name: null } });
    render(<ResponseActionCard action={action} />);

    expect(screen.getByText(/station-99/)).toBeInTheDocument();
  });

  it("renders predicted_risk as a human-readable label", () => {
    render(<ResponseActionCard action={makeAction({ target: { ...makeAction().target, target_type: "predicted_risk" } })} />);

    expect(screen.getByText("Predicted risk")).toBeInTheDocument();
  });

  it("formats a sub-kilometer distance in meters", () => {
    const { container } = render(
      <ResponseActionCard action={makeAction({ route: { ...makeAction().route, distance_meters: 850 } })} />,
    );

    expect(container.querySelector(".response-action-card__stats")).toHaveTextContent("850 m");
  });

  it("shows an explicit unavailable state for a null ETA and a null distance", () => {
    const { container } = render(
      <ResponseActionCard
        action={makeAction({ route: { ...makeAction().route, eta_seconds: null, distance_meters: null } })}
      />,
    );

    expect(container.querySelector(".response-action-card__stats")).toHaveTextContent(
      "Not available • ETA: Not available",
    );
  });

  it("shows no route-status chip for a reachable route", () => {
    render(<ResponseActionCard action={makeAction()} />);

    expect(screen.queryByText(/^Route:/)).not.toBeInTheDocument();
  });

  it("keeps an unreachable route explicitly unreachable", () => {
    const route = { status: "unreachable" as const, eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null };
    render(<ResponseActionCard action={makeAction({ route })} />);

    expect(screen.getByText("Route: Unreachable")).toBeInTheDocument();
  });

  it("keeps an unmappable route explicitly unmappable", () => {
    const route = { status: "unmappable" as const, eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null };
    render(<ResponseActionCard action={makeAction({ route })} />);

    expect(screen.getByText("Route: Unmappable")).toBeInTheDocument();
  });

  it("does not interpret a missing route status as reachable", () => {
    const route = { status: null, eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null };
    render(<ResponseActionCard action={makeAction({ route })} />);

    expect(screen.getByText("Route: Not available")).toBeInTheDocument();
  });

  it("renders no select control when onSelect is omitted", () => {
    render(<ResponseActionCard action={makeAction()} />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("calls onSelect with the action's key when the select control is clicked", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<ResponseActionCard action={makeAction()} onSelect={onSelect} />);

    await user.click(screen.getByRole("button", { name: "Highlight on map" }));

    expect(onSelect).toHaveBeenCalledWith("engine-1");
  });

  it("shows a distinct selected state via aria-pressed when isSelected is true", () => {
    render(<ResponseActionCard action={makeAction()} isSelected onSelect={() => {}} />);

    expect(screen.getByRole("button", { name: "Selected" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the unselected control label when isSelected is false", () => {
    render(<ResponseActionCard action={makeAction()} onSelect={() => {}} />);

    expect(screen.getByRole("button", { name: "Highlight on map" })).toHaveAttribute("aria-pressed", "false");
  });
});
