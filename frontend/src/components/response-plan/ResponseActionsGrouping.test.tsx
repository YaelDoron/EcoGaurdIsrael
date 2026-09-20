import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { ResponseActions } from "./ResponseActions";
import { RouteLegend } from "./RouteLegend";

function action(resourceId: string, targetId: number, overrides: Partial<ResponsePlanAction> = {}): ResponsePlanAction {
  return {
    resource: { resource_id: resourceId, station_id: "s1", station_name: "Central Station", origin: null },
    target: { response_target_id: targetId, target_type: "active_fire", priority_score: 0.9, latitude: 1, longitude: 1 },
    route: { status: "reachable", eta_seconds: 125, distance_meters: 800, node_path: null, path_coordinates: null },
    ...overrides,
  };
}

describe("ResponseActions grouping", () => {
  it("creates one card per destination target, holding all of its trucks", () => {
    render(
      <ResponseActions actions={[action("engine-1", 1), action("engine-2", 2), action("engine-3", 1)]} />,
    );

    const groups = screen.getAllByRole("article");
    expect(groups).toHaveLength(2);
    expect(within(groups[0]).getByRole("heading", { name: "Target #1 - Active fire" })).toBeInTheDocument();
    expect(within(groups[0]).getByText("engine-1")).toBeInTheDocument();
    expect(within(groups[0]).getByText("engine-3")).toBeInTheDocument();
    expect(within(groups[1]).getByText("engine-2")).toBeInTheDocument();
    expect(within(groups[0]).getByText(/2 trucks/)).toBeInTheDocument();
  });

  it("shows resource id, origin station name, route status and ETA for each truck", () => {
    render(<ResponseActions actions={[action("engine-1", 1)]} />);

    expect(screen.getByText("engine-1")).toBeInTheDocument();
    expect(screen.getByText("Central Station")).toBeInTheDocument();
    // A reachable route is the norm, so no "Reachable" badge is shown.
    expect(screen.queryByText("Reachable")).not.toBeInTheDocument();
    expect(screen.getByText("Travel time 2m 5s")).toBeInTheDocument();
  });

  it("falls back to the station id when the station name is unknown", () => {
    render(
      <ResponseActions
        actions={[action("engine-1", 1, { resource: { resource_id: "engine-1", station_id: "s9", station_name: null, origin: null } })]}
      />,
    );

    expect(screen.getByText("s9")).toBeInTheDocument();
  });
});

describe("RouteLegend", () => {
  it("explains the selected route, other routes, resource and target markers", () => {
    render(<RouteLegend />);

    for (const label of ["Selected route", "Other routes", "Resource (origin)", "Target"]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
  });
});
