import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { ResponseActions } from "./ResponseActions";

vi.mock("./stationContact", () => ({
  getStationContact: (name: string | null) =>
    name === "Nesher" ? { display: "04-820-1111", href: "tel:048201111" } : null,
}));

function action(resourceId: string, stationId: string, stationName: string | null): ResponsePlanAction {
  return {
    resource: { resource_id: resourceId, station_id: stationId, station_name: stationName, origin: null },
    target: { response_target_id: 1, target_type: "active_fire", priority_score: 1, latitude: null, longitude: null },
    route: { status: "reachable", eta_seconds: 60, distance_meters: 100, node_path: null, path_coordinates: null },
  };
}

describe("ResponseActions station phone binding", () => {
  it("looks up each station on its own: only the station with a number shows one", () => {
    const { container } = render(
      <ResponseActions actions={[action("t1", "s1", "Nesher"), action("t2", "s2", "Isfiya"), action("t3", "s3", null)]} />,
    );

    const links = container.querySelectorAll<HTMLAnchorElement>("a.response-action-row__phone");
    expect(links).toHaveLength(1);
    expect(links[0]).toHaveTextContent("04-820-1111");
    expect(links[0]).toHaveAttribute("href", "tel:048201111");
    expect(links[0].closest("button")).toBeNull();
    expect(screen.getAllByText(/^Dispatch Station:/)).toHaveLength(3);
  });
});
