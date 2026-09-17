import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { ResponseRouteLayer } from "./ResponseRouteLayer";
import type { ResponseRouteLayerData } from "./ResponseRouteLayerModel";

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
      latitude: 32.5,
      longitude: 35.5,
    },
    route: {
      status: "reachable",
      eta_seconds: 120,
      distance_meters: 800,
      node_path: [1, 2, 3],
      path_coordinates: [
        { latitude: 32.0, longitude: 35.0 },
        { latitude: 32.5, longitude: 35.5 },
      ],
    },
    ...overrides,
  };
}

describe("ResponseRouteLayer", () => {
  it("renders nothing when no children render-prop is given", () => {
    const { container } = render(<ResponseRouteLayer actions={[makeAction()]} selectedActionKey={null} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("passes the computed layer data to children", () => {
    let received: ResponseRouteLayerData | null = null;

    render(
      <ResponseRouteLayer actions={[makeAction()]} selectedActionKey={null}>
        {(layer) => {
          received = layer;
          return <div data-testid="route-count">{layer.routes.length}</div>;
        }}
      </ResponseRouteLayer>,
    );

    expect(screen.getByTestId("route-count")).toHaveTextContent("1");
    expect(received).not.toBeNull();
    expect(received!.routes[0].path).toEqual(makeAction().route.path_coordinates);
  });

  it("reflects the selected action in the layer data passed to children", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2" } }),
    ];

    render(
      <ResponseRouteLayer actions={actions} selectedActionKey="engine-2">
        {(layer) => (
          <ul>
            {layer.routes.map((route) => (
              <li key={route.actionKey} data-selected={route.isSelected}>
                {route.resourceId}
              </li>
            ))}
          </ul>
        )}
      </ResponseRouteLayer>,
    );

    expect(screen.getByText("engine-1")).toHaveAttribute("data-selected", "false");
    expect(screen.getByText("engine-2")).toHaveAttribute("data-selected", "true");
  });

  it("does not draw a route for an unmappable action even when rendered via children", () => {
    const action = makeAction({
      route: { status: "unmappable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
    });

    render(
      <ResponseRouteLayer actions={[action]} selectedActionKey={null}>
        {(layer) => <div data-testid="route-count">{layer.routes.length}</div>}
      </ResponseRouteLayer>,
    );

    expect(screen.getByTestId("route-count")).toHaveTextContent("0");
  });
});
