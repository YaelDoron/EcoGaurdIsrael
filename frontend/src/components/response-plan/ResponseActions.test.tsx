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
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-3", station_id: "station-engine-3" } }),
    ];

    render(<ResponseActions actions={actions} />);

    expect(screen.getByText("engine-1")).toBeInTheDocument();
    expect(screen.getByText("engine-2")).toBeInTheDocument();
    expect(screen.getByText("engine-3")).toBeInTheDocument();
  });

  it("preserves backend action order without sorting by ETA or priority", () => {
    const actions = [
      makeAction({
        resource: { ...makeAction().resource, resource_id: "slow-low-priority", station_id: "station-slow-low-priority" },
        route: { ...makeAction().route, eta_seconds: 900 },
        target: { ...makeAction().target, priority_score: 0.1 },
      }),
      makeAction({
        resource: { ...makeAction().resource, resource_id: "fast-high-priority", station_id: "station-fast-high-priority" },
        route: { ...makeAction().route, eta_seconds: 30 },
        target: { ...makeAction().target, priority_score: 0.9 },
      }),
    ];

    render(<ResponseActions actions={actions} />);

    const resourceIds = screen.getAllByText(/^(slow-low-priority|fast-high-priority)$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["slow-low-priority", "fast-high-priority"]);
  });

  it("shows the English origin station as the primary line, above a muted truck id", () => {
    const base = makeAction();
    const { container } = render(
      <ResponseActions
        actions={[makeAction({ resource: { ...base.resource, resource_id: "TRUCK-82-2", station_name: "נשר" } })]}
      />,
    );

    const station = container.querySelector(".response-action-row__station") as HTMLElement;
    const truck = container.querySelector(".response-action-row__truck") as HTMLElement;
    expect(station).toHaveTextContent("Dispatch Station: Nesher");
    expect(station.querySelector(".response-action-row__station-name")).toHaveTextContent(/^Nesher$/);
    expect(truck).toHaveTextContent("TRUCK-82-2");
    expect(container.textContent).not.toMatch(/[֐-׿]/);
    // station comes first in reading order
    expect(station.compareDocumentPosition(truck) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("lists a station once with all of its trucks as pills and a travel-time range", () => {
    const base = makeAction();
    const at = (id: string, eta: number) =>
      makeAction({
        resource: { ...base.resource, resource_id: id, station_id: "isfiya", station_name: "עוספיא" },
        route: { ...base.route, eta_seconds: eta },
      });
    const { container } = render(<ResponseActions actions={[at("TRUCK-87-2", 540), at("TRUCK-88-1", 720)]} />);

    expect(container.querySelectorAll(".response-action-row")).toHaveLength(1);
    expect(container.querySelector(".response-action-row__station-name")).toHaveTextContent(/^Isfiya$/);
    const pills = Array.from(container.querySelectorAll(".response-action-row__truck")).map((el) => el.textContent);
    expect(pills).toEqual(["TRUCK-87-2", "TRUCK-88-1"]);
    expect(container.querySelector(".response-action-row__chip")).toHaveTextContent(/Travel time .+ - .+/);
  });

  it("shows one travel time when every truck from the station has the same ETA", () => {
    const base = makeAction();
    const at = (id: string) => makeAction({ resource: { ...base.resource, resource_id: id } });
    const { container } = render(<ResponseActions actions={[at("a"), at("b")]} />);

    expect(container.querySelector(".response-action-row__chip")?.textContent).not.toContain(" - ");
  });

  it("selecting a merged station row highlights it when any of its trucks is selected", () => {
    const base = makeAction();
    const at = (id: string) => makeAction({ resource: { ...base.resource, resource_id: id } });
    render(<ResponseActions actions={[at("a"), at("b")]} selectedActionKey="b" onSelectAction={() => {}} />);

    expect(screen.getByRole("button", { name: "Selected" })).toHaveAttribute("aria-pressed", "true");
  });

  it("omits the phone row for a station whose only registry number is the shared hotline", () => {
    const base = makeAction();
    const { container } = render(
      <ResponseActions actions={[makeAction({ resource: { ...base.resource, station_name: "נשר" } })]} />,
    );

    expect(container.querySelector(".response-action-row__phone")).toBeNull();
    expect(container.textContent).not.toContain("*4964");
  });

  it("omits the phone row entirely for a station the registry does not know", () => {
    const { container } = render(<ResponseActions actions={[makeAction()]} />);

    expect(container.querySelector(".response-action-row__phone")).toBeNull();
    expect(container.querySelector("a[href^='tel:']")).toBeNull();
  });

  it("omits the phone row when the station has no name", () => {
    const base = makeAction();
    const { container } = render(
      <ResponseActions actions={[makeAction({ resource: { ...base.resource, station_name: null } })]} />,
    );

    expect(container.querySelector(".response-action-row__phone")).toBeNull();
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
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" } }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="engine-2" onSelectAction={() => {}} />);

    const buttons = screen.getAllByRole("button", { name: /Highlight on map|Selected/ });
    expect(buttons).toHaveLength(2);
    expect(buttons[0]).toHaveAttribute("aria-pressed", "false");
    expect(buttons[1]).toHaveAttribute("aria-pressed", "true");
  });

  it("calls onSelectAction with the clicked action's key", async () => {
    const user = userEvent.setup();
    const onSelectAction = vi.fn();
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" } }),
    ];

    render(<ResponseActions actions={actions} onSelectAction={onSelectAction} />);

    await user.click(screen.getAllByRole("button", { name: /Highlight on map|Selected/ })[1]);

    expect(onSelectAction).toHaveBeenCalledWith("engine-2");
  });

  it("does not change action order when a selection is made", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" } }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="engine-2" onSelectAction={() => {}} />);

    const resourceIds = screen.getAllByText(/^engine-[12]$/).map((el) => el.textContent);
    expect(resourceIds).toEqual(["engine-1", "engine-2"]);
  });

  it("keeps an action without a drawable route visible in the list regardless of selection", () => {
    const actions = [
      makeAction({
        resource: { ...makeAction().resource, resource_id: "engine-unreachable", station_id: "station-engine-unreachable" },
        route: { status: "unreachable", eta_seconds: null, distance_meters: null, node_path: null, path_coordinates: null },
      }),
    ];

    render(<ResponseActions actions={actions} selectedActionKey="some-other-action" onSelectAction={() => {}} />);

    expect(screen.getByText("engine-unreachable")).toBeInTheDocument();
    expect(screen.getByText("Unreachable")).toBeInTheDocument();
  });

  it("numbers each route and shows no Primary route badge", () => {
    const actions = [
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-1", station_id: "station-engine-1" } }),
      makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" } }),
    ];

    render(<ResponseActions actions={actions} />);

    expect(screen.getByLabelText("Route 1")).toBeInTheDocument();
    expect(screen.getByLabelText("Route 2")).toBeInTheDocument();
    expect(screen.queryByText(/primary route/i)).not.toBeInTheDocument();
  });

  it("titles groups with the inherited region name and target type instead of the raw target id", () => {
    render(<ResponseActions actions={[makeAction()]} locationName="Modiin" />);

    expect(screen.getByRole("heading", { name: "Active fire - Modiin" })).toBeInTheDocument();
    expect(screen.queryByText(/Target #1/)).not.toBeInTheDocument();
  });

  it("numbers groups that would otherwise share the same region title", () => {
    render(
      <ResponseActions
        actions={[
          makeAction({ target: { ...makeAction().target, response_target_id: 1 } }),
          makeAction({ resource: { ...makeAction().resource, resource_id: "engine-2", station_id: "station-engine-2" }, target: { ...makeAction().target, response_target_id: 2 } }),
        ]}
        locationName="Modiin"
      />,
    );

    expect(screen.getByRole("heading", { name: "Active fire - Modiin (1)" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Active fire - Modiin (2)" })).toBeInTheDocument();
  });

  it("colors the priority tier: red for high, orange for medium, green for lower", () => {
    const at = (id: number, score: number, resource: string) =>
      makeAction({
        resource: { ...makeAction().resource, resource_id: resource },
        target: { ...makeAction().target, response_target_id: id, priority_score: score },
      });
    render(<ResponseActions actions={[at(1, 100, "a"), at(2, 50, "b"), at(3, 10, "c")]} />);

    expect(screen.getByText("High Priority")).toHaveClass("response-action-group__priority--high");
    expect(screen.getByText("Medium Priority")).toHaveClass("response-action-group__priority--medium");
    expect(screen.getByText("Lower Priority")).toHaveClass("response-action-group__priority--low");
  });

  it("uses the whole header row as the expand toggle, with no separate arrow control", async () => {
    const user = userEvent.setup();
    render(<ResponseActions actions={[makeAction()]} onSelectAction={() => {}} />);

    const header = screen.getByRole("button", { name: /engine-1/ });
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("▸")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button")).toHaveLength(2); // header toggle + Highlight on map
    expect(screen.queryByText("Dispatch & Navigation")).not.toBeInTheDocument();

    await user.click(header);
    expect(header).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Dispatch & Navigation")).toBeInTheDocument();

    await user.click(header);
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Dispatch & Navigation")).not.toBeInTheDocument();
  });

  it("shows Waze and Google Maps deep links to the target, opening in a new tab", async () => {
    const user = userEvent.setup();
    render(<ResponseActions actions={[makeAction()]} />);

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    expect(screen.getByText("Launch real-time navigation:")).toBeInTheDocument();
    const waze = screen.getByRole("link", { name: "Navigate with Waze" });
    expect(waze).toHaveAttribute("href", "https://www.waze.com/live-map/directions?from=ll.32,35&to=ll.32.1,35.1");
    expect(waze).toHaveAttribute("target", "_blank");
    expect(waze).toHaveAttribute("rel", expect.stringContaining("noopener"));

    const google = screen.getByRole("link", { name: "Open Google Maps" });
    // Google Maps shows the station -> target path, not the dispatcher's own location.
    expect(google).toHaveAttribute(
      "href",
      "https://www.google.com/maps/dir/?api=1&origin=32,35&destination=32.1,35.1",
    );
    expect(google).toHaveAttribute("target", "_blank");
  });

  it("makes no network request and renders no turn-by-turn text when expanded", async () => {
    const user = userEvent.setup();
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    render(<ResponseActions actions={[makeAction()]} />);

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(screen.queryByRole("list", { name: "Turn-by-turn directions" })).not.toBeInTheDocument();
    fetchSpy.mockRestore();
  });

  it("shows a note instead of links when the target has no coordinates", async () => {
    const user = userEvent.setup();
    render(
      <ResponseActions actions={[makeAction({ target: { ...makeAction().target, latitude: null, longitude: null } })]} />,
    );

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    expect(screen.getByText(/Navigation unavailable/)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Navigate with Waze" })).not.toBeInTheDocument();
  });

  it("never shows the raw priority score; shows a semantic priority tier instead", () => {
    render(
      <ResponseActions
        actions={[
          makeAction({ target: { ...makeAction().target, response_target_id: 1, priority_score: 179.5600000007 } }),
        ]}
      />,
    );

    expect(screen.getByText(/High Priority/)).toBeInTheDocument();
    expect(screen.queryByText(/179/)).not.toBeInTheDocument();
    expect(screen.queryByText(/priority \d/i)).not.toBeInTheDocument();
  });

  it("tiers priority relative to the plan's highest-scoring target", () => {
    const at = (id: number, score: number, resource: string) =>
      makeAction({
        resource: { ...makeAction().resource, resource_id: resource },
        target: { ...makeAction().target, response_target_id: id, priority_score: score },
      });
    render(<ResponseActions actions={[at(1, 100, "a"), at(2, 50, "b"), at(3, 10, "c")]} />);

    expect(screen.getByText(/^High Priority/)).toBeInTheDocument();
    expect(screen.getByText(/^Medium Priority/)).toBeInTheDocument();
    expect(screen.getByText(/^Lower Priority/)).toBeInTheDocument();
  });

  it("titles each group with the target id and type when no location name exists", () => {
    render(<ResponseActions actions={[makeAction()]} />);

    expect(screen.getByRole("heading", { name: "Target #1 - Active fire" })).toBeInTheDocument();
  });

  it("omits the priority tier when the target has no score", () => {
    render(<ResponseActions actions={[makeAction({ target: { ...makeAction().target, priority_score: null } })]} />);

    expect(screen.queryByText(/Priority/)).not.toBeInTheDocument();
    expect(screen.getByText("1 truck")).toBeInTheDocument();
  });


  it("prefers a target's own reverse-geocoded place over the parent event's location", () => {
    render(<ResponseActions actions={[makeAction()]} locationName="Modiin" targetLocations={{ 1: "Haifa" }} />);

    expect(screen.getByRole("heading", { name: "Active fire - Haifa" })).toBeInTheDocument();
  });

  it("falls back to the event location while a target's place is unresolved or failed (null)", () => {
    const { rerender } = render(<ResponseActions actions={[makeAction()]} locationName="Modiin" targetLocations={{}} />);
    expect(screen.getByRole("heading", { name: "Active fire - Modiin" })).toBeInTheDocument();

    rerender(<ResponseActions actions={[makeAction()]} locationName="Modiin" targetLocations={{ 1: null }} />);
    expect(screen.getByRole("heading", { name: "Active fire - Modiin" })).toBeInTheDocument();
  });

  it("only shows the raw target id when no place name exists anywhere", () => {
    render(<ResponseActions actions={[makeAction()]} locationName={null} targetLocations={{ 1: null }} />);

    expect(screen.getByRole("heading", { name: "Target #1 - Active fire" })).toBeInTheDocument();
  });

  it("gives Waze an explicit station start and target end, latitude first", async () => {
    const user = userEvent.setup();
    render(<ResponseActions actions={[makeAction()]} />);

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    const href = screen.getByRole("link", { name: "Navigate with Waze" }).getAttribute("href");
    expect(href).toBe("https://www.waze.com/live-map/directions?from=ll.32,35&to=ll.32.1,35.1");
    expect(href).not.toContain("navigate=yes");
  });

  it("omits the Waze start when the station's coordinates are unknown", async () => {
    const user = userEvent.setup();
    render(<ResponseActions actions={[makeAction({ resource: { ...makeAction().resource, origin: null } })]} />);

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    expect(screen.getByRole("link", { name: "Navigate with Waze" })).toHaveAttribute(
      "href",
      "https://www.waze.com/live-map/directions?to=ll.32.1,35.1",
    );
  });

  it("omits the Google Maps origin when the station's coordinates are unknown", async () => {
    const user = userEvent.setup();
    render(<ResponseActions actions={[makeAction({ resource: { ...makeAction().resource, origin: null } })]} />);

    await user.click(screen.getByRole("button", { name: /engine-1/ }));

    expect(screen.getByRole("link", { name: "Open Google Maps" })).toHaveAttribute(
      "href",
      "https://www.google.com/maps/dir/?api=1&destination=32.1,35.1",
    );
  });
});
