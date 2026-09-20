import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { EventGroupList } from "./EventGroupList";

import type { ResponsePlanAction, ResponsePlanTarget } from "../../types/responsePlan";
import type { GlobalEventPlan } from "../../types/globalResponsePlan";

function makeAction(overrides: Partial<ResponsePlanAction> = {}): ResponsePlanAction {
  return {
    resource: {
      resource_id: "engine-1",
      station_id: "station-1",
      station_name: "Central Station",
      origin: { latitude: 32.0, longitude: 34.8 },
    },
    target: {
      response_target_id: 1,
      target_type: "active_fire",
      priority_score: 0.9,
      latitude: 32.7,
      longitude: 35.0,
    },
    route: {
      status: "reachable",
      eta_seconds: 120.0,
      distance_meters: 800.0,
      node_path: [1, 2],
      path_coordinates: null,
    },
    ...overrides,
  };
}

function makeUncoveredTarget(overrides: Partial<ResponsePlanTarget> = {}): ResponsePlanTarget {
  return {
    response_target_id: 5,
    target_type: "active_fire",
    priority_score: 0.3,
    latitude: 40.0,
    longitude: 41.0,
    ...overrides,
  };
}

function makeEvent(overrides: Partial<GlobalEventPlan> = {}): GlobalEventPlan {
  return {
    fire_event_id: 101,
    response_plan_id: 501,
    severity_level: "high",
    severity_score: 70.0,
    minimum_resources: 2,
    desired_resources: 4,
    assigned_resources: 3,
    coverage_score: 85.0,
    average_eta_seconds: 140.0,
    actions: [makeAction()],
    uncovered_targets: [],
    ...overrides,
  };
}

function renderList(
  events: GlobalEventPlan[],
  focusEventId: number | null = null,
  onFocusEvent = vi.fn(),
  eventLabels?: Record<number, string>,
) {
  return render(
    <MemoryRouter>
      <EventGroupList events={events} focusEventId={focusEventId} onFocusEvent={onFocusEvent} eventLabels={eventLabels} />
    </MemoryRouter>,
  );
}

describe("EventGroupList", () => {
  it("shows an empty state when there are no materialized events", () => {
    renderList([]);

    expect(screen.getByText("No materialized fire events")).toBeInTheDocument();
  });

  it("renders one group per event, in backend order", () => {
    renderList([makeEvent({ fire_event_id: 101 }), makeEvent({ fire_event_id: 202 })]);

    expect(screen.getByRole("link", { name: "Event #101" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Event #202" })).toBeInTheDocument();
  });

  it("links each group to its Event Details page", () => {
    renderList([makeEvent({ fire_event_id: 101 })]);

    expect(screen.getByRole("link", { name: "Event #101" })).toHaveAttribute("href", "/events/101");
  });

  it("shows severity, coverage, and average ETA for each group", () => {
    renderList([makeEvent()]);

    expect(screen.queryByText(/Score/)).not.toBeInTheDocument();
    expect(screen.getByText("85.0%")).toBeInTheDocument();
    expect(screen.getByText("2m 20s")).toBeInTheDocument();
  });

  it("shows minimum/desired/assigned resource counts", () => {
    renderList([makeEvent({ minimum_resources: 2, desired_resources: 4, assigned_resources: 3 })]);

    const line = screen.getByLabelText("Resources");
    expect(line).toHaveTextContent("Minimum 2");
    expect(line).toHaveTextContent("Desired 4");
    expect(line).toHaveTextContent("Assigned 3");
    expect(screen.queryByText("Minimum Resources")).not.toBeInTheDocument();
  });

  it("renders an action card for each assigned action", () => {
    renderList([makeEvent({ actions: [makeAction({ resource: { ...makeAction().resource, resource_id: "engine-9" } })] })]);

    expect(screen.getByText("(engine-9)")).toBeInTheDocument();
  });

  it("titles a group with its English label instead of the raw id", () => {
    renderList([makeEvent({ fire_event_id: 101 })], null, vi.fn(), { 101: "Haifa Subdistrict" });

    expect(screen.getByRole("link", { name: "Haifa Subdistrict" })).toHaveAttribute("href", "/events/101");
    expect(screen.queryByRole("link", { name: "Event #101" })).not.toBeInTheDocument();
  });

  it("returns the focus button to its default label when the group is not focused", () => {
    const { rerender } = renderList([makeEvent({ fire_event_id: 101 })], 101);
    expect(screen.getByRole("button", { name: "Focused" })).toHaveAttribute("aria-pressed", "true");

    rerender(
      <MemoryRouter>
        <EventGroupList events={[makeEvent({ fire_event_id: 101 })]} focusEventId={null} onFocusEvent={vi.fn()} />
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: "Focus on map" })).toHaveAttribute("aria-pressed", "false");
  });

  it("shows an empty state within the group when there are no assigned actions", () => {
    renderList([makeEvent({ actions: [] })]);

    expect(screen.getByText("No assigned resources")).toBeInTheDocument();
  });

  it("renders uncovered targets via the shared UncoveredTargets component", () => {
    renderList([makeEvent({ uncovered_targets: [makeUncoveredTarget()] })]);

    expect(screen.getByText("Uncovered Targets")).toBeInTheDocument();
    expect(screen.getByText("#5")).toBeInTheDocument();
  });

  it("hides the Uncovered Targets section entirely when there are none", () => {
    renderList([makeEvent({ uncovered_targets: [] })]);

    expect(screen.queryByText("Uncovered Targets")).not.toBeInTheDocument();
    expect(screen.queryByText("All response targets are covered by this plan.")).not.toBeInTheDocument();
  });

  it("marks the focused event's group distinctly and shows Focused on its button", () => {
    renderList([makeEvent({ fire_event_id: 101 }), makeEvent({ fire_event_id: 202 })], 101);

    const focusedHeading = screen.getByRole("link", { name: "Event #101" });
    const focusedGroup = focusedHeading.closest("article") as HTMLElement;
    expect(within(focusedGroup).getByRole("button", { name: "Focused" })).toBeInTheDocument();

    const otherHeading = screen.getByRole("link", { name: "Event #202" });
    const otherGroup = otherHeading.closest("article") as HTMLElement;
    expect(within(otherGroup).getByRole("button", { name: "Focus on map" })).toBeInTheDocument();
  });

  it("calls onFocusEvent with the group's fire_event_id when its focus button is clicked", async () => {
    const user = userEvent.setup();
    const onFocusEvent = vi.fn();
    renderList([makeEvent({ fire_event_id: 101 })], null, onFocusEvent);

    await user.click(screen.getByRole("button", { name: "Focus on map" }));

    expect(onFocusEvent).toHaveBeenCalledWith(101);
  });

  it("shows Not available for null severity/coverage/ETA fields, never a fabricated value", () => {
    renderList([
      makeEvent({ severity_level: null, severity_score: null, coverage_score: null, average_eta_seconds: null }),
    ]);

    expect(screen.getAllByText("Not available").length).toBeGreaterThanOrEqual(2);
  });
});
