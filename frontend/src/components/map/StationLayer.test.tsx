import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StationLayer, dynamicAvailableCount, dynamicUnavailableCount } from "./StationLayer";
import type { FireStation, StationSummary } from "../../types/eventDetails";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function makeStation(overrides: Partial<FireStation> = {}): FireStation {
  return {
    station_id: "S1",
    name: "Central Station",
    latitude: 32.0,
    longitude: 34.8,
    station_type: "urban",
    address: "1 Main St",
    ...overrides,
  };
}

function makeStationSummary(overrides: Partial<StationSummary> = {}): StationSummary {
  return {
    station_id: "S1",
    total_resources: 4,
    available: 2,
    assigned_status: 1,
    unavailable: 1,
    current_global_plan_allocations: [],
    ...overrides,
  };
}

describe("StationLayer", () => {
  it("renders nothing when there are no stations", () => {
    const { container } = render(<StationLayer stations={[]} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one marker per station at its coordinates", () => {
    render(
      <StationLayer
        stations={[makeStation({ station_id: "S1" }), makeStation({ station_id: "S2", latitude: 32.1 })]}
      />,
    );

    const markers = screen.getAllByTestId("marker");
    expect(markers).toHaveLength(2);
  });

  it("shows name, type, and address in the popup, omitting fields the backend sent as null", () => {
    render(<StationLayer stations={[makeStation({ station_type: null, address: null })]} />);

    expect(screen.getByText("Central Station")).toBeInTheDocument();
    expect(screen.queryByText("Type:")).not.toBeInTheDocument();
    expect(screen.queryByText("1 Main St")).not.toBeInTheDocument();
  });

  it("shows total/available/unavailable counts strictly derived from the matching station summary", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({ total_resources: 4, assigned_status: 1, unavailable: 1, current_global_plan_allocations: [] }),
        ]}
      />,
    );

    expect(screen.getByText("Total")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Available")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    // assigned_status(1) + unavailable(1) = 2 unavailable; Available = Total(4) - Allocated(0) - Unavailable(2) = 2.
    expect(container.querySelector(".station-popup__stat--available dd")).toHaveTextContent("2");
    expect(container.querySelector(".station-popup__stat--unavailable dd")).toHaveTextContent("2");
  });

  it("lists allocated resource ids from the current global plan", () => {
    render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            current_global_plan_allocations: [
              { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
            ],
          }),
        ]}
      />,
    );

    expect(screen.getByText("Allocated in current plan")).toBeInTheDocument();
    expect(screen.getByText("R1")).toBeInTheDocument();
    expect(screen.getByText("R2")).toBeInTheDocument();
  });

  it('shows the Hebrew "none allocated" text for allocations when the current plan allocates nothing at this station', () => {
    render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[makeStationSummary({ current_global_plan_allocations: [] })]}
      />,
    );

    expect(screen.getByText("No allocated trucks")).toBeInTheDocument();
  });

  it("omits the resource summary section when no matching station summary is given", () => {
    render(<StationLayer stations={[makeStation()]} />);

    expect(screen.queryByText("Allocated in current plan")).not.toBeInTheDocument();
    expect(screen.queryByText("Total")).not.toBeInTheDocument();
  });

  it("uses a location-pin glyph icon, not a plain dot", () => {
    render(<StationLayer stations={[makeStation()]} />);

    const icon = screen.getByTestId("marker-icon");
    expect(icon.querySelector("[data-station-icon] svg")).not.toBeNull();
  });

  it("keeps unallocated stations the default green, amber when only assigned trucks remain, grey when nothing is usable", () => {
    const { rerender } = render(
      <StationLayer stations={[makeStation()]} stationSummaries={[makeStationSummary({ available: 2 })]} />,
    );
    expect(screen.getByTestId("marker-icon").innerHTML).toContain("var(--color-success)");

    rerender(<StationLayer stations={[makeStation()]} />);
    expect(screen.getByTestId("marker-icon").innerHTML).toContain("var(--color-success)");

    rerender(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[makeStationSummary({ available: 0, assigned_status: 1, unavailable: 0, total_resources: 1 })]}
      />,
    );
    expect(screen.getByTestId("marker-icon").innerHTML).toContain("var(--color-warning)");

    rerender(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[makeStationSummary({ available: 0, assigned_status: 0, unavailable: 3, total_resources: 3 })]}
      />,
    );
    expect(screen.getByTestId("marker-icon").innerHTML).toContain("var(--color-text-muted)");
  });

  it("wraps the popup in an LTR container", () => {
    render(<StationLayer stations={[makeStation()]} stationSummaries={[makeStationSummary()]} />);

    const popup = screen.getByText("Central Station").closest(".station-popup") as HTMLElement;
    expect(popup).toHaveAttribute("dir", "ltr");
  });

  it("renders the station name as a header, the address as muted text, and the type as a badge", () => {
    render(<StationLayer stations={[makeStation({ station_type: "משנה", address: "Main St. 1" })]} />);

    expect(screen.getByRole("heading", { name: "Central Station" })).toBeInTheDocument();
    expect(screen.getByText("Main St. 1")).toHaveClass("station-popup__address");
    expect(screen.getByText("Type:")).toBeInTheDocument();
    expect(screen.getByText("Sub-station")).toHaveClass("station-popup__type-badge");
  });

  it("renders Hebrew station data in English, with no Hebrew left in the popup", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation({ name: "מפרץ חיפה", address: "החרש 1 חיפה", station_type: "משנה" })]}
        stationSummaries={[makeStationSummary({ current_global_plan_allocations: [] })]}
      />,
    );

    expect(screen.getByRole("heading", { name: "Haifa Bay Station" })).toBeInTheDocument();
    expect(screen.getByText("Haifa, HaCharash 1")).toHaveClass("station-popup__address");
    expect(screen.getByText("Sub-station")).toHaveClass("station-popup__type-badge");
    expect(container.textContent).not.toMatch(/[֐-׿]/);
  });

  it("lays the inventory out as four columns with semantic colors, and a divider before allocations", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[makeStationSummary({ total_resources: 4, available: 2, unavailable: 2 })]}
      />,
    );

    expect(container.querySelectorAll(".station-popup__stat")).toHaveLength(4);
    expect(container.querySelector(".station-popup__stat--available")).not.toBeNull();
    expect(container.querySelector(".station-popup__stat--unavailable")).not.toBeNull();
    const divider = container.querySelector("hr.station-popup__divider") as HTMLElement;
    const allocations = container.querySelector(".station-popup__allocations") as HTMLElement;
    expect(divider.compareDocumentPosition(allocations) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("mutes a zero unavailable count instead of coloring it as a problem", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[makeStationSummary({ assigned_status: 0, unavailable: 0 })]}
      />,
    );

    expect(container.querySelector(".station-popup__stat--unavailable")).toBeNull();
    expect(container.querySelector(".station-popup__stat--zero")).not.toBeNull();
  });

  it("shows the available count net of trucks allocated to the current plan", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 6,
            available: 5,
            unavailable: 1,
            current_global_plan_allocations: [
              { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
            ],
          }),
        ]}
      />,
    );

    const available = container.querySelector(".station-popup__stat--available dd") as HTMLElement;
    expect(available).toHaveTextContent("3");
    expect(container.querySelector(".station-popup__stat--available dt")).toHaveTextContent("Available");
  });

  it("mutes the available count to grey when the plan depletes the station's available trucks", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 2,
            assigned_status: 0,
            unavailable: 0,
            current_global_plan_allocations: [
              { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
            ],
          }),
        ]}
      />,
    );

    expect(container.querySelector(".station-popup__stat--available")).toBeNull();
    const zero = Array.from(container.querySelectorAll(".station-popup__stat--zero")).find((el) => el.textContent?.includes("Available"));
    expect(zero).toBeDefined();
    expect(zero).toHaveTextContent("0");
  });

  it("never shows a negative available count even when allocations exceed the station's total resources", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 2,
            current_global_plan_allocations: [
              { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R3", fire_event_id: 12, response_plan_id: 77 },
            ],
          }),
        ]}
      />,
    );

    expect(container.textContent).not.toContain("-");
    expect(
      dynamicAvailableCount(
        makeStationSummary({
          total_resources: 2,
          current_global_plan_allocations: [
            { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
            { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
            { resource_id: "R3", fire_event_id: 12, response_plan_id: 77 },
          ],
        }),
      ),
    ).toBe(0);
  });

  it("marks a station that contributes trucks to the current plan deep blue - never red, never inactive grey", () => {
    const { rerender } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            available: 1,
            current_global_plan_allocations: [{ resource_id: "R1", fire_event_id: 12, response_plan_id: 77 }],
          }),
        ]}
      />,
    );
    let html = screen.getByTestId("marker-icon").innerHTML;
    expect(html).toContain("#1e3a8a");
    expect(html).not.toContain("var(--color-danger)");

    // Fully allocated (0 available under the plan) stays blue, not grey.
    rerender(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            available: 1,
            unavailable: 0,
            total_resources: 1,
            current_global_plan_allocations: [{ resource_id: "R1", fire_event_id: 12, response_plan_id: 77 }],
          }),
        ]}
      />,
    );
    html = screen.getByTestId("marker-icon").innerHTML;
    expect(html).toContain("#1e3a8a");
    expect(html).not.toContain("var(--color-text-muted)");
  });

  it("shows Total, Available, Allocated and Unavailable in four columns whose counts balance", () => {
    const allocations = [
      { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
      { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
    ];
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 7,
            available: 5,
            assigned_status: 0,
            unavailable: 2,
            current_global_plan_allocations: allocations,
          }),
        ]}
      />,
    );

    const stats = Array.from(container.querySelectorAll(".station-popup__stat")).map((el) => [
      el.querySelector("dt")?.textContent,
      Number(el.querySelector("dd")?.textContent),
    ]);
    expect(stats).toEqual([
      ["Total", 7],
      ["Available", 3],
      ["Allocated", 2],
      ["Unavailable", 2],
    ]);
    const [, total] = stats[0] as [string, number];
    const sum = (stats[1][1] as number) + (stats[2][1] as number) + (stats[3][1] as number);
    expect(sum).toBe(total);
  });

  it("counts a truck assigned outside the current plan as unavailable, so Total still balances (Nesher-style bug)", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 2,
            available: 1,
            assigned_status: 1,
            unavailable: 0,
            current_global_plan_allocations: [],
          }),
        ]}
      />,
    );

    const stats = Array.from(container.querySelectorAll(".station-popup__stat")).map((el) => [
      el.querySelector("dt")?.textContent,
      Number(el.querySelector("dd")?.textContent),
    ]);
    expect(stats).toEqual([
      ["Total", 2],
      ["Available", 1],
      ["Allocated", 0],
      ["Unavailable", 1],
    ]);
  });

  it("never double-counts a truck as both Allocated and Unavailable when its DB status already flipped to assigned", () => {
    // Total 2, both trucks allocated to THIS event's plan; one of them already
    // shows as raw `assigned` in the DB (status updates can land before or
    // after a plan is generated). Before the strict Total-based formula this
    // produced Total 2 / Allocated 2 / Unavailable 1 (summing to 3).
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({
            total_resources: 2,
            assigned_status: 1,
            unavailable: 0,
            current_global_plan_allocations: [
              { resource_id: "R1", fire_event_id: 12, response_plan_id: 77 },
              { resource_id: "R2", fire_event_id: 12, response_plan_id: 77 },
            ],
          }),
        ]}
      />,
    );

    const stats = Array.from(container.querySelectorAll(".station-popup__stat")).map((el) => [
      el.querySelector("dt")?.textContent,
      Number(el.querySelector("dd")?.textContent),
    ]);
    expect(stats).toEqual([
      ["Total", 2],
      ["Available", 0],
      ["Allocated", 2],
      ["Unavailable", 0],
    ]);
    const sum = (stats[1][1] as number) + (stats[2][1] as number) + (stats[3][1] as number);
    expect(sum).toBe(stats[0][1]);
  });

  it("dynamicUnavailableCount folds assigned-elsewhere trucks into the unavailable bucket, defensively treating missing counts as zero", () => {
    expect(dynamicUnavailableCount(makeStationSummary({ unavailable: 1, assigned_status: 1 }))).toBe(2);
    expect(
      dynamicUnavailableCount({ ...makeStationSummary(), unavailable: null as unknown as number, assigned_status: 2 }),
    ).toBe(2);
  });

  it("styles the Allocated number with the operational orange used by the allocated truck pills", () => {
    const { container } = render(
      <StationLayer
        stations={[makeStation()]}
        stationSummaries={[
          makeStationSummary({ current_global_plan_allocations: [{ resource_id: "R1", fire_event_id: 12, response_plan_id: 77 }] }),
        ]}
      />,
    );

    expect(container.querySelector(".station-popup__stat--allocated dd")).toHaveTextContent("1");
    expect(container.querySelector(".station-popup__allocation")).not.toBeNull();
  });

  it("shows 0 allocated when the plan allocates nothing at this station", () => {
    const { container } = render(
      <StationLayer stations={[makeStation()]} stationSummaries={[makeStationSummary({ current_global_plan_allocations: [] })]} />,
    );

    expect(container.querySelector(".station-popup__stat--allocated dd")).toHaveTextContent("0");
  });
});
