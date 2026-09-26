import { readFileSync } from "node:fs";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OperationsActivityFeed } from "./OperationsActivityFeed";
import { NEW_ACTIVITY_HIGHLIGHT_MS } from "./activityFeedPresentation";
import type { OperationsActivityFeedItem } from "../../types/operationsOverview";

function makeItem(
  id: number,
  activity_type: OperationsActivityFeedItem["activity_type"],
  overrides: Partial<OperationsActivityFeedItem> = {},
): OperationsActivityFeedItem {
  const base = {
    activity_id: `${activity_type}:${id}`,
    entity_id: id,
    occurred_at: `2026-09-20T09:${String(id).padStart(2, "0")}:00Z`,
    available_at: `2026-09-20T09:${String(id).padStart(2, "0")}:00Z`,
    title: `Item ${id}`,
    location: null,
  };
  switch (activity_type) {
    case "fire_danger":
      return {
        ...base,
        activity_type,
        preview: { area_name: "Northern District", status: "valid", level: "high", score: 72.5 },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "satellite_hotspot":
      return {
        ...base,
        activity_type,
        preview: { confidence: "n", frp: 12.3, location_name: null },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "news_report":
      return {
        ...base,
        activity_type,
        preview: { source: "Ynet", headline: "Wildfire spreads near Haifa" },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "fire_event":
      return {
        ...base,
        activity_type,
        preview: { status: "confirmed", confidence: 0.87 },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "fire_severity":
      return {
        ...base,
        activity_type,
        preview: { fire_event_id: 4, level: "critical", score: 91.2 },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "global_planning_run":
      return {
        ...base,
        activity_type,
        preview: { status: "completed", fire_event_count: 2 },
        ...overrides,
      } as OperationsActivityFeedItem;
    case "weather_conditions":
      return {
        ...base,
        activity_type,
        preview: {
          area_name: "Northern District",
          fire_danger_level: "very_high",
          fire_danger_assessment_id: 1,
          temperature_c: 34,
          relative_humidity_pct: 19,
          wind_speed_kmh: 28,
          wind_gust_kmh: null,
        },
        ...overrides,
      } as OperationsActivityFeedItem;
  }
}

const FIRE_DANGER_ITEM = makeItem(1, "fire_danger");
const SATELLITE_ITEM = makeItem(2, "satellite_hotspot");
const NEWS_ITEM = makeItem(3, "news_report");
const FIRE_EVENT_ITEM = makeItem(4, "fire_event");
const SEVERITY_ITEM = makeItem(5, "fire_severity");
const GLOBAL_PLANNING_ITEM = makeItem(6, "global_planning_run");
const WEATHER_ITEM = makeItem(7, "weather_conditions");

/** Includes one of each excluded type - none must ever render as a row. */
const ALL_ITEMS = [
  GLOBAL_PLANNING_ITEM,
  SEVERITY_ITEM,
  FIRE_EVENT_ITEM,
  NEWS_ITEM,
  SATELLITE_ITEM,
  FIRE_DANGER_ITEM,
  WEATHER_ITEM,
];
/** The same set, in the same order, with every ineligible item removed - what should actually render (final semantics: news_report/satellite_hotspot/fire_danger/weather_conditions only). */
const ELIGIBLE_ITEMS = [NEWS_ITEM, SATELLITE_ITEM, FIRE_DANGER_ITEM, WEATHER_ITEM];

function rowButtons() {
  return screen.getAllByRole("button").filter((button) => button.className.includes("activity-row"));
}

describe("OperationsActivityFeed final visible-type semantics (news_report/satellite_hotspot/fire_danger/weather_conditions only)", () => {
  it("shows an empty-state message when there are no items", () => {
    render(<OperationsActivityFeed items={[]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("No recent operational activity")).toBeInTheDocument();
  });

  it("shows an empty-state message when only excluded types exist", () => {
    render(
      <OperationsActivityFeed
        items={[GLOBAL_PLANNING_ITEM, SEVERITY_ITEM, FIRE_EVENT_ITEM]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    expect(screen.getByText("No recent operational activity")).toBeInTheDocument();
  });

  it("never renders a global_planning_run item as a feed row", () => {
    render(<OperationsActivityFeed items={ALL_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText(/global response plan/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/2 active fires/)).not.toBeInTheDocument();
  });

  it("never renders a fire_severity item as a feed row (already shown on Active Fires cards)", () => {
    render(<OperationsActivityFeed items={ALL_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText(/critical severity/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/severity not available/i)).not.toBeInTheDocument();
  });

  it("never renders a fire_event lifecycle row (already shown on map/Active Fires/Event Details)", () => {
    render(<OperationsActivityFeed items={ALL_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText(/confirmed$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/suspected$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Event #4/)).not.toBeInTheDocument();
  });

  it("renders one row per ELIGIBLE item (not a card grid), in exact server order, never resorted", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const rows = rowButtons();
    expect(rows).toHaveLength(4);
    expect(rows[0]).toHaveTextContent("Wildfire spreads near Haifa");
    expect(rows[2]).toHaveTextContent("Northern District");
  });

  it("lays the feed out as a true vertical list, never a multi-column grid", () => {
    // jsdom does not apply imported CSS, so this is a source-level guard.
    const css = readFileSync("src/components/dashboard/OperationsActivityFeed.css", "utf-8");
    const listRule = css.match(/\.operations-activity-feed__list\s*\{[^}]*\}/)?.[0] ?? "";

    expect(listRule).not.toMatch(/display:\s*grid/);
    expect(listRule).not.toMatch(/grid-template-columns/);
  });

  it("never renders a System Update pseudo-category", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText(/system update/i)).not.toBeInTheDocument();
  });

  it("renders concise, human-readable report lines for the four remaining types", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("Northern District - High fire danger")).toBeInTheDocument();
    expect(screen.getByText("New satellite hotspot detected")).toBeInTheDocument();
    expect(screen.getByText(/Nominal confidence/)).toBeInTheDocument();
    expect(screen.getByText("Wildfire spreads near Haifa")).toBeInTheDocument();
    expect(screen.getByText("Weather conditions - Northern District")).toBeInTheDocument();
    expect(screen.getByText(/34°C · 19% humidity · 28 km\/h wind/)).toBeInTheDocument();
  });

  it.each(["low", "moderate", "high", "very_high", "extreme"] as const)(
    "uses fully neutral 'Weather conditions' wording for every Fire Danger level, never implying WeatherAgent classified danger (level=%s)",
    (level) => {
      const item = makeItem(20, "weather_conditions", {
        preview: {
          area_name: "Golan Heights Demo Area",
          fire_danger_level: level,
          fire_danger_assessment_id: 1,
          temperature_c: 22,
          relative_humidity_pct: 65,
          wind_speed_kmh: 8,
          wind_gust_kmh: null,
        },
      });
      render(<OperationsActivityFeed items={[item]} selectedActivityId={null} onSelectItem={vi.fn()} />);

      expect(screen.getByText("Weather conditions - Golan Heights Demo Area")).toBeInTheDocument();
      expect(screen.queryByText(/Risk-elevating/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Weather update/)).not.toBeInTheDocument();
    },
  );

  it("never shows a raw score or a raw one-letter confidence code", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText(/score/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/72\.5/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Confidence: n\b/)).not.toBeInTheDocument();
  });

  it("never shows a textual type abbreviation badge (SEV/SAT/NEWS/FE/GP)", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    for (const forbidden of ["SEV", "SAT", "NEWS", "FE", "GP"]) {
      expect(screen.queryByText(forbidden, { selector: "span:not(.visually-hidden)" })).not.toBeInTheDocument();
    }
  });

  it("shows the real persisted area name for a satellite hotspot inside a known Fire Danger area", () => {
    const item = makeItem(10, "satellite_hotspot", {
      preview: { confidence: "h", frp: 40, location_name: "Carmel Demo Area" },
    });
    render(<OperationsActivityFeed items={[item]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("New satellite hotspot detected - Carmel Demo Area")).toBeInTheDocument();
    expect(screen.getByText(/High confidence/)).toBeInTheDocument();
  });

  it("renders a safe fallback with no fabricated location when location_name is null", () => {
    const item = makeItem(10, "satellite_hotspot", { preview: { confidence: "l", frp: 10, location_name: null } });
    render(<OperationsActivityFeed items={[item]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("New satellite hotspot detected")).toBeInTheDocument();
    expect(screen.getByText(/Low confidence/)).toBeInTheDocument();
    for (const fake of ["Carmel", "Golan", "Jerusalem Forest", "Galilee", "Judean Hills"]) {
      expect(screen.queryByText(new RegExp(fake))).not.toBeInTheDocument();
    }
  });

  it("never hardcodes a coordinate/area-name mapping in the frontend (source-level guard)", () => {
    const source = readFileSync("src/components/dashboard/OperationsActivityItem.tsx", "utf-8");
    for (const forbidden of ["Carmel", "Golan", "Jerusalem Forest", "Galilee", "Judean Hills"]) {
      expect(source).not.toContain(forbidden);
    }
  });

  it("selects a row on click and reports it to the parent", async () => {
    const user = userEvent.setup();
    const onSelectItem = vi.fn();
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={onSelectItem} />);

    await user.click(screen.getByText("Northern District - High fire danger"));

    expect(onSelectItem).toHaveBeenCalledWith(FIRE_DANGER_ITEM);
  });

  it("marks the selected row for assistive tech and keyboard users", () => {
    render(
      <OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={FIRE_DANGER_ITEM.activity_id} onSelectItem={vi.fn()} />,
    );

    const button = screen.getByText("Northern District - High fire danger").closest("button");
    expect(button).toHaveAttribute("aria-pressed", "true");
  });

  it("keeps the readable Hebrew headline (dir=auto, never romanized) when backend translation failed", () => {
    // A backend LLM translation failure at ingestion is best-effort and
    // leaves the original Hebrew persisted (see news_client.py's
    // "TRANSLATION FALLBACK TRIGGERED"). Readable Hebrew is shown as-is -
    // never letter-by-letter romanization - and dir="auto" lays it out RTL.
    const hebrewItem = makeItem(7, "news_report", {
      preview: { source: "Ynet", headline: "האש ממשיכה להיראות באזור הכרמל" },
    });
    render(<OperationsActivityFeed items={[hebrewItem]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const main = screen.getByText("האש ממשיכה להיראות באזור הכרמל", { exact: false });
    expect(main).toHaveAttribute("dir", "auto");
    expect(screen.queryByText(/HaEsh|Mmshykh/)).not.toBeInTheDocument();
    expect(document.querySelector(".operations-activity-feed")).not.toHaveAttribute("dir", "rtl");
  });

  it("shows the HH:MM:SS timestamp on each row", () => {
    render(<OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const time = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time");
    expect(time).toHaveTextContent(/^\d{2}:\d{2}:\d{2}$/);
  });

  it("Task 4, Part J: a real news headline containing an em-dash is rendered byte-for-byte unchanged - only OUR generated strings get ASCII hyphens", () => {
    const item = makeItem(8, "news_report", {
      preview: { source: "Ynet", headline: "Fire spreads — residents evacuated" },
    });
    render(<OperationsActivityFeed items={[item]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("Fire spreads — residents evacuated")).toBeInTheDocument();
  });

  it("Task 4, Part F: the row timestamp is available_at, not occurred_at, when the two differ", () => {
    const occurredAt = "2026-09-20T09:37:12Z";
    const availableAt = "2026-09-20T09:40:05Z";
    const item = makeItem(9, "fire_danger", { occurred_at: occurredAt, available_at: availableAt });
    render(<OperationsActivityFeed items={[item]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    // Rendered in browser-local time (see ActivityTimestamp) - compare
    // against the same formatter rather than a hardcoded UTC string, so
    // this assertion is not tied to the test runner's local timezone.
    const formatter = new Intl.DateTimeFormat("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    const time = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time");
    expect(time).toHaveTextContent(formatter.format(new Date(availableAt)));
    expect(time).not.toHaveTextContent(formatter.format(new Date(occurredAt)));
  });
});

describe("OperationsActivityFeed visible-count windowing and Load more (eligible = news_report/satellite_hotspot/fire_danger/weather_conditions)", () => {
  function makeManyEligibleItems(count: number): OperationsActivityFeedItem[] {
    return Array.from({ length: count }, (_, index) => makeItem(index + 1, "fire_danger"));
  }

  it("shows exactly 5 eligible items initially when 5 or more exist", () => {
    render(<OperationsActivityFeed items={makeManyEligibleItems(12)} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(rowButtons()).toHaveLength(5);
  });

  it("does not count global_planning_run, fire_severity, or fire_event items toward the initial five", () => {
    const items = [
      makeItem(100, "global_planning_run"),
      makeItem(101, "fire_severity"),
      makeItem(104, "fire_event"),
      ...makeManyEligibleItems(5),
      makeItem(102, "global_planning_run"),
      makeItem(103, "fire_severity"),
      makeItem(105, "fire_event"),
    ];

    render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(rowButtons()).toHaveLength(5);
    expect(screen.getByText("Showing 5 of 5")).toBeInTheDocument();
  });

  it("shows all eligible items when fewer than 5 exist", () => {
    render(<OperationsActivityFeed items={makeManyEligibleItems(3)} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(rowButtons()).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });

  it("Load more reveals the next 5 eligible items", async () => {
    const user = userEvent.setup();
    render(<OperationsActivityFeed items={makeManyEligibleItems(12)} selectedActivityId={null} onSelectItem={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Load more" }));

    expect(rowButtons()).toHaveLength(10);
  });

  it("hides Load more once all currently available eligible items are visible", async () => {
    const user = userEvent.setup();
    render(<OperationsActivityFeed items={makeManyEligibleItems(8)} selectedActivityId={null} onSelectItem={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "Load more" }));

    expect(rowButtons()).toHaveLength(8);
    expect(screen.queryByRole("button", { name: "Load more" })).not.toBeInTheDocument();
  });

  it("shows Showing X of Y using the eligible count, excluding global_planning_run/fire_severity/fire_event", () => {
    const items = [
      ...makeManyEligibleItems(25),
      ...Array.from({ length: 2 }, (_, i) => makeItem(200 + i, "global_planning_run")),
      ...Array.from({ length: 2 }, (_, i) => makeItem(300 + i, "fire_severity")),
      ...Array.from({ length: 1 }, (_, i) => makeItem(400 + i, "fire_event")),
    ];

    render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.getByText("Showing 5 of 25")).toBeInTheDocument();
    expect(screen.queryByText(/of 30/)).not.toBeInTheDocument();
  });

  it("keeps the visible count at 5 across a poll that does not add items", () => {
    const items = makeManyEligibleItems(9);
    const { rerender } = render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);
    expect(rowButtons()).toHaveLength(5);

    rerender(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);
    expect(rowButtons()).toHaveLength(5);
  });

  it("keeps the visible count at 10 across a poll after Load more was clicked", async () => {
    const user = userEvent.setup();
    const items = makeManyEligibleItems(9);
    const { rerender } = render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Load more" }));
    expect(rowButtons()).toHaveLength(9);

    const grown = [{ ...makeItem(999, "fire_danger") }, ...items];
    rerender(<OperationsActivityFeed items={grown} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(rowButtons()).toHaveLength(10);
  });

  it("naturally pushes the previous fifth eligible item out of view when a new eligible item arrives at the top", () => {
    const items = makeManyEligibleItems(5);
    const { rerender } = render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);
    const initialRows = rowButtons();
    expect(initialRows).toHaveLength(5);
    const previousFifthRowText = initialRows[4].textContent;

    const newItem = makeItem(99, "news_report");
    rerender(<OperationsActivityFeed items={[newItem, ...items]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const updatedRows = rowButtons();
    expect(updatedRows).toHaveLength(5);
    expect(updatedRows[0]).toHaveTextContent("Wildfire spreads near Haifa");
    expect(updatedRows.some((row) => row.textContent === previousFifthRowText)).toBe(false);
  });

  it("a newly-arriving global_planning_run, fire_severity, or fire_event item never occupies a visible slot or affects the count", () => {
    const items = makeManyEligibleItems(5);
    const { rerender } = render(<OperationsActivityFeed items={items} selectedActivityId={null} onSelectItem={vi.fn()} />);

    rerender(
      <OperationsActivityFeed
        items={[makeItem(999, "global_planning_run"), makeItem(998, "fire_severity"), makeItem(997, "fire_event"), ...items]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    expect(rowButtons()).toHaveLength(5);
    expect(screen.getByText("Showing 5 of 5")).toBeInTheDocument();
  });
});

describe("OperationsActivityFeed newly-arrived highlighting", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not mark any activity as new on initial page load", () => {
    render(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
  });

  it("marks a newly-appearing satellite_hotspot as newly arrived, at the top of the list", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    expect(screen.queryByText("NEW")).not.toBeInTheDocument();

    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    const rows = rowButtons();
    expect(within(rows[0]).getByText("NEW")).toBeInTheDocument();
    expect(within(rows[1]).queryByText("NEW")).not.toBeInTheDocument();
  });

  it("marks a newly-appearing news_report as newly arrived", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    rerender(<OperationsActivityFeed items={[NEWS_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const rows = rowButtons();
    expect(within(rows[0]).getByText("NEW")).toBeInTheDocument();
  });

  it("keeps the real backend item (same activity_id, same content) when marking it new, with no duplicate row", () => {
    const onSelectItem = vi.fn();
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={onSelectItem} />,
    );

    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={onSelectItem} />,
    );

    expect(screen.getAllByText("New satellite hotspot detected")).toHaveLength(1);
    rowButtons()[0].click();
    expect(onSelectItem).toHaveBeenCalledWith(SATELLITE_ITEM);
  });

  it("clears the New indication after the configured presentation interval", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    expect(screen.getByText("NEW")).toBeInTheDocument();

    vi.advanceTimersByTime(NEW_ACTIVITY_HIGHLIGHT_MS + 1);
    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
    expect(screen.getByText("New satellite hotspot detected")).toBeInTheDocument();
  });

  it("does not re-mark the same activity as new on a repeated identical snapshot", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    vi.advanceTimersByTime(NEW_ACTIVITY_HIGHLIGHT_MS + 1);
    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    expect(screen.queryByText("NEW")).not.toBeInTheDocument();

    rerender(
      <OperationsActivityFeed items={[SATELLITE_ITEM, FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
  });

  it("marks multiple simultaneously-arriving eligible items as new", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    rerender(
      <OperationsActivityFeed
        items={[NEWS_ITEM, SATELLITE_ITEM, FIRE_DANGER_ITEM]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    expect(screen.getAllByText("NEW")).toHaveLength(2);
  });

  it("never marks a newly-arriving global_planning_run item as new (it never renders at all)", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    rerender(
      <OperationsActivityFeed
        items={[GLOBAL_PLANNING_ITEM, FIRE_DANGER_ITEM]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
  });

  it("a new fire_severity item never pushes a visible row down, never counts as visible, and never shows NEW", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    const before = rowButtons().map((row) => row.textContent);

    rerender(
      <OperationsActivityFeed
        items={[makeItem(500, "fire_severity"), ...ELIGIBLE_ITEMS]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    const after = rowButtons();
    expect(after.map((row) => row.textContent)).toEqual(before);
    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
  });

  it("a new fire_event confirmation never creates a visible NEW feed row", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );
    const before = rowButtons().map((row) => row.textContent);

    rerender(
      <OperationsActivityFeed
        items={[makeItem(600, "fire_event", { preview: { status: "confirmed", confidence: 0.9 } }), ...ELIGIBLE_ITEMS]}
        selectedActivityId={null}
        onSelectItem={vi.fn()}
      />,
    );

    const after = rowButtons();
    expect(after.map((row) => row.textContent)).toEqual(before);
    expect(screen.queryByText("NEW")).not.toBeInTheDocument();
    expect(screen.queryByText(/confirmed$/)).not.toBeInTheDocument();
  });

  it("never reorders the feed because of new-item highlighting", () => {
    const { rerender } = render(
      <OperationsActivityFeed items={[FIRE_DANGER_ITEM]} selectedActivityId={null} onSelectItem={vi.fn()} />,
    );

    rerender(<OperationsActivityFeed items={ELIGIBLE_ITEMS} selectedActivityId={null} onSelectItem={vi.fn()} />);

    const rows = rowButtons();
    expect(rows[0]).toHaveTextContent("Wildfire spreads near Haifa");
    expect(rows[2]).toHaveTextContent("Northern District");
  });
});
