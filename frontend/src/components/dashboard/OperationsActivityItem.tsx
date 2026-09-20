import { FIRE_DANGER_PRESENTATION } from "../status/fireDangerPresentation";
import { SEVERITY_PRESENTATION, STATUS_PRESENTATION } from "../status/presentation";
import { SATELLITE_CONFIDENCE_LABEL } from "./activityFeedPresentation";
import { ActivityTimestamp } from "./ActivityTimestamp";
import type { OperationsActivityFeedItem } from "../../types/operationsOverview";
import "./OperationsActivityItem.css";

export interface OperationsActivityItemProps {
  item: OperationsActivityFeedItem;
  isSelected: boolean;
  /** True for a brief period right after this activity_id first appears in an overview snapshot - presentation only, never persisted. */
  isNew: boolean;
  onSelect: (item: OperationsActivityFeedItem) => void;
}

interface ReportLine {
  /** The actual event/report text - the dominant content of the row. */
  main: string;
  /** Only genuinely useful secondary metadata (e.g. a satellite confidence label, a news source) - never technical ids/scores. */
  secondary?: string;
}

/** A subtle color cue per type (a thin left accent bar - see .activity-row__accent), never a textual abbreviation badge (SEV/SAT/NEWS/FE). */
const TYPE_ACCENT_COLOR: Record<Exclude<OperationsActivityFeedItem["activity_type"], "global_planning_run">, string> = {
  fire_danger: "var(--color-warning)",
  satellite_hotspot: "var(--color-focus-ring)",
  news_report: "var(--color-text-muted)",
  fire_event: "var(--color-danger)",
  fire_severity: "var(--color-fire-danger-extreme)",
  weather_conditions: "var(--color-success)",
};

const TYPE_LABEL: Record<OperationsActivityFeedItem["activity_type"], string> = {
  fire_danger: "Fire Danger",
  satellite_hotspot: "Satellite Hotspot",
  news_report: "News Report",
  fire_event: "Fire Event",
  fire_severity: "Fire Severity",
  global_planning_run: "Global Planning",
  weather_conditions: "Weather Conditions",
};

/**
 * One Activity Feed row: a thin colored accent bar on the left (a subtle
 * type cue - never a textual abbreviation like SEV/SAT/NEWS/FE), one
 * human-readable report line in the middle (dir="auto" so a Hebrew news
 * headline renders correctly without forcing the whole row/feed RTL), and
 * a timestamp + NEW indicator on the right. A real `<button>` so the whole
 * row is keyboard-selectable. The type is still available to assistive
 * tech via a visually-hidden label - removing the visible badge should not
 * remove the information entirely for screen reader users.
 *
 * `global_planning_run` items never reach this component - see
 * OperationsActivityFeed's `isEligibleForFeed` filter (Global Planning is
 * now reached via the "View Response Plan" action next to Active Fires).
 */
export function OperationsActivityItem({ item, isSelected, isNew, onSelect }: OperationsActivityItemProps) {
  const classNames = ["activity-row"];
  if (isSelected) {
    classNames.push("activity-row--selected");
  }
  if (isNew) {
    classNames.push("activity-row--new");
  }
  const accentColor = item.activity_type === "global_planning_run" ? undefined : TYPE_ACCENT_COLOR[item.activity_type];
  const reportLine = buildReportLine(item);

  return (
    <li className="activity-row-wrapper">
      <button type="button" className={classNames.join(" ")} aria-pressed={isSelected} onClick={() => onSelect(item)}>
        <span className="activity-row__accent" style={{ background: accentColor }} aria-hidden="true" />
        <span className="visually-hidden">{TYPE_LABEL[item.activity_type]}: </span>
        <span className="activity-row__main" dir="auto">
          {reportLine.main}
          {reportLine.secondary ? <span className="activity-row__secondary"> · {reportLine.secondary}</span> : null}
        </span>
        <span className="activity-row__meta">
          <ActivityTimestamp value={item.available_at} />
          {isNew ? <span className="activity-row__new-badge">NEW</span> : null}
        </span>
      </button>
    </li>
  );
}

/**
 * One concise, human-readable sentence (+ optional subtle secondary detail)
 * per activity type - what happened, not raw debug output. Never shows a
 * numeric score/one-letter code; a known persisted categorical value (e.g.
 * FIRMS confidence) is shown via its canonical label, never a
 * newly-invented threshold. Full numeric detail remains available in the
 * A5 drawer (OperationsActivityDrawer).
 */
function buildReportLine(item: OperationsActivityFeedItem): ReportLine {
  switch (item.activity_type) {
    case "fire_danger": {
      const { area_name, status, level } = item.preview;
      return status === "valid" && level !== null
        ? { main: `${area_name} - ${FIRE_DANGER_PRESENTATION[level].label} fire danger` }
        : { main: `${area_name} - Insufficient data for a fire danger assessment` };
    }
    case "satellite_hotspot": {
      const { confidence, location_name } = item.preview;
      const confidenceLabel =
        confidence !== null ? (SATELLITE_CONFIDENCE_LABEL[confidence.toLowerCase()] ?? confidence) : null;
      return {
        main: location_name !== null ? `New satellite hotspot detected - ${location_name}` : "New satellite hotspot detected",
        secondary: confidenceLabel ? `${confidenceLabel} confidence` : undefined,
      };
    }
    case "news_report": {
      const { source, headline } = item.preview;
      return { main: headline, secondary: source };
    }
    case "fire_event": {
      const { status } = item.preview;
      return { main: `Event #${item.entity_id} ${STATUS_PRESENTATION[status].label.toLowerCase()}` };
    }
    case "fire_severity": {
      const { fire_event_id, level } = item.preview;
      return level !== null
        ? { main: `Event #${fire_event_id} - ${SEVERITY_PRESENTATION[level].label} severity` }
        : { main: `Event #${fire_event_id} - Severity not available` };
    }
    case "global_planning_run": {
      // Unreachable in practice - OperationsActivityFeed filters these out
      // before this component is ever rendered for one.
      return { main: item.title };
    }
    case "weather_conditions": {
      // Part J: deliberately neutral for every level - "Risk-elevating"
      // wording would imply WeatherAgent itself classified the weather as
      // dangerous. WeatherAgent only supplies observations; the danger
      // interpretation belongs solely to the following Fire Danger row
      // (which reads the actual persisted level). No frontend danger
      // classifier is introduced here.
      const { area_name, temperature_c, relative_humidity_pct, wind_speed_kmh, wind_gust_kmh } = item.preview;
      const windPart =
        wind_gust_kmh !== null
          ? `${Math.round(wind_speed_kmh)} km/h wind (gust ${Math.round(wind_gust_kmh)})`
          : `${Math.round(wind_speed_kmh)} km/h wind`;
      return {
        main: `Weather conditions - ${area_name}`,
        secondary: `${Math.round(temperature_c)}°C · ${Math.round(relative_humidity_pct)}% humidity · ${windPart}`,
      };
    }
  }
}
