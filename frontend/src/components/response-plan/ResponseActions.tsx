import { useState } from "react";
import { translateStationLabel } from "../map/stationTranslations";
import { getStationContact } from "./stationContact";
import type { ResponsePlanAction } from "../../types/responsePlan";
import { EmptyState } from "../feedback/EmptyState";
import { formatDurationSeconds, NOT_AVAILABLE_LABEL } from "./formatting";
import { ROUTE_STATUS_LABELS, TARGET_TYPE_LABELS } from "./presentation";
import { DispatchNavigation } from "./DispatchNavigation";
import { getResponseActionKey } from "./ResponseRouteLayerModel";
import "./ResponseActionCard.css";
import "./ResponseActions.css";
import "./ResponsePlanSummary.css";

export interface ResponseActionsProps {
  actions: ResponsePlanAction[];
  /** The currently selected action's key, or `null`/omitted when nothing is
   * selected. See `getResponseActionKey`. */
  selectedActionKey?: string | null;
  /** Called with an action's key when the user selects/deselects it in the
   * list. Omit to render a non-selectable list. */
  onSelectAction?: (actionKey: string) => void;
  /** The parent FireEvent's location name, used to title target groups (targets carry none). */
  locationName?: string | null;
  /** Per-target place names from reverse geocoding (target id -> name); preferred over `locationName`. */
  targetLocations?: Record<number, string | null>;
  /** Overrides the section heading's id (and its aria-labelledby). Needed
   * when this component renders more than once on the same page (e.g. once
   * per event in the Global Response Plan's event list) so every instance
   * gets a unique, valid DOM id - defaults to the original fixed id for the
   * common single-plan-per-page case. */
  headingId?: string;
}

const EMPTY_TITLE = "No response actions";
const EMPTY_MESSAGE = "This response plan has no assigned resources.";

interface TargetGroup {
  targetId: number;
  target: ResponsePlanAction["target"];
  entries: { action: ResponsePlanAction; position: number }[];
}

/**
 * Human-readable priority tier for a target, relative to the highest-scoring
 * target in this plan (the raw persisted score is an algorithmic value and is
 * never shown). Presentation bucketing only - nothing is recalculated or
 * reordered, and a missing score yields no label rather than a guess.
 */
type PriorityTier = { label: string; tone: "high" | "medium" | "low" };

function priorityTier(score: number | null, maxScore: number): PriorityTier | null {
  if (score === null || maxScore <= 0) {
    return null;
  }
  const ratio = score / maxScore;
  if (ratio >= 2 / 3) return { label: "High Priority", tone: "high" };
  if (ratio >= 1 / 3) return { label: "Medium Priority", tone: "medium" };
  return { label: "Lower Priority", tone: "low" };
}

/**
 * Group title: the target type plus the place it is in. The place is the
 * reverse-geocoded name for the target's coordinates, falling back to the
 * parent FireEvent's location name; only when neither exists does it fall
 * back to the raw target id. When several groups would share a title, a
 * running number keeps them distinguishable.
 */
function groupTitles(groups: TargetGroup[], places: (string | null)[]): string[] {
  const base = groups.map((group, index) => {
    const type = group.target.target_type !== null ? TARGET_TYPE_LABELS[group.target.target_type] : null;
    const place = places[index];
    if (place) {
      return type !== null ? `${type} - ${place}` : place;
    }
    return type !== null ? `Target #${group.targetId} - ${type}` : `Target #${group.targetId}`;
  });
  const seen = new Map<string, number>();
  return base.map((title) => {
    const total = base.filter((other) => other === title).length;
    const n = (seen.get(title) ?? 0) + 1;
    seen.set(title, n);
    return total > 1 ? `${title} (${n})` : title;
  });
}

/**
 * Groups actions by destination target, keeping backend order both between
 * groups (first appearance) and within each group. Pure presentation - no
 * sorting, merging, or dropping of any action.
 */
function groupByTarget(actions: ResponsePlanAction[]): TargetGroup[] {
  const groups = new Map<number, TargetGroup>();
  actions.forEach((action, position) => {
    const targetId = action.target.response_target_id;
    const existing = groups.get(targetId);
    if (existing) {
      existing.entries.push({ action, position });
    } else {
      groups.set(targetId, { targetId, target: action.target, entries: [{ action, position }] });
    }
  });
  return Array.from(groups.values());
}

/**
 * Within one target, merges the actions dispatched from the same station into
 * a single row (backend order kept, by first appearance). Actions whose
 * station is unknown are never merged. Presentation only: every action is
 * still present in exactly one row.
 */
function groupByStation(entries: TargetGroup["entries"]): TargetGroup["entries"][] {
  const rows = new Map<string, TargetGroup["entries"]>();
  entries.forEach((entry, index) => {
    const key = entry.action.resource.station_id ? `station:${entry.action.resource.station_id}` : `action:${index}`;
    const row = rows.get(key);
    if (row) {
      row.push(entry);
    } else {
      rows.set(key, [entry]);
    }
  });
  return Array.from(rows.values());
}

/** Single time when the trucks share an ETA, otherwise "min-max". */
function formatTravelTimes(actions: ResponsePlanAction[]): string {
  const etas = actions.map((a) => a.route.eta_seconds).filter((eta): eta is number => eta !== null);
  if (etas.length === 0) {
    return NOT_AVAILABLE_LABEL;
  }
  const min = Math.min(...etas);
  const max = Math.max(...etas);
  const low = formatDurationSeconds(min);
  const high = formatDurationSeconds(max);
  return low === high ? low : `${low} - ${high}`;
}

/**
 * Renders every persisted response action (`plan.actions`) grouped by its
 * destination target: one card per target, listing each origin station once
 * with all of its trucks (resource ids) and their route status and ETA. Never sorts by
 * ETA/priority, never drops an incomplete action (an unreachable/unmappable
 * action stays visible even though it has no drawable route), and never
 * fabricates one. Selection is UI-only highlight state.
 */
export function ResponseActions({
  actions,
  selectedActionKey = null,
  onSelectAction,
  locationName = null,
  targetLocations = {},
  headingId = "response-actions-heading",
}: ResponseActionsProps) {
  const groups = groupByTarget(actions);
  const titles = groupTitles(
    groups,
    groups.map((group) => targetLocations[group.targetId] ?? locationName),
  );
  const maxScore = Math.max(0, ...groups.map((group) => group.target.priority_score ?? 0));

  return (
    <section aria-labelledby={headingId} className="response-plan-summary__section">
      <h2 id={headingId} className="response-plan-summary__section-title">
        Response Actions
      </h2>
      {actions.length === 0 ? (
        <EmptyState title={EMPTY_TITLE} message={EMPTY_MESSAGE} />
      ) : (
        <div className="response-action-groups">
          {groups.map((group, groupIndex) => (
            <article key={group.targetId} className="response-action-group">
              <header className="response-action-group__header">
                <h3 className="response-action-group__title">{titles[groupIndex]}</h3>
                <span className="response-action-group__meta">
                  {(() => {
                    const tier = priorityTier(group.target.priority_score, maxScore);
                    return tier !== null ? (
                      <>
                        <span className={`response-action-group__priority response-action-group__priority--${tier.tone}`}>
                          {tier.label}
                        </span>
                        <span aria-hidden="true"> · </span>
                      </>
                    ) : null;
                  })()}
                  <span>{`${group.entries.length} ${group.entries.length === 1 ? "truck" : "trucks"}`}</span>
                </span>
              </header>
              <ul className="response-action-group__list">
                {groupByStation(group.entries).map((entries, rowIndex) => (
                  <ActionRow
                    key={getResponseActionKey(entries[0].action)}
                    entries={entries}
                    position={rowIndex}
                    selectedActionKey={selectedActionKey}
                    onSelect={onSelectAction}
                  />
                ))}
              </ul>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

interface ActionRowProps {
  /** Every action dispatched from this station to this target (at least one). */
  entries: TargetGroup["entries"];
  /** Zero-based row number within the target group. */
  position: number;
  selectedActionKey: string | null;
  onSelect?: (actionKey: string) => void;
}

function ActionRow({ entries, position, selectedActionKey, onSelect }: ActionRowProps) {
  const [expanded, setExpanded] = useState(false);
  const actions = entries.map((entry) => entry.action);
  const first = actions[0];
  const keys = actions.map(getResponseActionKey);
  const rowKey = keys[0];
  const isSelected = selectedActionKey !== null && keys.includes(selectedActionKey);
  const { resource, target } = first;
  const detailId = `route-detail-${rowKey}`;
  const otherStatuses = Array.from(
    new Set(actions.map((a) => a.route.status).filter((status) => status !== "reachable")),
  );

  return (
    <li className={isSelected ? "response-action-row response-action-row--selected" : "response-action-row"}>
      <div className="response-action-row__top">
        <button
          type="button"
          className="response-action-row__header"
          aria-expanded={expanded}
          aria-controls={detailId}
          onClick={() => setExpanded((current) => !current)}
        >
          <span className="response-action-row__index" aria-label={`Route ${position + 1}`}>
            {position + 1}
          </span>
          <span className="response-action-row__main">
            <span className="response-action-row__station">
              <span className="response-action-row__station-label">Dispatch Station: </span>
              <span className="response-action-row__station-name">
                {translateStationLabel(resource.station_name ?? resource.station_id)}
              </span>
            </span>
            <span className="response-action-row__trucks">
              {actions.map((action) => (
                <span key={action.resource.resource_id} className="response-action-row__truck">
                  {action.resource.resource_id}
                </span>
              ))}
            </span>
          </span>
          <span className="response-action-row__stats">
            {otherStatuses.map((status) => (
              <span key={status ?? "none"} className="response-action-row__chip">
                {status !== null ? ROUTE_STATUS_LABELS[status] : NOT_AVAILABLE_LABEL}
              </span>
            ))}
            <span className="response-action-row__chip">Travel time {formatTravelTimes(actions)}</span>
          </span>
        </button>
        {onSelect ? (
          <button
            type="button"
            className="response-action-card__select"
            aria-pressed={isSelected}
            onClick={() => onSelect(rowKey)}
          >
            {isSelected ? "Selected" : "Highlight on map"}
          </button>
        ) : null}
      </div>
      <StationPhone stationName={resource.station_name} />
      {expanded ? (
        <div id={detailId} className="response-action-row__detail">
          <DispatchNavigation
            origin={resource.origin}
            destination={
              target.latitude !== null && target.longitude !== null
                ? { latitude: target.latitude, longitude: target.longitude }
                : null
            }
          />
        </div>
      ) : null}
    </li>
  );
}

/** Tap-to-call link for the dispatching station; omitted entirely when the registry has no number for it. Sits outside the row's toggle button (no nested interactive elements). */
function StationPhone({ stationName }: { stationName: string | null }) {
  const contact = getStationContact(stationName);
  if (contact === null) {
    return null;
  }
  return (
    <a className="response-action-row__phone" href={contact.href} aria-label={`Call station ${contact.display}`}>
      <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden="true">
        <path d="M3.654 1.328a.678.678 0 0 0-1.015-.063L1.605 2.3c-.483.484-.661 1.169-.45 1.77a17.6 17.6 0 0 0 4.168 6.608 17.6 17.6 0 0 0 6.608 4.168c.601.211 1.286.033 1.77-.45l1.034-1.034a.678.678 0 0 0-.063-1.015l-2.307-1.794a.68.68 0 0 0-.58-.122l-2.19.547a1.75 1.75 0 0 1-1.657-.459L5.482 8.062a1.75 1.75 0 0 1-.46-1.657l.548-2.19a.68.68 0 0 0-.122-.58z" />
      </svg>
      {contact.display}
    </a>
  );
}
