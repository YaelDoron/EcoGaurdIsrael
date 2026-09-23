import { Link } from "react-router-dom";
import { EmptyState } from "../feedback/EmptyState";
import { ResponseActions } from "../response-plan/ResponseActions";
import { UncoveredTargets } from "../response-plan/UncoveredTargets";
import { formatCoveragePercentage, formatDurationSeconds, NOT_AVAILABLE_LABEL } from "../response-plan/formatting";
import { SeverityBadge } from "../status/SeverityBadge";
import type { GlobalEventPlan } from "../../types/globalResponsePlan";
import "./EventGroupList.css";

export interface EventGroupListProps {
  events: GlobalEventPlan[];
  /** The currently focused FireEvent id (from the page's `focusEventId`
   * query param), or `null` when nothing is focused. Purely a UI concern -
   * never filters which events are fetched or rendered. */
  focusEventId: number | null;
  /** Called with a group's `fire_event_id` when the user asks to focus it
   * on the map. The page owns turning this into a `focusEventId` URL
   * update - this component never navigates or mutates the URL itself. */
  onFocusEvent: (fireEventId: number) => void;
  /** Display name per event id (English location name). Falls back to
   * "Event #id" for an id without one. */
  eventLabels?: Record<number, string>;
  /** The RAW (nullable) resolved place name per event id - unlike
   * `eventLabels`, never pre-filled with "Event #id", so it can be handed
   * straight to `ResponseActions`' `locationName` (which does its own,
   * different fallback to "Target #id"). */
  eventPlaceNames?: Record<number, string | null>;
  /** Per-target place names from reverse geocoding (target id -> name),
   * shared across every event's `ResponseActions` group. */
  targetLocations?: Record<number, string | null>;
}

const NO_EVENTS_TITLE = "No materialized fire events";
const NO_EVENTS_MESSAGE = "This generation has no fire events with a materialized response plan.";

/**
 * Every materialized FireEvent in the current global generation
 * (`plan.events`), already grouped one-per-event by the backend - this
 * component performs no client-side grouping of its own, only rendering.
 * Each group's severity/coverage/resource-count fields and its
 * routes/actions/uncovered-targets are shown exactly as the backend
 * assembled them, reusing `ResponseActions`/`UncoveredTargets` (the same
 * presentation components the single-event Response Plan page uses,
 * including its per-station truck grouping) rather than a parallel
 * rendering of the same `ResponsePlanAction`/`ResponsePlanTarget` shapes.
 */
export function EventGroupList({
  events,
  focusEventId,
  onFocusEvent,
  eventLabels = {},
  eventPlaceNames = {},
  targetLocations = {},
}: EventGroupListProps) {
  return (
    <section aria-labelledby="event-group-list-heading" className="event-group-list">
      <h2 id="event-group-list-heading" className="event-group-list__title">
        Fire Events
      </h2>
      {events.length === 0 ? (
        <EmptyState title={NO_EVENTS_TITLE} message={NO_EVENTS_MESSAGE} />
      ) : (
        <div className="event-group-list__groups">
          {events.map((event) => {
            const isFocused = event.fire_event_id === focusEventId;
            return (
              <article
                key={event.fire_event_id}
                className={isFocused ? "event-group event-group--focused" : "event-group"}
              >
                <div className="event-group__header">
                  <h3 className="event-group__title">
                    <Link to={`/events/${event.fire_event_id}`}>
                      {eventLabels[event.fire_event_id] ?? `Event #${event.fire_event_id}`}
                    </Link>
                  </h3>
                  <button
                    type="button"
                    className="event-group__focus-button"
                    onClick={() => onFocusEvent(event.fire_event_id)}
                    aria-pressed={isFocused}
                  >
                    {isFocused ? "Focused" : "Focus on map"}
                  </button>
                </div>

                <dl className="event-group__facts">
                  <div className="event-group__fact">
                    <dt>Severity</dt>
                    <dd>
                      <SeverityBadge level={event.severity_level} />
                    </dd>
                  </div>
                  <div className="event-group__fact">
                    <dt>Coverage</dt>
                    <dd>
                      {event.coverage_score !== null
                        ? formatCoveragePercentage(event.coverage_score)
                        : NOT_AVAILABLE_LABEL}
                    </dd>
                  </div>
                  <div className="event-group__fact">
                    <dt>Average ETA</dt>
                    <dd>
                      {event.average_eta_seconds !== null
                        ? formatDurationSeconds(event.average_eta_seconds)
                        : NOT_AVAILABLE_LABEL}
                    </dd>
                  </div>
                </dl>

                <p className="event-group__resources" aria-label="Resources">
                  <span>
                    Minimum <strong>{event.minimum_resources ?? NOT_AVAILABLE_LABEL}</strong>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Desired <strong>{event.desired_resources ?? NOT_AVAILABLE_LABEL}</strong>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Assigned <strong>{event.assigned_resources ?? NOT_AVAILABLE_LABEL}</strong>
                  </span>
                </p>

                <ResponseActions
                  actions={event.actions}
                  locationName={eventPlaceNames[event.fire_event_id] ?? null}
                  targetLocations={targetLocations}
                  headingId={`response-actions-heading-${event.fire_event_id}`}
                />

                {event.uncovered_targets.length > 0 ? <UncoveredTargets targets={event.uncovered_targets} /> : null}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
