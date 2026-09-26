import { ActiveFireEventCard } from "../fire-events/ActiveFireEventCard";
import { EmptyState } from "../feedback/EmptyState";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import "./ActiveFiresPanel.css";

export interface ActiveFiresPanelProps {
  activeFires: ActiveFireEvent[];
  /** Empty-state copy override, e.g. the "No simulation started" demo state. */
  emptyTitle?: string;
  emptyMessage?: string;
}

/**
 * The Operations Overview's compact Active Fires panel (Task A8): every A6
 * `active_fires` entry, rendered in exact server order - never re-sorted or
 * re-ranked by an invented priority score (Part 24). Reuses the existing
 * ActiveFireEventCard unchanged (including its Task A8 emphasis styling and
 * its existing navigation to Event Details).
 *
 * Production polish pass: this panel no longer carries its own "View
 * Response Plan" action - the page header's "Global Response Plan" link
 * (ActiveWildfiresPage) is the one route to it, so this panel does not
 * duplicate that navigation.
 */
export function ActiveFiresPanel({
  activeFires,
  emptyTitle = "No active wildfire events",
  emptyMessage = "There are currently no suspected or confirmed wildfire events.",
}: ActiveFiresPanelProps) {
  return (
    <section aria-labelledby="active-fires-panel-heading" className="active-fires-panel">
      <div className="active-fires-panel__header">
        <h2 id="active-fires-panel-heading" className="active-fires-panel__title">
          Active Fires
          {activeFires.length > 0 ? (
            <span className="active-fires-panel__count">{activeFires.length}</span>
          ) : null}
        </h2>
      </div>
      {activeFires.length === 0 ? (
        <EmptyState title={emptyTitle} message={emptyMessage} />
      ) : (
        <div className="active-fires-panel__list">
          {activeFires.map((fire) => (
            <ActiveFireEventCard key={fire.fire_event_id} event={fire} />
          ))}
        </div>
      )}
    </section>
  );
}
