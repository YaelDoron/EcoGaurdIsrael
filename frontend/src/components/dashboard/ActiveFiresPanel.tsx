import { ActiveFireEventCard } from "../fire-events/ActiveFireEventCard";
import { EmptyState } from "../feedback/EmptyState";
import { ViewResponsePlanAction } from "./ViewResponsePlanAction";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import "./ActiveFiresPanel.css";

export interface ActiveFiresPanelProps {
  activeFires: ActiveFireEvent[];
  /** The newest persisted GlobalPlanningRun's id (from A6's activity feed), or `null` when none exists yet - drives the header's "View Response Plan" action. */
  globalPlanningRunId: number | null;
}

/**
 * The Operations Overview's compact Active Fires panel (Task A8): every A6
 * `active_fires` entry, rendered in exact server order - never re-sorted or
 * re-ranked by an invented priority score (Part 24). Reuses the existing
 * ActiveFireEventCard unchanged (including its Task A8 emphasis styling and
 * its existing navigation to Event Details).
 *
 * The header also carries the one "View Response Plan" action - Global
 * Planning is no longer an Activity Feed row (see OperationsActivityFeed),
 * so this is now the operator's route to it.
 */
export function ActiveFiresPanel({ activeFires, globalPlanningRunId }: ActiveFiresPanelProps) {
  return (
    <section aria-labelledby="active-fires-panel-heading" className="active-fires-panel">
      <div className="active-fires-panel__header">
        <h2 id="active-fires-panel-heading" className="active-fires-panel__title">
          Active Fires
          {activeFires.length > 0 ? (
            <span className="active-fires-panel__count">{activeFires.length}</span>
          ) : null}
        </h2>
        <ViewResponsePlanAction globalPlanningRunId={globalPlanningRunId} />
      </div>
      {activeFires.length === 0 ? (
        <EmptyState
          title="No active wildfire events"
          message="There are currently no suspected or confirmed wildfire events."
        />
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
