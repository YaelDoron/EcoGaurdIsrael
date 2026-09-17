import type { ResponsePlanAction } from "../../types/responsePlan";
import { EmptyState } from "../feedback/EmptyState";
import { ResponseActionCard } from "./ResponseActionCard";
import { getResponseActionKey } from "./ResponseRouteLayerModel";
import "./ResponseActionCard.css";
import "./ResponsePlanSummary.css";

export interface ResponseActionsProps {
  actions: ResponsePlanAction[];
  /** The currently selected action's key, or `null`/omitted when nothing is
   * selected. See `getResponseActionKey`. */
  selectedActionKey?: string | null;
  /** Called with an action's key when the user selects/deselects it in the
   * list. Omit to render a non-selectable list (e.g. existing Task 9
   * usages/tests). */
  onSelectAction?: (actionKey: string) => void;
}

const EMPTY_TITLE = "No response actions";
const EMPTY_MESSAGE = "This response plan has no assigned resources.";

/**
 * Renders every persisted response action (`plan.actions`) exactly as
 * returned by the backend, in backend order. Never sorts by ETA/priority,
 * never drops an incomplete action (an unreachable/unmappable action stays
 * visible even though it has no drawable route), never merges actions, and
 * never fabricates one from `plan.uncovered_targets` or any resource/target
 * list. Selection is UI-only highlight state, passed through to each card -
 * it never changes the rendered order or the action data itself.
 */
export function ResponseActions({ actions, selectedActionKey = null, onSelectAction }: ResponseActionsProps) {
  return (
    <section aria-labelledby="response-actions-heading" className="response-plan-summary__section">
      <h2 id="response-actions-heading" className="response-plan-summary__section-title">
        Response Actions
      </h2>
      {actions.length === 0 ? (
        <EmptyState title={EMPTY_TITLE} message={EMPTY_MESSAGE} />
      ) : (
        <div className="response-action-list">
          {actions.map((action) => {
            const actionKey = getResponseActionKey(action);
            return (
              <ResponseActionCard
                key={actionKey}
                action={action}
                isSelected={actionKey === selectedActionKey}
                onSelect={onSelectAction}
              />
            );
          })}
        </div>
      )}
    </section>
  );
}
