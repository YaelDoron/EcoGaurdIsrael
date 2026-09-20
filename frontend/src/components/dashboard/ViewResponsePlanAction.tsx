import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getOperationsActivityDetail } from "../../api/operations";
import "./ViewResponsePlanAction.css";

export interface ViewResponsePlanActionProps {
  /** The newest persisted GlobalPlanningRun's id from A6's activity feed, or `null` when none exists yet. No button renders when `null` - never a fake destination. */
  globalPlanningRunId: number | null;
}

const GENERIC_ERROR_MESSAGE = "Unable to open the response plan. Please try again.";

/**
 * The Global Planning workflow is no longer represented as an Activity Feed
 * row (see OperationsActivityFeed) - instead it is one explicit action next
 * to Active Fires. A6's activity item only carries the GlobalPlanningRun's
 * own id, not the ResponsePlan id the existing `/plans/:planId` route
 * needs, so the real A5 detail
 * (`GET /api/v1/operations/activity/global_planning_run/{id}`) is fetched
 * ONLY when the operator clicks - never on a poll/timer, and never more
 * than once per click (single-flight guarded, mirroring
 * `useStartSimulation`'s own pattern). A GlobalPlanningRun can cover
 * multiple FireEvents/ResponsePlans; this opens the first real persisted
 * `response_plan_id` through the SAME existing Response Plan route already
 * used elsewhere (see OperationsActivityDrawer's GlobalPlanningRunDetailView) -
 * never a fabricated id.
 */
export function ViewResponsePlanAction({ globalPlanningRunId }: ViewResponsePlanActionProps) {
  const navigate = useNavigate();
  const [isResolving, setIsResolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isFetchingRef = useRef(false);

  if (globalPlanningRunId === null) {
    return null;
  }

  const handleClick = () => {
    if (isFetchingRef.current) {
      return;
    }
    isFetchingRef.current = true;
    setIsResolving(true);
    setError(null);

    getOperationsActivityDetail("global_planning_run", globalPlanningRunId)
      .then((detail) => {
        if (detail.activity_type !== "global_planning_run") {
          return;
        }
        const [firstResponsePlanId] = detail.details.response_plan_ids;
        if (firstResponsePlanId === undefined) {
          setError(GENERIC_ERROR_MESSAGE);
          return;
        }
        navigate(`/plans/${firstResponsePlanId}`);
      })
      .catch(() => {
        setError(GENERIC_ERROR_MESSAGE);
      })
      .finally(() => {
        isFetchingRef.current = false;
        setIsResolving(false);
      });
  };

  return (
    <div className="view-response-plan-action">
      <button type="button" className="view-response-plan-action__button" onClick={handleClick} disabled={isResolving}>
        {isResolving ? "Opening…" : "View Response Plan"}
      </button>
      {error ? <p className="view-response-plan-action__error">{error}</p> : null}
    </div>
  );
}
