import type { ResponsePlan } from "../../types/responsePlan";
import "./ResponsePlanSummary.css";

export interface PlanStateNoticesProps {
  plan: Pick<ResponsePlan, "status" | "no_resources_during_planning">;
}

const PARTIAL_MESSAGE = "This response plan does not cover all response targets.";
const NO_FEASIBLE_MESSAGE = "This response plan contains no feasible assignments.";
const NO_RESOURCES_MESSAGE =
  "No firefighting resources were available in the planning snapshot used for this plan.";

type NoticeTone = "warning" | "danger";

/**
 * Presentation-only operational notices, driven strictly by persisted
 * backend fields (`plan.status`, `plan.no_resources_during_planning`).
 * Never determines partial/no-feasible status by comparing actions against
 * targets itself, and never infers *why* resources were unavailable beyond
 * the boolean flag. A `complete` plan renders no warning here (the Task 7
 * status badge already shows it). `partial` and `no_resources_during_planning`
 * are independent facts - if the backend returns both, both notices render;
 * neither is dropped in favor of the other.
 */
export function PlanStateNotices({ plan }: PlanStateNoticesProps) {
  const notices: { key: string; tone: NoticeTone; message: string }[] = [];

  if (plan.status === "partial") {
    notices.push({ key: "partial", tone: "warning", message: PARTIAL_MESSAGE });
  }
  if (plan.status === "no_feasible_assignments") {
    notices.push({ key: "no-feasible", tone: "danger", message: NO_FEASIBLE_MESSAGE });
  }
  if (plan.no_resources_during_planning) {
    notices.push({ key: "no-resources", tone: "warning", message: NO_RESOURCES_MESSAGE });
  }

  if (notices.length === 0) {
    return null;
  }

  return (
    <div className="response-plan-summary__notices">
      {notices.map((notice) => (
        <p
          key={notice.key}
          role="status"
          className={`response-plan-summary__notice response-plan-summary__notice--${notice.tone}`}
        >
          {notice.message}
        </p>
      ))}
    </div>
  );
}
