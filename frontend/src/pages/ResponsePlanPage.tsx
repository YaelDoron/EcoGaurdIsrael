import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { BaselineComparisonSection } from "../components/response-plan/BaselineComparison";
import { OptimizationDetails } from "../components/response-plan/OptimizationDetails";
import { PlanMetrics } from "../components/response-plan/PlanMetrics";
import { PlanStateNotices } from "../components/response-plan/PlanStateNotices";
import { ResponseActions } from "../components/response-plan/ResponseActions";
import { UncoveredTargets } from "../components/response-plan/UncoveredTargets";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { PageHeader } from "../components/layout/PageHeader";
import { TimestampDisplay } from "../components/data/TimestampDisplay";
import "../components/status/badges.css";
import { useResponsePlan } from "../hooks/useResponsePlan";
import type { ResponsePlanSource } from "../hooks/useResponsePlan";
import type { ResponsePlanStatus } from "../types/responsePlan";
import "./ResponsePlanPage.css";

const PAGE_TITLE = "Response Plan";

const INVALID_REQUEST_TITLE = "Invalid response plan request.";
const INVALID_REQUEST_MESSAGE = "The requested identifier is not valid.";
const LOAD_ERROR_TITLE = "Unable to load the response plan.";
const NO_PLAN_TITLE = "No response plan available";
const NO_PLAN_MESSAGE = "There is currently no response plan available for this wildfire event.";

type StatusTone = "success" | "warning" | "danger";

/**
 * Presentation-only label/tone for the persisted plan `status`. Never
 * derives status from action counts or anything else - it only formats the
 * value the backend already resolved (see backend/src/models/response_plan_status.py).
 */
const STATUS_PRESENTATION: Record<ResponsePlanStatus, { label: string; tone: StatusTone }> = {
  complete: { label: "Complete", tone: "success" },
  partial: { label: "Partial", tone: "warning" },
  no_feasible_assignments: { label: "No feasible assignments", tone: "danger" },
};

/**
 * Reserved for both `/events/:fireEventId/plan` and `/plans/:planId` (see
 * AppRouter.tsx, still pointed at ResponsePlanPlaceholderPage until a
 * teammate wires this component in). Which US 6.3 data source to request is
 * decided purely from which route param is present - never both at once.
 */
export function ResponsePlanPage() {
  const { fireEventId: fireEventIdParam, planId: planIdParam } = useParams<{
    fireEventId?: string;
    planId?: string;
  }>();

  const source = resolveSource(fireEventIdParam, planIdParam);

  if (source === null) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <ErrorState title={INVALID_REQUEST_TITLE} message={INVALID_REQUEST_MESSAGE} />
      </section>
    );
  }

  return <ResponsePlanContent source={source} />;
}

function resolveSource(fireEventIdParam: string | undefined, planIdParam: string | undefined): ResponsePlanSource | null {
  if (fireEventIdParam !== undefined) {
    const fireEventId = parsePositiveIntParam(fireEventIdParam);
    return fireEventId !== null ? { kind: "current", fireEventId } : null;
  }
  if (planIdParam !== undefined) {
    const planId = parsePositiveIntParam(planIdParam);
    return planId !== null ? { kind: "by-id", planId } : null;
  }
  return null;
}

function parsePositiveIntParam(value: string): number | null {
  if (!/^\d+$/.test(value)) {
    return null;
  }
  const parsed = Number(value);
  return parsed > 0 ? parsed : null;
}

function ResponsePlanContent({ source }: { source: ResponsePlanSource }) {
  const { plan, isLoading, error, retry } = useResponsePlan(source);
  // UI-only highlight state (Task 11): selecting an action never reorders
  // plan.actions, changes the assignment, or recalculates anything - it
  // only marks which action's card/route should be visually emphasized.
  // Clicking the currently-selected action again clears the selection.
  const [selectedActionKey, setSelectedActionKey] = useState<string | null>(null);
  const handleSelectAction = (actionKey: string) => {
    setSelectedActionKey((current) => (current === actionKey ? null : actionKey));
  };

  if (isLoading) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <LoadingState message="Loading response plan…" />
      </section>
    );
  }

  if (error) {
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <ErrorState title={LOAD_ERROR_TITLE} message={error} onRetry={retry} />
      </section>
    );
  }

  if (plan === null) {
    // Only reachable for the "current" source - a missing plan-by-id
    // surfaces as an ApiError (404), not a successful `plan: null`.
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        <EmptyState title={NO_PLAN_TITLE} message={NO_PLAN_MESSAGE} />
      </section>
    );
  }

  const statusPresentation = STATUS_PRESENTATION[plan.status];

  return (
    <section>
      <PageHeader title={PAGE_TITLE} />

      <dl className="response-plan-page__identity">
        <div className="response-plan-page__row">
          <dt>Plan ID</dt>
          <dd>{plan.plan_id}</dd>
        </div>

        <div className="response-plan-page__row">
          <dt>Fire event</dt>
          <dd>{plan.fire_event_id}</dd>
        </div>

        <div className="response-plan-page__row">
          <dt>Generated</dt>
          <dd>
            <TimestampDisplay value={plan.generated_at} />
          </dd>
        </div>

        <div className="response-plan-page__row">
          <dt>Currency</dt>
          <dd>
            <span className={`badge badge--${plan.is_current ? "success" : "neutral"}`}>
              {plan.is_current ? "CURRENT" : "SUPERSEDED"}
            </span>
          </dd>
        </div>

        <div className="response-plan-page__row">
          <dt>Outcome</dt>
          <dd>
            <span className={`badge badge--${statusPresentation.tone}`}>{statusPresentation.label}</span>
          </dd>
        </div>

        <div className="response-plan-page__row">
          <dt>Methodology</dt>
          <dd>
            {plan.methodology} (v{plan.methodology_version})
          </dd>
        </div>
      </dl>

      <PlanStateNotices plan={plan} />

      <PlanMetrics metrics={plan.metrics} />

      <BaselineComparisonSection comparison={plan.baseline_comparison} />

      <ResponseActions
        actions={plan.actions}
        selectedActionKey={selectedActionKey}
        onSelectAction={handleSelectAction}
      />

      {/*
        US 6.2 map integration point: once the shared map surface (e.g. a
        `MapView`) exists, mount it here and feed it
        `buildResponseRouteLayer(plan.actions, selectedActionKey)` (or wrap
        it in `<ResponseRouteLayer actions={plan.actions}
        selectedActionKey={selectedActionKey}>{(layer) => ...}</ResponseRouteLayer>`)
        to draw persisted routes/origin/target markers, with the selected
        action's route/markers visually emphasized via each feature's
        `isSelected` flag. See src/components/response-plan/responseRouteLayer.ts.
      */}

      <UncoveredTargets targets={plan.uncovered_targets} />

      <OptimizationDetails plan={plan} />

      <Link to={`/events/${plan.fire_event_id}`} className="response-plan-page__back-link">
        Back to Event
      </Link>
    </section>
  );
}
