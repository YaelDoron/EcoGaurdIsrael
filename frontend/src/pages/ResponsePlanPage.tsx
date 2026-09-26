import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { LatLngPoint } from "../components/map/mapTypes";
import { MapView } from "../components/map/MapView";
import { BaselineComparisonSection } from "../components/response-plan/BaselineComparison";
import { PlanMetrics } from "../components/response-plan/PlanMetrics";
import { PlanStateNotices } from "../components/response-plan/PlanStateNotices";
import { ResponseActions } from "../components/response-plan/ResponseActions";
import { RouteLegend } from "../components/response-plan/RouteLegend";
import { ResponsePlanMapLayer } from "../components/response-plan/ResponsePlanMapLayer";
import { ResponseRouteLayer } from "../components/response-plan/ResponseRouteLayer";
import { getResponseActionKey } from "../components/response-plan/ResponseRouteLayerModel";
import { UncoveredTargets } from "../components/response-plan/UncoveredTargets";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { PageHeader } from "../components/layout/PageHeader";
import "../components/status/badges.css";
import "../components/response-plan/ResponsePlanSummary.css";
import { useEventLocationName } from "../hooks/useEventLocationName";
import { useResponsePlan } from "../hooks/useResponsePlan";
import { useTargetLocationNames, type GeocodeTarget } from "../hooks/useTargetLocationNames";
import type { ResponsePlanSource } from "../hooks/useResponsePlan";
import type { ResponsePlan, ResponsePlanStatus } from "../types/responsePlan";
import "./ResponsePlanPage.css";

const PAGE_TITLE = "Response Plan";
// Shown while loading so the header matches the spinner instead of a bare, incomplete title.
const LOADING_TITLE = "Loading Response Plan…";

const INVALID_REQUEST_TITLE = "Invalid response plan request.";
const INVALID_REQUEST_MESSAGE = "The requested identifier is not valid.";
const LOAD_ERROR_TITLE = "Unable to load the response plan.";
const NO_PLAN_TITLE = "No response plan available";
const NO_PLAN_MESSAGE = "There is currently no response plan available for this wildfire event.";
const GENERATING_PLAN_MESSAGE = "Generating response plan...";
const GENERATING_PLAN_DETAIL =
  "This fire is confirmed; routing and resource allocation are running. The plan appears here automatically.";

type StatusTone = "success" | "warning" | "danger";

/**
 * Presentation-only label/tone for the persisted plan `status`. Never
 * derives status from action counts or anything else - it only formats the
 * value the backend already resolved (see backend/src/models/response_plan_status.py).
 */
const STATUS_PRESENTATION: Record<ResponsePlanStatus, { label: string; tone: StatusTone }> = {
  complete: { label: "Calculation: Success", tone: "success" },
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

/**
 * Every persisted origin/target coordinate across this plan's actions - the
 * only input to the shared map's auto-fit viewport (see
 * `FitBoundsToPoints`), so the map always frames this plan's real data and
 * never a hardcoded region.
 */
function buildBoundsPoints(plan: ResponsePlan): LatLngPoint[] {
  const points: LatLngPoint[] = [];
  for (const action of plan.actions) {
    if (action.resource.origin !== null) {
      points.push({ lat: action.resource.origin.latitude, lng: action.resource.origin.longitude });
    }
    if (action.target.latitude !== null && action.target.longitude !== null) {
      points.push({ lat: action.target.latitude, lng: action.target.longitude });
    }
    for (const point of action.route.path_coordinates ?? []) {
      points.push({ lat: point.latitude, lng: point.longitude });
    }
  }
  return points;
}

const GENERATED_TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit" });

/** Time of day only ("13:33") - the date is deliberately omitted to keep the header quiet. */
function formatGeneratedTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Not available" : GENERATED_TIME_FORMATTER.format(date);
}

/** Each distinct target that has coordinates, for reverse geocoding. */
function collectGeocodeTargets(plan: ResponsePlan): GeocodeTarget[] {
  const byId = new Map<number, GeocodeTarget>();
  for (const action of plan.actions) {
    const { response_target_id: id, latitude, longitude } = action.target;
    if (latitude !== null && longitude !== null && !byId.has(id)) {
      byId.set(id, { id, latitude, longitude });
    }
  }
  return Array.from(byId.values());
}

function ResponsePlanContent({ source }: { source: ResponsePlanSource }) {
  const { plan, planStatus, isLoading, error, retry } = useResponsePlan(source);
  // UI-only highlight state (Task 11): selecting an action never reorders
  // plan.actions, changes the assignment, or recalculates anything - it
  // only marks which action's card/route should be visually emphasized.
  // Clicking the currently-selected action again clears the selection.
  // `undefined` means "no choice made yet": the first listed action is
  // highlighted by default. Clicking the selected action clears it (`null`).
  const [selectedActionKey, setSelectedActionKey] = useState<string | null | undefined>(undefined);
  const firstActionKey = plan && plan.actions.length > 0 ? getResponseActionKey(plan.actions[0]) : null;
  const effectiveSelectedKey = selectedActionKey === undefined ? firstActionKey : selectedActionKey;
  const handleSelectAction = (actionKey: string) => {
    setSelectedActionKey(effectiveSelectedKey === actionKey ? null : actionKey);
  };
  const boundsPoints = useMemo(() => (plan ? buildBoundsPoints(plan) : []), [plan]);
  // Region name inherited from the parent FireEvent (targets carry none).
  const locationName = useEventLocationName(plan ? plan.fire_event_id : null);
  // Per-target place names from the target's own coordinates.
  const geocodeTargets = useMemo(() => (plan ? collectGeocodeTargets(plan) : []), [plan]);
  const targetLocations = useTargetLocationNames(geocodeTargets);
  // Title place: the parent event's location, else the first place resolved
  // from the plan's own target coordinates (in action order).
  const titlePlace =
    locationName ??
    (plan
      ? geocodeTargets.map((target) => targetLocations[target.id]).find((name): name is string => !!name) ?? null
      : null);

  if (isLoading) {
    return (
      <section>
        <PageHeader title={LOADING_TITLE} />
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
    // A CONFIRMED event's plan is produced by the confirmed-fire pipeline
    // (severity -> spread -> targets -> routing -> allocation); until it is
    // persisted the backend reports `generating` and the hook keeps polling.
    // A SUSPECTED (monitoring-only) event never implies a plan is coming.
    return (
      <section>
        <PageHeader title={PAGE_TITLE} />
        {planStatus === "generating" ? (
          <>
            <LoadingState message={GENERATING_PLAN_MESSAGE} />
            <p className="response-plan-page__lifecycle-detail">{GENERATING_PLAN_DETAIL}</p>
          </>
        ) : (
          <EmptyState title={NO_PLAN_TITLE} message={NO_PLAN_MESSAGE} />
        )}
      </section>
    );
  }

  const statusPresentation = STATUS_PRESENTATION[plan.status];

  return (
    <section>
      <nav aria-label="Breadcrumb" className="response-plan-page__breadcrumb-nav">
        <Link to={`/events/${plan.fire_event_id}`} className="response-plan-page__breadcrumb">
          <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M10 3 5 8l5 5" />
          </svg>
          Back to Event
        </Link>
      </nav>
      <PageHeader
        title={titlePlace ? `${PAGE_TITLE} for ${titlePlace}` : `${PAGE_TITLE} for Event #${plan.fire_event_id}`}
      />

      <div className="response-plan-page__meta" aria-label="Plan details">
        <span className="response-plan-page__generated">
          Generated: <time dateTime={plan.generated_at}>{formatGeneratedTime(plan.generated_at)}</time>
        </span>
        {/* The title already carries the event id when no location name is known. */}
        {titlePlace ? <span className="response-plan-page__event-badge">Event #{plan.fire_event_id}</span> : null}
        {!plan.is_current ? <span className="badge badge--neutral">SUPERSEDED</span> : null}
        {plan.status !== "complete" ? (
          <span className={`badge badge--${statusPresentation.tone}`}>{statusPresentation.label}</span>
        ) : null}
      </div>

      <div className="response-plan-page__layout">
        <div className="response-plan-page__details">
          <PlanStateNotices plan={plan} />

          <PlanMetrics metrics={plan.metrics} />

          <ResponseActions
            actions={plan.actions}
            locationName={locationName}
            targetLocations={targetLocations}
            selectedActionKey={effectiveSelectedKey}
            onSelectAction={handleSelectAction}
          />

          <UncoveredTargets targets={plan.uncovered_targets} />

          {plan.baseline_comparison !== null ? (
            <BaselineComparisonSection comparison={plan.baseline_comparison} />
          ) : null}

        </div>

        <section
          aria-labelledby="response-plan-map-heading"
          className="response-plan-summary__section response-plan-page__map"
        >
          <h2 id="response-plan-map-heading" className="response-plan-summary__section-title">
            Map
          </h2>
          <RouteLegend />
          <MapView boundsPoints={boundsPoints} ariaLabel={`Map of Response Plan for Event #${plan.fire_event_id}`}>
            <ResponseRouteLayer actions={plan.actions} selectedActionKey={effectiveSelectedKey}>
              {(layer) => <ResponsePlanMapLayer layer={layer} locationName={locationName} targetLocations={targetLocations} />}
            </ResponseRouteLayer>
          </MapView>
        </section>
      </div>
    </section>
  );
}
