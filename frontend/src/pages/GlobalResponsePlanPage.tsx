import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { EventGroupList } from "../components/global-response/EventGroupList";
import { GlobalMetricsPanel } from "../components/global-response/GlobalMetricsPanel";
import { GlobalResponseMapLayer } from "../components/global-response/GlobalResponseMapLayer";
import { buildGlobalResponseMapLayer } from "../components/global-response/GlobalResponseMapLayerModel";
import { PageHeader } from "../components/layout/PageHeader";
import type { LatLngPoint } from "../components/map/mapTypes";
import { MapView } from "../components/map/MapView";
import { useEventLocationNames } from "../hooks/useEventLocationNames";
import { isDemoRunLive, useCurrentDemoSimulation, useDemoDataVisibility } from "../hooks/demoSession";
import { GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS } from "../config/polling";
import { useGlobalResponsePlan } from "../hooks/useGlobalResponsePlan";
import { useTargetLocationNames, type GeocodeTarget } from "../hooks/useTargetLocationNames";
import type { GlobalEventPlan, GlobalPlanningRunStatus } from "../types/globalResponsePlan";
import "./GlobalResponsePlanPage.css";

const PAGE_TITLE = "Global Response Plan";
const PAGE_DESCRIPTION = "The latest materialized global generation across every active wildfire event.";
const LOAD_ERROR_TITLE = "Unable to load the global response plan.";
const NO_GENERATION_TITLE = "No response plan available";
const NO_GENERATION_MESSAGE =
  "No global response plan has been generated yet. One appears here once a confirmed fire has been planned.";

const GENERATING_TITLE = "Generating response plan...";
// After the run ends, a plan still being generated/updated is re-checked at
// most this many more times (~1 minute) - never forever.
const MAX_POLLS_AFTER_RUN_ENDED = 12;
const UPDATING_TITLE = "Updating response plan...";

function joinNames(names: string[]): string {
  return names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

function confirmedFires(count: number): string {
  return `${count} confirmed fire${count === 1 ? "" : "s"}`;
}

const FOCUS_EVENT_ID_PARAM = "focusEventId";

// Presentation-only wording for the backend's run status - never derived.
const RUN_STATUS_LABEL: Record<GlobalPlanningRunStatus, string> = {
  running: "Calculating…",
  completed: "Complete",
  partial: "Partial plan",
  failed: "Calculation failed",
  no_active_events: "No active events",
};

// A stable reference (not a fresh `[]` literal every render) so `events`
// stays referentially unchanged across renders when there is no plan,
// keeping the `useMemo` calls below actually memoized.
const EMPTY_EVENTS: GlobalEventPlan[] = [];

/**
 * Reserved for a future `/global-response-plan` route (see AppRouter.tsx -
 * not wired in here, matching `ResponsePlanPage`'s own precedent: shared
 * route-table wiring is left to whoever owns that file). Fetches
 * `GET /api/v1/global-response-plan/current` via `useGlobalResponsePlan()`
 * and renders the latest materialized generation's metrics, shortage,
 * optimization config, per-event breakdown, and an interactive map - all
 * straight from that one API response.
 *
 * `focusEventId` (Tasks B-FE-5/B-FE-8) lives entirely in the URL query
 * string: reading/writing it never triggers a new fetch (`useGlobalResponsePlan`
 * takes no arguments at all) and only changes which event's markers/routes
 * are drawn emphasized vs. dimmed (`GlobalResponseMapLayer`) and which
 * group is visually marked in `EventGroupList` - a pure frontend
 * highlight/filter concern, never a server-side query parameter.
 */
export function GlobalResponsePlanPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  // A plan left over from a previous demo run is not shown until this browser
  // session starts/observes a run (hooks/demoSession.ts) - presentation only.
  // The (instant) current-run check settles first: when the gate hides the
  // plan, the page renders its empty state at once and never waits for, or
  // even requests, the multi-second plan read.
  //
  // Live refresh: the run state is re-read every poll interval (a stale
  // one-shot read taken at mount used to gate a page opened before a run went
  // live forever), and the plan/lifecycle is re-read every interval while the
  // run is PREPARING/RUNNING/STOPPING - in EVERY lifecycle state, so a page
  // sitting on "No response plan available" discovers a newly confirmed fire.
  // After the run ends, polling continues only while a plan is still
  // generating/updating, for at most MAX_POLLS_AFTER_RUN_ENDED checks.
  const simulation = useCurrentDemoSimulation(GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS);
  const showDemoData = useDemoDataVisibility(simulation);
  const runLive = isDemoRunLive(simulation);
  // While PREPARING, the mandatory reset has not committed yet: the database
  // still holds the PREVIOUS run's fires and plans. Nothing is read (or shown)
  // until the new run is RUNNING.
  const preparing = simulation?.run?.state === "preparing";
  const [keepPolling, setKeepPolling] = useState(true);
  const { response: planResponse, isLoading: isPlanLoading, error, retry } = useGlobalResponsePlan({
    enabled: simulation !== null && showDemoData && !preparing,
    polling: keepPolling,
  });
  // Until the first plan response arrives the page is loading - never a
  // transient "No response plan available" in the render between the gate
  // opening and the fetch starting.
  const isLoading =
    simulation === null || preparing || (showDemoData && (isPlanLoading || (planResponse === null && !error)));
  const response = showDemoData ? planResponse : null;

  // Lifecycle, derived by the backend from persisted state (GlobalPlanCoverage)
  // strictly from CONFIRMED (response-eligible) fires: a confirmed fire is
  // planned asynchronously by the confirmed-fire pipeline, so "no plan yet" /
  // "plan not covering every confirmed fire" are generating / updating; a
  // SUSPECTED fire is monitoring-only and never counts.
  const coverage = response?.coverage ?? null;
  const lifecycleBuilding = coverage?.state === "generating" || coverage?.state === "updating";
  const pollsAfterRunEndedRef = useRef(0);
  useEffect(() => {
    if (simulation === null) {
      return; // run state not known yet - keep the default (polling on)
    }
    if (runLive) {
      pollsAfterRunEndedRef.current = 0;
      setKeepPolling(true);
      return;
    }
    if (planResponse !== null) {
      pollsAfterRunEndedRef.current += 1;
    }
    setKeepPolling(lifecycleBuilding && pollsAfterRunEndedRef.current <= MAX_POLLS_AFTER_RUN_ENDED);
  }, [simulation, runLive, planResponse, lifecycleBuilding]);

  const pendingIds = useMemo(() => coverage?.pending_fire_event_ids ?? [], [coverage]);
  const pendingNames = useEventLocationNames(pendingIds);
  const pendingLabels = pendingIds.map((id) => pendingNames[id] ?? `Event #${id}`);
  const monitoringCount = coverage?.monitoring_fire_event_ids?.length ?? 0;

  const focusEventId = parseFocusEventId(searchParams.get(FOCUS_EVENT_ID_PARAM));

  // Clicking a group's focus button toggles it: focusing an already-focused
  // event releases the focus.
  const handleFocusEvent = (fireEventId: number) => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      if (parseFocusEventId(previous.get(FOCUS_EVENT_ID_PARAM)) === fireEventId) {
        next.delete(FOCUS_EVENT_ID_PARAM);
      } else {
        next.set(FOCUS_EVENT_ID_PARAM, String(fireEventId));
      }
      return next;
    });
  };

  const events = response?.plan?.events ?? EMPTY_EVENTS;
  // Always frame every materialized event (fitBounds zooms out as far as
  // needed when they are geographically spread), so focusing never hides the
  // others - it only restyles them. Presentation only, no refetch.
  const boundsPoints = useMemo(() => buildBoundsPoints(events), [events]);
  // English display name per event: persisted news-evidence location name
  // (the backend ingestion pipeline already translates this to English
  // before saving - displayed as-is, never re-translated here), else a
  // place reverse-geocoded from the event's own ACTIVE_FIRE target
  // coordinates, else "Event #id" - the same fallback EventDetailsPage/
  // ResponsePlanPage already use, so an event with no news evidence still
  // gets a real name instead of a bare id.
  const eventIds = useMemo(() => events.map((event) => event.fire_event_id), [events]);
  const locationNames = useEventLocationNames(eventIds);
  const eventGeocodeTargets = useMemo(() => collectEventGeocodeTargets(events), [events]);
  const geocodedEventPlaces = useTargetLocationNames(eventGeocodeTargets);
  const eventPlaceNames = useMemo(() => {
    const names: Record<number, string | null> = {};
    for (const id of eventIds) {
      names[id] = locationNames[id] ?? geocodedEventPlaces[id] ?? null;
    }
    return names;
  }, [eventIds, locationNames, geocodedEventPlaces]);
  const eventLabels = useMemo(() => {
    const labels: Record<number, string> = {};
    for (const id of eventIds) {
      labels[id] = eventPlaceNames[id] ?? `Event #${id}`;
    }
    return labels;
  }, [eventIds, eventPlaceNames]);
  // Per-target place names (used inside each event's ResponseActions group
  // headers), reverse-geocoded from every action's own target coordinates.
  const targetGeocodeTargets = useMemo(() => collectTargetGeocodeTargets(events), [events]);
  const targetLocations = useTargetLocationNames(targetGeocodeTargets);
  const mapLayer = useMemo(() => buildGlobalResponseMapLayer(events, focusEventId), [events, focusEventId]);

  if (isLoading) {
    return (
      <section>
        <>
          <BackToActiveFires />
          <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />
        </>
        <LoadingState message={preparing ? "Preparing simulation..." : "Loading global response plan…"} />
      </section>
    );
  }

  if (error && showDemoData) {
    return (
      <section>
        <>
          <BackToActiveFires />
          <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />
        </>
        <ErrorState title={LOAD_ERROR_TITLE} message={error} onRetry={retry} />
      </section>
    );
  }

  const plan = response?.plan ?? null;
  const pendingCount = pendingIds.length;

  if (plan === null) {
    return (
      <section>
        <>
          <BackToActiveFires />
          <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />
        </>
        {coverage?.state === "generating" ? (
          <div className="global-response-plan-page__lifecycle">
            <LoadingState message={GENERATING_TITLE} />
            <p className="global-response-plan-page__lifecycle-detail">
              {confirmedFires(pendingCount)} {pendingCount === 1 ? "is" : "are"} being processed:{" "}
              {joinNames(pendingLabels)}.
            </p>
          </div>
        ) : (
          <EmptyState
            title={NO_GENERATION_TITLE}
            message={
              monitoringCount > 0
                ? `No fire is confirmed yet. ${monitoringCount} suspected fire${monitoringCount === 1 ? " is" : "s are"} being monitored - response plans are generated only for confirmed fires.`
                : NO_GENERATION_MESSAGE
            }
          />
        )}
      </section>
    );
  }

  return (
    <section>
      <BackToActiveFires />
      <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} />

      <div className="global-response-plan-page__layout">
        <div className="global-response-plan-page__details">
        <div className="global-response-plan-page__meta" aria-label="Run details">
          <span className="global-response-plan-page__updated">
            Last updated:{" "}
            <time dateTime={plan.completed_at ?? plan.started_at}>
              {formatLastUpdated(plan.completed_at ?? plan.started_at)}
            </time>
          </span>
          {plan.status !== "completed" ? (
            <span className="badge badge--warning">{RUN_STATUS_LABEL[plan.status]}</span>
          ) : null}
        </div>

        {coverage?.state === "updating" ? (
          <div className="global-response-plan-page__lifecycle global-response-plan-page__lifecycle--updating" role="status" aria-live="polite">
            <strong>{UPDATING_TITLE}</strong>{" "}
            <span className="global-response-plan-page__lifecycle-detail">
              Currently covers {coverage.covered_count} of {confirmedFires(coverage.eligible_count)}. Still being
              planned: {joinNames(pendingLabels)}.
            </span>
          </div>
        ) : null}

        <GlobalMetricsPanel metrics={plan.metrics} shortage={plan.shortage} />

        <EventGroupList
          events={plan.events}
          focusEventId={focusEventId}
          onFocusEvent={handleFocusEvent}
          eventLabels={eventLabels}
          eventPlaceNames={eventPlaceNames}
          targetLocations={targetLocations}
        />
        </div>

        <section
          aria-labelledby="global-response-map-heading"
          className="global-response-plan-page__section global-response-plan-page__section--map"
        >
          <h2 id="global-response-map-heading" className="global-response-plan-page__section-title">
            Map
          </h2>
          <MapView boundsPoints={boundsPoints} ariaLabel="Global Response Map">
            <GlobalResponseMapLayer layer={mapLayer} eventLabels={eventLabels} />
          </MapView>
        </section>
      </div>
    </section>
  );
}

/** Subtle breadcrumb-style link above the title (same pattern as the other screens). */
function BackToActiveFires() {
  return (
    <nav aria-label="Breadcrumb" className="global-response-plan-page__breadcrumb-nav">
      <Link to="/events" className="global-response-plan-page__breadcrumb">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10 3 5 8l5 5" />
        </svg>
        Back to Active Fires
      </Link>
    </nav>
  );
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"];

/** "19 Sept 2026, 13:33" - explicit so the wording does not vary with the browser's ICU data. */
function formatLastUpdated(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Not available";
  }
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getDate()} ${MONTHS[date.getMonth()]} ${date.getFullYear()}, ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function parseFocusEventId(raw: string | null): number | null {
  if (raw === null || !/^\d+$/.test(raw)) {
    return null;
  }
  const parsed = Number(raw);
  return parsed > 0 ? parsed : null;
}

/**
 * Every persisted origin/target/uncovered-target coordinate across every
 * materialized event - the only input to the shared map's auto-fit
 * viewport (see `FitBoundsToPoints`), so the map always frames this
 * generation's real data and never a hardcoded region.
 */
function buildBoundsPoints(events: GlobalEventPlan[]): LatLngPoint[] {
  const points: LatLngPoint[] = [];
  for (const event of events) {
    for (const action of event.actions) {
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
    for (const target of event.uncovered_targets) {
      if (target.latitude !== null && target.longitude !== null) {
        points.push({ lat: target.latitude, lng: target.longitude });
      }
    }
  }
  return points;
}

/**
 * Each event's own ACTIVE_FIRE target coordinates (searching its actions,
 * then its uncovered targets), for reverse-geocoding a fallback event name
 * when no news-evidence location name is on record - the same coordinate
 * source EventDetailsPage's own title uses (`fire_event.latitude/longitude`),
 * since a materialized GlobalEventPlan carries no FireEvent coordinate of
 * its own. An event with no ACTIVE_FIRE target coordinate at all is simply
 * omitted, never geocoded from a PREDICTED_RISK target or fabricated.
 */
function collectEventGeocodeTargets(events: GlobalEventPlan[]): GeocodeTarget[] {
  const targets: GeocodeTarget[] = [];
  for (const event of events) {
    const activeFireTarget =
      event.actions.map((action) => action.target).find((target) => target.target_type === "active_fire") ??
      event.uncovered_targets.find((target) => target.target_type === "active_fire");
    if (activeFireTarget && activeFireTarget.latitude !== null && activeFireTarget.longitude !== null) {
      targets.push({ id: event.fire_event_id, latitude: activeFireTarget.latitude, longitude: activeFireTarget.longitude });
    }
  }
  return targets;
}

/**
 * Every unique persisted action-target coordinate across every event
 * (mirroring ResponsePlanPage's own `collectGeocodeTargets`), for
 * `ResponseActions`' per-target group titles.
 */
function collectTargetGeocodeTargets(events: GlobalEventPlan[]): GeocodeTarget[] {
  const byId = new Map<number, GeocodeTarget>();
  for (const event of events) {
    for (const action of event.actions) {
      const { response_target_id: id, latitude, longitude } = action.target;
      if (latitude !== null && longitude !== null && !byId.has(id)) {
        byId.set(id, { id, latitude, longitude });
      }
    }
  }
  return Array.from(byId.values());
}
