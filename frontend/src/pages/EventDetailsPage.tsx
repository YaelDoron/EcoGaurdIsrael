import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CoordinateDisplay } from "../components/data/CoordinateDisplay";
import { MetricCard } from "../components/data/MetricCard";
import { TimestampDisplay } from "../components/data/TimestampDisplay";
import { DetectionEvidencePanel } from "../components/fire-events/DetectionEvidencePanel";
import { SEVERITY_STATUS_CAPTION } from "../components/fire-events/severityStatusCaption";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { PageHeader } from "../components/layout/PageHeader";
import { FireEventMarker } from "../components/map/FireEventMarker";
import { LayerControls } from "../components/map/LayerControls";
import { MapView } from "../components/map/MapView";
import { ResponseTargetLayer } from "../components/map/ResponseTargetLayer";
import { SpreadLayer } from "../components/map/SpreadLayer";
import { StationLayer } from "../components/map/StationLayer";
import { translateIfUntranslated } from "../components/map/stationTranslations";
import type { LatLngPoint, LayerToggle, LayerVisibility } from "../components/map/mapTypes";
import { SeverityBadge } from "../components/status/SeverityBadge";
import { StatusBadge } from "../components/status/StatusBadge";
import { useEventDetails } from "../hooks/useEventDetails";
import { useTargetLocationNames, type GeocodeTarget } from "../hooks/useTargetLocationNames";
import type { DangerAssessment, EventDetailsResult, SpreadPrediction } from "../types/eventDetails";
import "./EventDetailsPage.css";

// Shown while loading so the header never flashes the raw "Event #id" before the
// location-based title resolves.
const LOADING_TITLE = "Loading Event…";

const NO_GEOCODE_TARGETS: GeocodeTarget[] = [];

const DANGER_LEVEL_LABEL: Record<NonNullable<DangerAssessment["level"]>, string> = {
  low: "Low",
  moderate: "Moderate",
  high: "High",
  very_high: "Very high",
  extreme: "Extreme",
};

// ml_assessment.model_score is a model-estimated score from the
// synthetic-trained Logistic Regression V3 classifier, not a calibrated
// real-world probability of wildfire occurrence - see
// backend/docs/fire_detection_runtime_ml.md, "Probability interpretation".
const AI_MODEL_SCORE_HELP_TEXT = "Experimental score produced by the Fire Detection ML model.";
const AI_MODEL_SCORE_UNAVAILABLE_LABEL = "Unavailable";

const SPREAD_STATUS_LABEL: Record<SpreadPrediction["status"], string> = {
  valid: "Valid",
  insufficient_data: "Insufficient data",
  inactive_event: "Inactive event",
};

/**
 * Operational wording for one horizon's spread prediction. A "valid" run with
 * no predicted cells means the model predicts no spread - shown as such
 * rather than as a raw "Valid (0 predicted cells)". Other statuses keep
 * their own (already readable) label.
 */
function describeSpread(prediction: SpreadPrediction): { value: string; helperText?: string } {
  if (prediction.status !== "valid") {
    return { value: SPREAD_STATUS_LABEL[prediction.status] };
  }
  const cells = prediction.cells.length;
  if (cells === 0) {
    return { value: "No spread predicted" };
  }
  return { value: "Spread predicted", helperText: `${cells} predicted cell${cells === 1 ? "" : "s"}` };
}

/** A valid run that predicts no spread: a negative/fallback state shown quietly. */
function isNegativeSpread(prediction: SpreadPrediction): boolean {
  return prediction.status === "valid" && prediction.cells.length === 0;
}

function ChevronLeftIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 3 5 8l5 5" />
    </svg>
  );
}

/** Subtle breadcrumb-style link above the title (replaces the boxed Back button). */
function BackToActiveEvents() {
  return (
    <nav aria-label="Breadcrumb" className="event-details-page__breadcrumb-nav">
      <Link to="/events" className="event-details-page__breadcrumb" aria-label="Back to Active Events">
        <ChevronLeftIcon />
        Active Events
      </Link>
    </nav>
  );
}

function ClipboardIcon() {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="2.5" width="10" height="12" rx="1.5" />
      <path d="M6 1.5h4v2H6zM5.5 7.5h5M5.5 10.5h5" />
    </svg>
  );
}

function parseFireEventId(raw: string | undefined): number {
  return raw ? Number(raw) : Number.NaN;
}

function buildLayerToggles(data: EventDetailsResult): LayerToggle[] {
  const spreadCellCount = data.spread_predictions.reduce((sum, prediction) => sum + prediction.cells.length, 0);
  return [
    { id: "spread", label: "Predicted spread", count: spreadCellCount },
    { id: "targets", label: "Response targets", count: data.targets.length },
    { id: "stations", label: "Fire stations", count: data.stations.length },
  ];
}

function buildBoundsPoints(data: EventDetailsResult): LatLngPoint[] {
  const points: LatLngPoint[] = [{ lat: data.fire_event.latitude, lng: data.fire_event.longitude }];
  for (const prediction of data.spread_predictions) {
    for (const cell of prediction.cells) {
      points.push({ lat: cell.latitude, lng: cell.longitude });
    }
  }
  for (const target of data.targets) {
    points.push({ lat: target.latitude, lng: target.longitude });
  }
  // Only stations actually serving this event: framing every known station
  // would zoom the map out to the whole country.
  const relevantStationIds = new Set<string>();
  for (const summary of data.station_summaries) {
    if (summary.current_global_plan_allocations.length > 0) {
      relevantStationIds.add(summary.station_id);
    }
  }
  for (const action of data.current_response_plan?.actions ?? []) {
    relevantStationIds.add(action.station_id);
  }
  for (const station of data.stations) {
    if (relevantStationIds.has(station.station_id)) {
      points.push({ lat: station.latitude, lng: station.longitude });
    }
  }
  return points;
}

/**
 * US 6.2's real Event Details page: fetches
 * GET /api/v1/fire-events/{fireEventId}/details via useEventDetails() and
 * renders the FireEvent's current state, its latest assessments, its
 * targets/resources, and an interactive map - all straight from the API
 * response. Grouping/counting/coloring here is presentation only; it never
 * calculates or infers a wildfire state the backend did not send.
 */
export function EventDetailsPage() {
  const { fireEventId: fireEventIdParam } = useParams<{ fireEventId: string }>();
  const fireEventId = parseFireEventId(fireEventIdParam);
  const { data, isLoading, loadError, notFound, refresh } = useEventDetails(fireEventId);
  // Fallback place name from the fire's own coordinates (used when the event's
  // news evidence has no location name).
  const geocodedPlaces = useTargetLocationNames(
    data
      ? [{ id: data.fire_event.fire_event_id, latitude: data.fire_event.latitude, longitude: data.fire_event.longitude }]
      : NO_GEOCODE_TARGETS,
  );
  const [layerVisibility, setLayerVisibility] = useState<LayerVisibility>({});

  const boundsPoints = useMemo(() => (data ? buildBoundsPoints(data) : []), [data]);
  const layerToggles = useMemo(() => (data ? buildLayerToggles(data) : []), [data]);

  const toggleLayer = (layerId: string) => {
    setLayerVisibility((previous) => ({ ...previous, [layerId]: !(previous[layerId] ?? true) }));
  };

  const pageTitle = Number.isFinite(fireEventId) ? `Event #${fireEventId}` : "Event Details";

  if (notFound) {
    return (
      <section>
        <BackToActiveEvents />
      <PageHeader title={pageTitle} />
        <ErrorState
          title="Event Not Found"
          message="No wildfire event exists with this ID. It may have been resolved, dismissed, or never existed."
        />
      </section>
    );
  }

  if (isLoading) {
    return (
      <section>
        <BackToActiveEvents />
      <PageHeader title={LOADING_TITLE} />
        <LoadingState message="Loading event details…" />
      </section>
    );
  }

  if (loadError || !data) {
    return (
      <section>
        <BackToActiveEvents />
      <PageHeader title={pageTitle} />
        <ErrorState title="Unable to load event details." message="Please try again." onRetry={refresh} />
      </section>
    );
  }

  const {
    fire_event: fireEvent,
    severity,
    ml_assessment: mlAssessment,
    danger,
    detection_evidence: detectionEvidence,
    spread_predictions: spreadPredictions,
    targets,
    station_summaries: stationSummaries,
    current_response_plan: currentPlan,
  } = data;
  // The deterministic rule-based detection score. `ml_assessment.rule_confidence`
  // (when a row exists) and `fire_event.detection_confidence` are the same
  // rule result in SHADOW mode (FireEvent.detection_confidence is always the
  // rule decision's own confidence) - this only prefers the ML-assessment
  // copy when present, it never changes which value is authoritative.
  const ruleScore = (mlAssessment?.rule_confidence ?? fireEvent.detection_confidence).toFixed(2);
  // AI Model Score: omitted entirely when there is no ml_assessment row at
  // all (legacy FireEvent, or RULE_ONLY mode); "Unavailable" (never 0) when
  // a row exists but the classifier could not produce a score.
  const aiModelScore =
    mlAssessment && mlAssessment.available && mlAssessment.model_score !== null
      ? mlAssessment.model_score.toFixed(2)
      : AI_MODEL_SCORE_UNAVAILABLE_LABEL;
  const severityCaption = severity && severity.status !== "valid" ? SEVERITY_STATUS_CAPTION[severity.status] : null;
  // Title: the event's persisted location name (news evidence - the backend
  // ingestion pipeline translates this to English before it is ever saved,
  // so it is displayed as-is here, never re-translated on the frontend),
  // else a place reverse-geocoded from its coordinates (already English),
  // else a generic title with the id.
  const newsLocationName = translateIfUntranslated(
    detectionEvidence.news.find((item) => item.location_name)?.location_name ?? null,
  );
  const placeName = newsLocationName ?? geocodedPlaces[fireEvent.fire_event_id] ?? null;
  const headerTitle = placeName ? `${placeName} Wildfire Event` : `Wildfire Event #${fireEvent.fire_event_id}`;
  // A still-unverified event must not read as a confirmed critical one.
  const isUnverifiedCritical = fireEvent.status === "suspected" && severity?.level === "critical";
  // No plan is generated for a resolved/dismissed event: only a still-active
  // event (suspected/confirmed) with no plan YET is "pending" - the button
  // stays visible in a disabled/loading state rather than disappearing, so it
  // doesn't look like it never renders on a page the operator just opened.
  const isPlanStatusPending = fireEvent.status === "suspected" || fireEvent.status === "confirmed";

  return (
    <section className="event-details-page">
      <BackToActiveEvents />
      <PageHeader title={headerTitle} />

      <p className="event-details-page__as-of">
        Data as of: <TimestampDisplay value={data.as_of} />
        {/* The title already carries the event id when no place name is known. */}
        {placeName ? <span className="event-details-page__event-badge">Event #{fireEvent.fire_event_id}</span> : null}
      </p>

      <div className="event-details-page__layout">
        <section
          aria-labelledby="map-heading"
          className="event-details-page__section event-details-page__section--map"
        >
          <h2 id="map-heading" className="event-details-page__section-title">
            Map
          </h2>
          <LayerControls layers={layerToggles} visibility={layerVisibility} onToggle={toggleLayer} />
          <MapView boundsPoints={boundsPoints} ariaLabel={`Map of Event #${fireEvent.fire_event_id}`}>
            <FireEventMarker fireEvent={fireEvent} locationName={placeName} />
            {(layerVisibility.spread ?? true) ? <SpreadLayer predictions={spreadPredictions} /> : null}
            {(layerVisibility.targets ?? true) ? (
              <ResponseTargetLayer targets={targets} eventLocationName={placeName} />
            ) : null}
            {(layerVisibility.stations ?? true) ? (
              <StationLayer stations={data.stations} stationSummaries={stationSummaries} />
            ) : null}
          </MapView>
        </section>

        <div className="event-details-page__cards">
          <section aria-labelledby="fire-event-heading" className="event-details-page__section">
            <div className="event-details-page__card-header">
              <h2 id="fire-event-heading" className="event-details-page__section-title">
                Fire Event
              </h2>
              {currentPlan && currentPlan.actions.length > 0 ? (
                <Link
                  to={`/events/${fireEvent.fire_event_id}/plan`}
                  className="event-details-page__action event-details-page__action--primary"
                  aria-label="View Current Response Plan"
                >
                  <ClipboardIcon />
                  Response Plan
                </Link>
              ) : currentPlan ? (
                // A plan was generated but allocated zero resources (e.g. the
                // global optimizer found "No feasible assignments"). This is
                // a real, current plan - not a missing one - so the button
                // stays active/clickable rather than reverting to the
                // "pending" state, letting the operator click through and
                // see the warning on the plan details page.
                <Link
                  to={`/events/${fireEvent.fire_event_id}/plan`}
                  className="event-details-page__action event-details-page__action--warning"
                  aria-label="View Response Plan (No Resources Available)"
                >
                  <ClipboardIcon />
                  No Resources Available
                </Link>
              ) : isPlanStatusPending ? (
                <span
                  className="event-details-page__action event-details-page__action--primary event-details-page__action--pending"
                  aria-disabled="true"
                >
                  <ClipboardIcon />
                  Response Plan pending…
                </span>
              ) : null}
            </div>
            <dl className="event-details-page__facts event-details-page__facts--grid">
              <div className="event-details-page__fact">
                <dt>Status</dt>
                <dd>
                  <StatusBadge status={fireEvent.status} />
                </dd>
              </div>
              <div className="event-details-page__fact">
                <dt>Coordinates</dt>
                <dd>
                  <CoordinateDisplay latitude={fireEvent.latitude} longitude={fireEvent.longitude} />
                </dd>
              </div>
              <div className="event-details-page__fact">
                <dt>Detected</dt>
                <dd>
                  <TimestampDisplay value={fireEvent.detected_at} />
                </dd>
              </div>
              <div className="event-details-page__fact">
                <dt>Rule Score</dt>
                <dd>{ruleScore}</dd>
              </div>
              {mlAssessment ? (
                <div className="event-details-page__fact">
                  <dt title={AI_MODEL_SCORE_HELP_TEXT}>AI Model Score</dt>
                  <dd>{aiModelScore}</dd>
                </div>
              ) : null}
            </dl>
          </section>

          <section aria-labelledby="assessments-heading" className="event-details-page__section">
            <h2 id="assessments-heading" className="event-details-page__section-title">
              Assessments
            </h2>
            <dl className="event-details-page__facts">
              <div className="event-details-page__fact">
                <dt>Severity</dt>
                <dd>
                  {isUnverifiedCritical ? (
                    <>
                      <span className="badge badge--warning">Pending verification / Critical</span>
                      <span className="event-details-page__note">
                        Severity is critical, but this event is still suspected and not yet verified.
                      </span>
                    </>
                  ) : (
                    <SeverityBadge level={severity?.level ?? null} />
                  )}
                  {severityCaption ? <span className="event-details-page__note">{severityCaption}</span> : null}
                </dd>
              </div>
              {danger && (danger.level !== null || danger.score !== null) ? (
                <div className="event-details-page__fact">
                  <dt>Danger</dt>
                  <dd>
                    <span>{danger.level ? DANGER_LEVEL_LABEL[danger.level] : "Not available"}</span>
                    {danger.score !== null ? (
                      <span className="event-details-page__note">Danger score: {danger.score.toFixed(1)}</span>
                    ) : null}
                  </dd>
                </div>
              ) : null}
            </dl>
          </section>

          <div className="event-details-page__pair">
            <section aria-labelledby="detection-evidence-heading" className="event-details-page__section">
              <h2 id="detection-evidence-heading" className="event-details-page__section-title">
                Detection Evidence
              </h2>
              <DetectionEvidencePanel evidence={detectionEvidence} />
            </section>

            <section aria-labelledby="spread-heading" className="event-details-page__section">
              <h2 id="spread-heading" className="event-details-page__section-title">
                Spread Prediction
              </h2>
              {spreadPredictions.length === 0 ? (
                <EmptyState
                  title="No spread prediction"
                  message="No spread prediction has been generated for this event yet."
                />
              ) : (
                <div className="event-details-page__metrics event-details-page__metrics--stacked">
                  {spreadPredictions.map((prediction) => (
                    <div
                      key={prediction.horizon_minutes}
                      className={
                        isNegativeSpread(prediction)
                          ? "event-details-page__spread-horizon event-details-page__spread-horizon--empty"
                          : "event-details-page__spread-horizon"
                      }
                    >
                      <MetricCard label={`${prediction.horizon_minutes} min horizon`} {...describeSpread(prediction)} />
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </section>
  );
}
