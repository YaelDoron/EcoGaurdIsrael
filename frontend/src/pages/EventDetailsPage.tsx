import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CoordinateDisplay } from "../components/data/CoordinateDisplay";
import { MetricCard } from "../components/data/MetricCard";
import { TimestampDisplay } from "../components/data/TimestampDisplay";
import { SEVERITY_STATUS_CAPTION } from "../components/fire-events/severityStatusCaption";
import { EmptyState } from "../components/feedback/EmptyState";
import { ErrorState } from "../components/feedback/ErrorState";
import { LoadingState } from "../components/feedback/LoadingState";
import { PageHeader } from "../components/layout/PageHeader";
import { FireEventMarker } from "../components/map/FireEventMarker";
import { LayerControls } from "../components/map/LayerControls";
import { MapView } from "../components/map/MapView";
import { OperationalLayer, RESOURCE_STATUS_LABEL } from "../components/map/OperationalLayer";
import { ResponseTargetLayer, TARGET_TYPE_LABEL } from "../components/map/ResponseTargetLayer";
import { SpreadLayer } from "../components/map/SpreadLayer";
import { StationLayer } from "../components/map/StationLayer";
import type { LatLngPoint, LayerToggle, LayerVisibility } from "../components/map/mapTypes";
import { SeverityBadge } from "../components/status/SeverityBadge";
import { StatusBadge } from "../components/status/StatusBadge";
import { useEventDetails } from "../hooks/useEventDetails";
import type { DangerAssessment, EventDetailsResult } from "../types/eventDetails";
import "./EventDetailsPage.css";

const PAGE_DESCRIPTION = "Full details and interactive map for this wildfire event.";

const DANGER_LEVEL_LABEL: Record<NonNullable<DangerAssessment["level"]>, string> = {
  low: "Low",
  moderate: "Moderate",
  high: "High",
  very_high: "Very high",
  extreme: "Extreme",
};

function parseFireEventId(raw: string | undefined): number {
  return raw ? Number(raw) : Number.NaN;
}

function buildLayerToggles(data: EventDetailsResult): LayerToggle[] {
  const spreadCellCount = data.spread_predictions.reduce((sum, prediction) => sum + prediction.cells.length, 0);
  return [
    { id: "spread", label: "Predicted spread", count: spreadCellCount },
    { id: "targets", label: "Response targets", count: data.targets.length },
    { id: "stations", label: "Fire stations", count: data.stations.length },
    { id: "resources", label: "Resources", count: data.resources.length },
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
  for (const station of data.stations) {
    points.push({ lat: station.latitude, lng: station.longitude });
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
  const { data, isLoading, isRefreshing, loadError, refreshError, notFound, refresh } = useEventDetails(fireEventId);
  const [layerVisibility, setLayerVisibility] = useState<LayerVisibility>({});

  const boundsPoints = useMemo(() => (data ? buildBoundsPoints(data) : []), [data]);
  const layerToggles = useMemo(() => (data ? buildLayerToggles(data) : []), [data]);

  const toggleLayer = (layerId: string) => {
    setLayerVisibility((previous) => ({ ...previous, [layerId]: !(previous[layerId] ?? true) }));
  };

  const backAction = (
    <Link to="/events" className="event-details-page__action event-details-page__action--secondary">
      Back to Active Events
    </Link>
  );

  const pageTitle = Number.isFinite(fireEventId) ? `Event #${fireEventId}` : "Event Details";

  if (notFound) {
    return (
      <section>
        <PageHeader title={pageTitle} description={PAGE_DESCRIPTION} actions={backAction} />
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
        <PageHeader title={pageTitle} description={PAGE_DESCRIPTION} actions={backAction} />
        <LoadingState message="Loading event details…" />
      </section>
    );
  }

  if (loadError || !data) {
    return (
      <section>
        <PageHeader title={pageTitle} description={PAGE_DESCRIPTION} actions={backAction} />
        <ErrorState title="Unable to load event details." message="Please try again." onRetry={refresh} />
      </section>
    );
  }

  const { fire_event: fireEvent, severity, danger, targets, resources, current_response_plan: currentPlan } = data;
  const confidencePercent = Math.round(fireEvent.detection_confidence * 100);
  const severityCaption = severity && severity.status !== "valid" ? SEVERITY_STATUS_CAPTION[severity.status] : null;

  return (
    <section>
      <PageHeader
        title={pageTitle}
        description={PAGE_DESCRIPTION}
        actions={
          <>
            {backAction}
            {currentPlan ? (
              <Link
                to={`/events/${fireEvent.fire_event_id}/plan`}
                className="event-details-page__action event-details-page__action--primary"
              >
                View Current Response Plan
              </Link>
            ) : null}
            <button
              type="button"
              className="event-details-page__refresh"
              onClick={refresh}
              disabled={isRefreshing}
            >
              {isRefreshing ? "Refreshing…" : "Refresh"}
            </button>
          </>
        }
      />

      <p className="event-details-page__as-of">
        Data as of: <TimestampDisplay value={data.as_of} />
      </p>

      {refreshError ? (
        <p role="alert" className="event-details-page__refresh-error">
          {refreshError}
        </p>
      ) : null}

      <section aria-labelledby="fire-event-heading" className="event-details-page__section">
        <h2 id="fire-event-heading" className="event-details-page__section-title">
          Fire Event
        </h2>
        <div className="event-details-page__metrics">
          <MetricCard label="Fire Event ID" value={fireEvent.fire_event_id} />
          <MetricCard label="Detection Confidence" value={`${confidencePercent}%`} />
        </div>
        <dl className="event-details-page__facts">
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
            <dt>Updated</dt>
            <dd>
              <TimestampDisplay value={fireEvent.updated_at} />
            </dd>
          </div>
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
              <SeverityBadge level={severity?.level ?? null} />
              {severity && severity.status === "valid" && severity.score !== null ? (
                <span className="event-details-page__note">Severity score: {severity.score.toFixed(1)}</span>
              ) : null}
              {severityCaption ? <span className="event-details-page__note">{severityCaption}</span> : null}
            </dd>
          </div>
          <div className="event-details-page__fact">
            <dt>Danger</dt>
            <dd>
              {danger ? (
                <>
                  <span>{danger.level ? DANGER_LEVEL_LABEL[danger.level] : "Not available"}</span>
                  {danger.score !== null ? (
                    <span className="event-details-page__note">Danger score: {danger.score.toFixed(1)}</span>
                  ) : null}
                </>
              ) : (
                <span className="event-details-page__not-available">Not available</span>
              )}
            </dd>
          </div>
        </dl>
      </section>

      <section aria-labelledby="targets-heading" className="event-details-page__section">
        <h2 id="targets-heading" className="event-details-page__section-title">
          Response Targets
        </h2>
        {targets.length === 0 ? (
          <EmptyState
            title="No response targets"
            message="No response targets have been generated for this event yet."
          />
        ) : (
          <ul className="event-details-page__list">
            {targets.map((target) => (
              <li key={target.target_order} className="event-details-page__list-item">
                <span className="event-details-page__list-item-title">
                  {TARGET_TYPE_LABEL[target.target_type] ?? target.target_type}
                </span>
                <span className="event-details-page__note">Priority: {target.priority_score.toFixed(2)}</span>
                {target.prediction_horizon_minutes !== null ? (
                  <span className="event-details-page__note">
                    Horizon: {target.prediction_horizon_minutes} min
                  </span>
                ) : null}
                <CoordinateDisplay latitude={target.latitude} longitude={target.longitude} />
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="resources-heading" className="event-details-page__section">
        <h2 id="resources-heading" className="event-details-page__section-title">
          Firefighting Resources
        </h2>
        {resources.length === 0 ? (
          <EmptyState title="No firefighting resources" message="No firefighting resources are on record." />
        ) : (
          <ul className="event-details-page__list">
            {resources.map((resource) => (
              <li key={resource.resource_id} className="event-details-page__list-item">
                <span className="event-details-page__list-item-title">{resource.resource_id}</span>
                <span className="event-details-page__note">Station: {resource.station_id}</span>
                <span className="event-details-page__note">{RESOURCE_STATUS_LABEL[resource.status]}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section aria-labelledby="map-heading" className="event-details-page__section">
        <h2 id="map-heading" className="event-details-page__section-title">
          Map
        </h2>
        <LayerControls layers={layerToggles} visibility={layerVisibility} onToggle={toggleLayer} />
        <MapView boundsPoints={boundsPoints} ariaLabel={`Map of Event #${fireEvent.fire_event_id}`}>
          <FireEventMarker fireEvent={fireEvent} />
          {(layerVisibility.spread ?? true) ? <SpreadLayer predictions={data.spread_predictions} /> : null}
          {(layerVisibility.targets ?? true) ? <ResponseTargetLayer targets={targets} /> : null}
          {(layerVisibility.stations ?? true) ? <StationLayer stations={data.stations} /> : null}
          {(layerVisibility.resources ?? true) ? (
            <OperationalLayer resources={resources} stations={data.stations} />
          ) : null}
        </MapView>
      </section>
    </section>
  );
}
