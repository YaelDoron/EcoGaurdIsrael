import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useOperationsActivityDetail } from "../../hooks/useOperationsActivityDetail";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import { TimestampDisplay } from "../data/TimestampDisplay";
import { ErrorState } from "../feedback/ErrorState";
import { SeverityBadge } from "../status/SeverityBadge";
import { StatusBadge } from "../status/StatusBadge";
import { FIRE_DANGER_PRESENTATION } from "../status/fireDangerPresentation";
import { GLOBAL_PLANNING_RUN_STATUS_LABEL } from "./activityFeedPresentation";
import type {
  FireDangerAssessmentDetail,
  FireEventActivityDetails,
  FireSeverityDetails,
  GlobalPlanningRunDetails,
  NewsReportDetails,
  OperationsActivityDetailResponse,
  SatelliteHotspotDetails,
  WeatherConditionsDetails,
} from "../../types/operationsActivity";
import type { OperationsActivityFeedItem } from "../../types/operationsOverview";
import "./OperationsActivityDrawer.css";

export interface OperationsActivityDrawerProps {
  /** The currently selected feed item, or null when the drawer is closed. */
  selectedItem: OperationsActivityFeedItem | null;
  onClose: () => void;
}

/**
 * The Operations Overview's Activity Detail drawer (Task A8, Part 13):
 * fetches exactly the ONE selected item's full A5 detail via
 * `useOperationsActivityDetail`, and renders a type-specific summary built
 * only from that response's typed fields - never a raw JSON dump, and
 * never a bespoke full page duplicating Event Details/Global Response Plan.
 */
export function OperationsActivityDrawer({ selectedItem, onClose }: OperationsActivityDrawerProps) {
  const { data, isLoading, error, notFound, retry } = useOperationsActivityDetail({
    activityType: selectedItem?.activity_type ?? null,
    entityId: selectedItem?.entity_id ?? null,
    enabled: selectedItem !== null,
  });

  if (selectedItem === null) {
    return null;
  }

  return (
    <div className="operations-activity-drawer" role="dialog" aria-labelledby="operations-activity-drawer-title">
      <div className="operations-activity-drawer__header">
        <h2 id="operations-activity-drawer-title" className="operations-activity-drawer__title">
          {selectedItem.title}
        </h2>
        <button
          type="button"
          className="operations-activity-drawer__close"
          onClick={onClose}
          aria-label="Close activity details"
        >
          &times;
        </button>
      </div>

      <div className="operations-activity-drawer__body">
        {isLoading ? <p>Loading details&hellip;</p> : null}
        {notFound ? <ErrorState title="Not found" message="This activity is no longer available." /> : null}
        {error ? <ErrorState title="Unable to load details" message={error} onRetry={retry} /> : null}
        {!isLoading && !notFound && !error && data ? <DrawerContent detail={data} /> : null}
      </div>
    </div>
  );
}

function DrawerContent({ detail }: { detail: OperationsActivityDetailResponse }) {
  switch (detail.activity_type) {
    case "fire_danger":
      return <FireDangerDetailView detail={detail.details} />;
    case "satellite_hotspot":
      return <SatelliteHotspotDetailView detail={detail.details} />;
    case "news_report":
      return <NewsReportDetailView detail={detail.details} />;
    case "fire_event":
      return <FireEventDetailView detail={detail.details} />;
    case "fire_severity":
      return <FireSeverityDetailView detail={detail.details} />;
    case "global_planning_run":
      return <GlobalPlanningRunDetailView detail={detail.details} />;
    case "weather_conditions":
      return <WeatherConditionsDetailView detail={detail.details} />;
  }
}

function DetailRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="operations-activity-drawer__row">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function FireDangerDetailView({ detail }: { detail: FireDangerAssessmentDetail }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Area">{detail.area_name}</DetailRow>
      <DetailRow label="Level">
        {detail.status === "valid" && detail.level !== null ? (
          FIRE_DANGER_PRESENTATION[detail.level].label
        ) : (
          "Insufficient data"
        )}
      </DetailRow>
      <DetailRow label="Score">{detail.score !== null ? detail.score.toFixed(1) : "Not available"}</DetailRow>
      <DetailRow label="Assessed">
        <TimestampDisplay value={detail.assessed_at} />
      </DetailRow>
      <DetailRow label="Radius">{detail.radius_km.toFixed(1)} km</DetailRow>
      <DetailRow label="Methodology">
        {detail.methodology} (v{detail.methodology_version})
      </DetailRow>
    </dl>
  );
}

function SatelliteHotspotDetailView({ detail }: { detail: SatelliteHotspotDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Confidence">{detail.confidence ?? "Not available"}</DetailRow>
      <DetailRow label="Fire radiative power">{detail.frp !== null ? detail.frp.toFixed(1) : "Not available"}</DetailRow>
      <DetailRow label="Brightness">{detail.brightness !== null ? detail.brightness.toFixed(1) : "Not available"}</DetailRow>
      <DetailRow label="Satellite">{detail.satellite ?? "Not available"}</DetailRow>
      <DetailRow label="Detected">
        <TimestampDisplay value={detail.detected_at} />
      </DetailRow>
      <DetailRow label="Coordinates">
        <CoordinateDisplay latitude={detail.latitude} longitude={detail.longitude} />
      </DetailRow>
    </dl>
  );
}

function NewsReportDetailView({ detail }: { detail: NewsReportDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Source">{detail.source_feed}</DetailRow>
      <DetailRow label="Headline">{detail.title}</DetailRow>
      <DetailRow label="Summary">{detail.summary}</DetailRow>
      <DetailRow label="Published">
        <TimestampDisplay value={detail.published_at} />
      </DetailRow>
      <DetailRow label="Location">{detail.location_name ?? "Not available"}</DetailRow>
    </dl>
  );
}

function FireEventDetailView({ detail }: { detail: FireEventActivityDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Location">{detail.location_name ?? "Location unavailable"}</DetailRow>
      <DetailRow label="Status">
        <StatusBadge status={detail.status} />
      </DetailRow>
      <DetailRow label="Confidence">{Math.round(detail.detection_confidence * 100)}%</DetailRow>
      <DetailRow label="Latest severity">
        <SeverityBadge level={detail.latest_severity?.level ?? null} />
      </DetailRow>
      <DetailRow label="Opened">
        <TimestampDisplay value={detail.created_at} />
      </DetailRow>
      <DetailRow label="Detected">
        <TimestampDisplay value={detail.detected_at} />
      </DetailRow>
      <DetailRow label="Coordinates">
        <CoordinateDisplay latitude={detail.latitude} longitude={detail.longitude} />
      </DetailRow>
      <div className="operations-activity-drawer__actions">
        <Link to={`/events/${detail.fire_event_id}`} className="operations-activity-drawer__link">
          View Fire Event
        </Link>
      </div>
    </dl>
  );
}

function FireSeverityDetailView({ detail }: { detail: FireSeverityDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Level">
        <SeverityBadge level={detail.level} />
      </DetailRow>
      <DetailRow label="Score">{detail.score !== null ? detail.score.toFixed(1) : "Not available"}</DetailRow>
      <DetailRow label="Assessed">
        <TimestampDisplay value={detail.assessed_at} />
      </DetailRow>
      <DetailRow label="Dominant land cover">{detail.vegetation_dominant_land_cover ?? "Not available"}</DetailRow>
      <div className="operations-activity-drawer__actions">
        <Link to={`/events/${detail.fire_event_id}`} className="operations-activity-drawer__link">
          View Fire Event
        </Link>
      </div>
    </dl>
  );
}

function WeatherConditionsDetailView({ detail }: { detail: WeatherConditionsDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Area">{detail.area_name}</DetailRow>
      <DetailRow label="Fire danger level">{FIRE_DANGER_PRESENTATION[detail.fire_danger_level].label}</DetailRow>
      <DetailRow label="Assessed">
        <TimestampDisplay value={detail.assessed_at} />
      </DetailRow>
      {detail.readings.map((reading) => (
        <DetailRow key={reading.observation_id} label={reading.station_name}>
          {reading.temperature !== null ? `${Math.round(reading.temperature)}°C` : "Not available"} ·{" "}
          {reading.relative_humidity !== null ? `${Math.round(reading.relative_humidity)}% humidity` : "Not available"} ·{" "}
          {reading.wind_speed !== null ? `${Math.round(reading.wind_speed)} km/h wind` : "Not available"}
          {reading.wind_gust !== null ? ` (gust ${Math.round(reading.wind_gust)})` : ""}
          <br />
          <TimestampDisplay value={reading.observed_at} />
        </DetailRow>
      ))}
    </dl>
  );
}

function GlobalPlanningRunDetailView({ detail }: { detail: GlobalPlanningRunDetails }) {
  return (
    <dl className="operations-activity-drawer__fields">
      <DetailRow label="Status">{GLOBAL_PLANNING_RUN_STATUS_LABEL[detail.status]}</DetailRow>
      <DetailRow label="Fire events">{detail.fire_event_ids.length}</DetailRow>
      <DetailRow label="Started">
        <TimestampDisplay value={detail.started_at} />
      </DetailRow>
      <DetailRow label="Completed">
        <TimestampDisplay value={detail.completed_at} />
      </DetailRow>
      <DetailRow label="Coverage score">
        {detail.coverage_score !== null ? detail.coverage_score.toFixed(1) : "Not available"}
      </DetailRow>
      {detail.response_plan_ids.length > 0 ? (
        <div className="operations-activity-drawer__actions">
          {detail.response_plan_ids.map((planId) => (
            <Link key={planId} to={`/plans/${planId}`} className="operations-activity-drawer__link">
              View Response Plan #{planId}
            </Link>
          ))}
        </div>
      ) : null}
    </dl>
  );
}
