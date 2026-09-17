import { Link } from "react-router-dom";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import { TimestampDisplay } from "../data/TimestampDisplay";
import { SeverityBadge } from "../status/SeverityBadge";
import { StatusBadge } from "../status/StatusBadge";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import { SEVERITY_STATUS_CAPTION } from "./severityStatusCaption";
import "./ActiveFireEventCard.css";

export interface ActiveFireEventCardProps {
  event: ActiveFireEvent;
}

/**
 * One active FireEvent's presentation card. Pure display: formats the
 * fields it is given (confidence as a percentage, coordinates, timestamps)
 * but never recalculates status/severity - those always come straight
 * from the API response.
 */
export function ActiveFireEventCard({ event }: ActiveFireEventCardProps) {
  const confidencePercent = Math.round(event.detection_confidence * 100);
  const severity = event.severity;
  const severityCaption = severity && severity.status !== "valid" ? SEVERITY_STATUS_CAPTION[severity.status] : null;

  return (
    <article className="fire-event-card">
      <div className="fire-event-card__header">
        <h3 className="fire-event-card__title">Event #{event.fire_event_id}</h3>
        <StatusBadge status={event.status} />
      </div>

      <dl className="fire-event-card__details">
        <div className="fire-event-card__row">
          <dt>Severity</dt>
          <dd>
            <SeverityBadge level={severity?.level ?? null} />
            {severity && severity.status === "valid" && severity.score !== null ? (
              <span className="fire-event-card__severity-score">Severity score: {severity.score.toFixed(1)}</span>
            ) : null}
            {severityCaption ? <span className="fire-event-card__severity-caption">{severityCaption}</span> : null}
          </dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Confidence</dt>
          <dd>{confidencePercent}%</dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Coordinates</dt>
          <dd>
            <CoordinateDisplay latitude={event.latitude} longitude={event.longitude} />
          </dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Detected</dt>
          <dd>
            <TimestampDisplay value={event.detected_at} />
          </dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Updated</dt>
          <dd>
            <TimestampDisplay value={event.updated_at} />
          </dd>
        </div>
      </dl>

      <Link to={`/events/${event.fire_event_id}`} className="fire-event-card__link">
        View Event
      </Link>
    </article>
  );
}
