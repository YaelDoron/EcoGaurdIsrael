import { Link } from "react-router-dom";
import { translateIfUntranslated } from "../map/stationTranslations";
import { TimestampDisplay } from "../data/TimestampDisplay";
import { SeverityBadge } from "../status/SeverityBadge";
import { StatusBadge } from "../status/StatusBadge";
import type { ActiveFireEvent, ActiveFireEventMLSummary } from "../../types/activeFireEvents";
import { SEVERITY_STATUS_CAPTION } from "./severityStatusCaption";
import {
  AI_LIKELIHOOD_HELP_TEXT,
  SUSPECTED_MONITORING_LABEL,
  formatLikelihood,
  isAiHybridMode,
} from "../status/fireDetectionPresentation";
import "./ActiveFireEventCard.css";

export interface ActiveFireEventCardProps {
  event: ActiveFireEvent;
}

// AI Model Score is an experimental Logistic Regression V3 score, never a
// calibrated real-world probability - see
// backend/docs/fire_detection_runtime_ml.md, "Probability interpretation".
// "-" (not "AI 0.00") is used both when there is no ml_summary row at all
// (legacy/RULE_ONLY event) and when a row exists but the classifier could
// not produce a score - one consistent convention for "no AI score to show".
const AI_SCORE_UNAVAILABLE = "-";

function formatAiScore(mlSummary: ActiveFireEventMLSummary | null): string {
  if (mlSummary && mlSummary.available && mlSummary.model_score !== null) {
    return mlSummary.model_score.toFixed(2);
  }
  return AI_SCORE_UNAVAILABLE;
}

/**
 * One active FireEvent's presentation card. Pure display: formats the
 * fields it is given (Rule/AI scores, timestamps) but never recalculates
 * status/severity - those always come straight from the API response.
 *
 * ML Task 7: the old single "Confidence: 80%" row is now a compact
 * "Rule 0.88 | AI 0.90" row - Rule from the unchanged `detection_confidence`
 * field (decimal, matching Event Details' terminology instead of a
 * percentage), AI from the lightweight `ml_summary` the active-events API
 * now also returns (batched server-side - see ActiveFireEventsService - so
 * this card never issues its own per-event details request).
 *
 * This card is a DASHBOARD SUMMARY only - full detail (coordinates,
 * updated time, evidence, etc.) already lives on Event Details
 * (`/events/:fireEventId`, unchanged by this component). Keeping only the
 * most operationally useful fields here (status, severity, confidence,
 * opened time, View Event) is what lets the Active Fires column sit
 * beside the map without forcing a large blank area under it in the
 * normal 1-2 fire demo case.
 *
 * "Opened" shows `created_at` (when EcoGuard actually persisted this
 * FireEvent), NOT `detected_at` (the earliest correlated evidence's own,
 * possibly-earlier, source observation time) - showing the evidence time
 * here previously made a fire that became visible at e.g. 11:46 appear to
 * say "Detected: 11:44", implying an apparent creation time that was
 * actually just older evidence. The full evidence-vs-opened distinction
 * remains available on Event Details.
 *
 * Task A8, Part 9: a CONFIRMED fire with the latest persisted Severity at
 * HIGH or CRITICAL gets a stronger visual border/emphasis - pure
 * presentation over two already-persisted facts (`event.status` +
 * `event.severity?.level`), never a new business score, and never applied
 * merely from a SUSPECTED event's severity (status must also be confirmed).
 */
export function ActiveFireEventCard({ event }: ActiveFireEventCardProps) {
  // Rule score: the existing Fire Detection rule-based score, unchanged
  // field/semantics (detection_confidence) - only the presentation (decimal,
  // not percent) and label ("Rule") match Event Details' terminology now.
  const ruleScore = event.detection_confidence.toFixed(2);
  const aiScore = formatAiScore(event.ml_summary);
  // AI Hybrid mode: the event's PERSISTED decision mode (ml_summary.mode) decides the presentation - the AI assessment is then THE
  // detection value; the rule score is diagnostics only and is not shown. Legacy events (rule_only / shadow / hybrid, incl. old
  // persisted rows) keep the "Rule | AI" row.
  const isAiMode = isAiHybridMode(event.ml_summary?.mode);
  const aiLikelihood = event.ml_summary && event.ml_summary.available ? formatLikelihood(event.ml_summary.model_score) : "Unavailable";
  // SUSPECTED = monitored for more evidence; it is never response-planned, so say so instead of leaving it unexplained.
  const isMonitoringOnly = event.status === "suspected";
  const severity = event.severity;
  const severityCaption = severity && severity.status !== "valid" ? SEVERITY_STATUS_CAPTION[severity.status] : null;
  const emphasize = event.status === "confirmed" && (severity?.level === "high" || severity?.level === "critical");
  const cardClassName = emphasize ? "fire-event-card fire-event-card--emphasized" : "fire-event-card";
  const locationName = translateIfUntranslated(event.location_name);

  return (
    <article className={cardClassName} data-emphasized={emphasize}>
      <div className="fire-event-card__header">
        <div>
          <h3 className="fire-event-card__title">Event #{event.fire_event_id}</h3>
          <p
            className={
              locationName !== null
                ? "fire-event-card__location"
                : "fire-event-card__location fire-event-card__location--unavailable"
            }
          >
            {locationName !== null ? locationName : "Location unavailable"}
          </p>
        </div>
        <StatusBadge status={event.status} />
      </div>

      <dl className="fire-event-card__details">
        <div className="fire-event-card__row">
          <dt>Severity</dt>
          <dd>
            <SeverityBadge level={severity?.level ?? null} />
            {severityCaption ? <span className="fire-event-card__severity-caption">{severityCaption}</span> : null}
          </dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Detection</dt>
          <dd className="fire-event-card__scores" title={isAiMode ? AI_LIKELIHOOD_HELP_TEXT : undefined}>
            {isAiMode ? (
              <span>AI likelihood {aiLikelihood}</span>
            ) : (
              <>
                <span>Rule {ruleScore}</span>
                <span className="fire-event-card__scores-divider" aria-hidden="true">
                  |
                </span>
                <span>AI {aiScore}</span>
              </>
            )}
          </dd>
        </div>

        <div className="fire-event-card__row">
          <dt>Opened</dt>
          <dd>
            <TimestampDisplay value={event.created_at} />
          </dd>
        </div>
      </dl>

      {isMonitoringOnly ? <p className="fire-event-card__monitoring">{SUSPECTED_MONITORING_LABEL}</p> : null}

      <Link to={`/events/${event.fire_event_id}`} className="fire-event-card__link">
        View Event
      </Link>
    </article>
  );
}
