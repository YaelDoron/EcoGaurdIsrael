import { TimestampDisplay } from "../data/TimestampDisplay";
import { EmptyState } from "../feedback/EmptyState";
import type { DetectionEvidence, NewsEvidence, SatelliteEvidence } from "../../types/eventDetails";
import "./DetectionEvidencePanel.css";

export interface DetectionEvidencePanelProps {
  evidence: DetectionEvidence;
}

const DAY_NIGHT_LABEL: Record<string, string> = {
  D: "Day",
  N: "Night",
};

function dayNightLabel(value: string): string {
  return DAY_NIGHT_LABEL[value] ?? value;
}

const CONFIDENCE_LABEL: Record<string, string> = {
  n: "Nominal (n)",
  h: "High (h)",
  l: "Low (l)",
  nominal: "Nominal",
  high: "High",
  low: "Low",
};

/** Translates the satellite's raw single-letter confidence code into readable text; unknown values pass through unchanged. */
function confidenceLabel(value: string): string {
  return CONFIDENCE_LABEL[value.trim().toLowerCase()] ?? value;
}

const SATELLITE_SOURCE_LABEL = "Source: Meteorological Satellite";
const NEWS_SOURCE_LABEL = "Source: News Media";

function SatelliteEvidenceCard({ item }: { item: SatelliteEvidence }) {
  // Technical sensor details are secondary: small muted pills under the
  // primary (confidence) line, only for values the backend actually sent.
  const details: { label: string; value: string }[] = [];
  if (item.instrument !== null) details.push({ label: "Instrument", value: item.instrument });
  if (item.day_night !== null) details.push({ label: "Day / Night", value: dayNightLabel(item.day_night) });
  if (item.frp !== null) details.push({ label: "FRP", value: item.frp.toFixed(1) });
  if (item.brightness !== null) details.push({ label: "Brightness", value: item.brightness.toFixed(1) });

  return (
    <article className="detection-evidence-panel__card">
      <div className="detection-evidence-panel__card-header">
        <span className="detection-evidence-panel__card-title">{item.satellite ?? "Satellite hotspot"}</span>
        <TimestampDisplay value={item.detected_at} />
      </div>
      {item.confidence !== null ? (
        <dl className="detection-evidence-panel__facts">
          <div className="detection-evidence-panel__fact">
            <dt>Confidence</dt>
            <dd>{confidenceLabel(item.confidence)}</dd>
          </div>
        </dl>
      ) : null}
      {details.length > 0 ? (
        <ul className="detection-evidence-panel__pills" aria-label="Sensor details">
          {details.map((detail) => (
            <li key={detail.label} className="detection-evidence-panel__pill">
              <span className="detection-evidence-panel__pill-label">{detail.label}</span>
              {detail.value}
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  );
}

function NewsEvidenceCard({ item }: { item: NewsEvidence }) {
  return (
    <article className="detection-evidence-panel__card">
      <div className="detection-evidence-panel__card-header">
        <span className="detection-evidence-panel__card-title">{item.title}</span>
      </div>
      <p className="detection-evidence-panel__summary">{item.summary}</p>
      <dl className="detection-evidence-panel__facts">
        <div className="detection-evidence-panel__fact">
          <dt>Source</dt>
          <dd>{item.source}</dd>
        </div>
        <div className="detection-evidence-panel__fact">
          <dt>Reported</dt>
          <dd>
            <TimestampDisplay value={item.observed_at} />
          </dd>
        </div>
        {item.location_name !== null ? (
          <div className="detection-evidence-panel__fact">
            <dt>Location</dt>
            <dd>{item.location_name}</dd>
          </div>
        ) : null}
      </dl>
    </article>
  );
}

/**
 * The specific satellite hotspots and news reports that led to this
 * FireEvent's creation (`EventDetailsResult.detection_evidence`). Pure
 * display: every field shown comes straight from the API response, and a
 * satellite/news group is omitted entirely when the backend sent no items
 * for it - never a fabricated "no data" row standing in for real evidence.
 */
export function DetectionEvidencePanel({ evidence }: DetectionEvidencePanelProps) {
  const hasSatellite = evidence.satellite.length > 0;
  const hasNews = evidence.news.length > 0;

  if (!hasSatellite && !hasNews) {
    return (
      <EmptyState
        title="No detection evidence"
        message="No satellite hotspots or news reports are on record for this event."
      />
    );
  }

  return (
    <div className="detection-evidence-panel">
      {hasSatellite ? (
        <div className="detection-evidence-panel__group">
          <h3 className="detection-evidence-panel__group-title">Satellite Hotspots</h3>
          <p className="detection-evidence-panel__source">{SATELLITE_SOURCE_LABEL}</p>
          <div className="detection-evidence-panel__cards">
            {evidence.satellite.map((item) => (
              <SatelliteEvidenceCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}

      {hasNews ? (
        <div className="detection-evidence-panel__group">
          <h3 className="detection-evidence-panel__group-title">News Reports</h3>
          <p className="detection-evidence-panel__source">{NEWS_SOURCE_LABEL}</p>
          <div className="detection-evidence-panel__cards">
            {evidence.news.map((item) => (
              <NewsEvidenceCard key={item.id} item={item} />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
