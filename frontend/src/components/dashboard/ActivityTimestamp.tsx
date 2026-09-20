import "./ActivityTimestamp.css";

export interface ActivityTimestampProps {
  /** A backend ISO-8601 timestamp (the item's `available_at` - when it became available in EcoGuard, not its source `occurred_at`) - display formatting only, never mutated. */
  value: string;
}

const NOT_AVAILABLE_LABEL = "Not available";

const FORMATTER = new Intl.DateTimeFormat("en-GB", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

/**
 * Activity Feed-specific timestamp display: the shared `TimestampDisplay`
 * (day/month/year + hour:minute) made several items landing in the same
 * simulated minute look identical during a live demo run. This shows
 * HH:MM:SS instead so items arriving seconds apart stay visually distinct -
 * a presentation-only change, the underlying timestamp itself is never
 * touched, and the shared `TimestampDisplay` component (used across Event
 * Details/Response Plan/etc.) is left as-is.
 */
export function ActivityTimestamp({ value }: ActivityTimestampProps) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return <span className="activity-timestamp activity-timestamp--unavailable">{NOT_AVAILABLE_LABEL}</span>;
  }

  return (
    <time className="activity-timestamp" dateTime={value}>
      {FORMATTER.format(date)}
    </time>
  );
}
