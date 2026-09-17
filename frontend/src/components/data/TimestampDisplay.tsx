import "./DataDisplay.css";

export interface TimestampDisplayProps {
  /** A backend ISO-8601 timestamp string, or null/undefined when unavailable. */
  value: string | null | undefined;
}

const NOT_AVAILABLE_LABEL = "Not available";

const FORMATTER = new Intl.DateTimeFormat("en-GB", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

/**
 * Frontend display formatting only - never mutates or reinterprets the
 * underlying timestamp. Renders a machine-readable `<time dateTime="...">`
 * carrying the original ISO value alongside the human-friendly text.
 */
export function TimestampDisplay({ value }: TimestampDisplayProps) {
  if (!value) {
    return <span className="timestamp-display timestamp-display--unavailable">{NOT_AVAILABLE_LABEL}</span>;
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return <span className="timestamp-display timestamp-display--unavailable">{NOT_AVAILABLE_LABEL}</span>;
  }

  return (
    <time className="timestamp-display" dateTime={value}>
      {FORMATTER.format(date)}
    </time>
  );
}
