import "./DataDisplay.css";

export interface MetricCardProps {
  label: string;
  value: string | number;
  helperText?: string;
}

/**
 * Generic label/value metric tile. Presentation only - the page computes
 * `value` (e.g. an event count) and passes it in; this component performs
 * no calculation of its own.
 *
 * Uses a description list (<dt>/<dd>) rather than plain paragraphs so
 * assistive tech announces the label/value as one semantic group.
 */
export function MetricCard({ label, value, helperText }: MetricCardProps) {
  return (
    <dl className="metric-card">
      <dt className="metric-card__label">{label}</dt>
      <dd className="metric-card__value">{value}</dd>
      {helperText ? <dd className="metric-card__helper">{helperText}</dd> : null}
    </dl>
  );
}
