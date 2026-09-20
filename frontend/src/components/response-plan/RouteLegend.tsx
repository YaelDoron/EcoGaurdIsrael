import "./RouteLegend.css";

/**
 * A small key for the Response Plan map, so the route styling explains
 * itself: which line is the selected route, which are the alternatives, and
 * what the fire and resource markers mean. Static presentation only - the
 * line samples mirror the styles in `ResponsePlanMapLayer`.
 */
export function RouteLegend() {
  return (
    <ul className="route-legend" aria-label="Map legend">
      <li className="route-legend__item">
        <svg width="34" height="10" aria-hidden="true">
          <line x1="2" y1="5" x2="32" y2="5" stroke="var(--color-focus-ring)" strokeWidth="6" strokeLinecap="round" />
        </svg>
        Selected route
      </li>
      <li className="route-legend__item">
        <svg width="34" height="10" aria-hidden="true">
          <line
            x1="2"
            y1="5"
            x2="32"
            y2="5"
            stroke="var(--color-text-muted)"
            strokeWidth="3"
            strokeDasharray="6 6"
          />
        </svg>
        Other routes
      </li>
      <li className="route-legend__item">
        <span className="route-legend__dot" aria-hidden="true" />
        Resource (origin)
      </li>
      <li className="route-legend__item">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="#ff4500" aria-hidden="true">
          <path d="M8 16c3.314 0 6-2 6-5.5 0-1.5-.5-4-2.5-6 .25 1.5-1.25 2-1.25 2C11 4 9 .5 6 0c.357 2 .5 4-2 6-1.25 1-2 2.729-2 4.5C2 14 4.686 16 8 16" />
        </svg>
        Target
      </li>
    </ul>
  );
}
