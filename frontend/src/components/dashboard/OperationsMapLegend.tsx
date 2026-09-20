import { FIRE_DANGER_PRESENTATION } from "../status/fireDangerPresentation";
import "./OperationsMapLegend.css";

const FIRE_DANGER_LEVELS = ["low", "moderate", "high", "very_high", "extreme"] as const;

/**
 * A compact legend for the Operations map (Task A8, Part 22; visual polish
 * pass moved it to a small overlay INSIDE the map rather than a full-width
 * strip below it, so it no longer creates its own horizontal row). Explains
 * the Fire Danger color scale and the active-fire marker vocabulary -
 * meaning is conveyed by the adjacent text label, not by color alone.
 * Colors/business meanings themselves are unchanged - only the layout.
 */
export function OperationsMapLegend() {
  return (
    <div className="operations-map-legend">
      <h3 className="operations-map-legend__heading">Legend</h3>

      <div className="operations-map-legend__group">
        <h4 className="operations-map-legend__title">Fire danger</h4>
        <ul className="operations-map-legend__list">
          {FIRE_DANGER_LEVELS.map((level) => (
            <li key={level} className="operations-map-legend__item">
              <span
                className="operations-map-legend__swatch"
                style={{ background: FIRE_DANGER_PRESENTATION[level].color }}
                aria-hidden="true"
              />
              <span>{FIRE_DANGER_PRESENTATION[level].label}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="operations-map-legend__group">
        <h4 className="operations-map-legend__title">Active fires</h4>
        <ul className="operations-map-legend__list">
          <li className="operations-map-legend__item">
            <span
              className="operations-map-legend__marker operations-map-legend__marker--suspected"
              aria-hidden="true"
            />
            <span>Suspected</span>
          </li>
          <li className="operations-map-legend__item">
            <span
              className="operations-map-legend__marker operations-map-legend__marker--confirmed"
              aria-hidden="true"
            />
            <span>Confirmed</span>
          </li>
          <li className="operations-map-legend__item">
            <span
              className="operations-map-legend__marker operations-map-legend__marker--emphasized"
              aria-hidden="true"
            />
            <span>High/Critical</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
