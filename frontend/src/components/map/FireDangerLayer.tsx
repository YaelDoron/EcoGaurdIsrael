import { Circle, Popup } from "react-leaflet";
import { TimestampDisplay } from "../data/TimestampDisplay";
import { FIRE_DANGER_PRESENTATION, FIRE_DANGER_UNASSESSED_COLOR } from "../status/fireDangerPresentation";
import type { FireDangerArea } from "../../types/fireDanger";

export interface FireDangerLayerProps {
  areas: FireDangerArea[];
}

const FILL_OPACITY = 0.25;
const STROKE_WEIGHT = 2;
const METERS_PER_KM = 1000;

/**
 * Renders every A6 Fire Danger area as a true geographic circle (Task A8) -
 * `center`/`radius_km` come straight from the backend, converted to meters
 * for Leaflet's `Circle`; never a fabricated polygon. Color is looked up
 * from the persisted `level` via FIRE_DANGER_PRESENTATION - this component
 * performs no FFWI calculation and no score->level derivation. An area with
 * no persisted assessment yet (`assessment === null`) is still drawn (so an
 * operator can see it is a known/monitored area) but in a neutral color,
 * never implying "low risk".
 *
 * Data-only: this component does not fetch anything itself - the page
 * supplies `areas` from useOperationsOverview()'s own response.
 *
 * Popup wording (clarity pass, extended - dashboard Part 3): an explicit
 * "Fire Danger Assessment" heading makes it unmistakable that this is NOT
 * a FireEvent popup (see ActiveFireMapLayer's "Fire Event #N" popup, which
 * shows Detection/Severity and never FFWI) before the area name/level/FFWI
 * - color alone was not a sufficient signal. "Level" (never "Severity" -
 * that is a distinct FireEvent concept) and "FFWI" (the current methodology
 * is Fosberg Fire Weather Index, never a bare ambiguous "Score") are shown
 * for EVERY level (Low/Moderate/High/Very High/Extreme) whenever the
 * persisted assessment has a score - rendering is conditioned only on
 * `assessment.score !== null`, never on `level`, so a High/Extreme reading
 * is exactly as likely to show FFWI as a Low one. Still no calculation of
 * any kind - only presentation labels around the exact same persisted
 * `level`/`score`/`assessed_at`.
 */
function toTitleCase(label: string): string {
  return label.replace(/\b\w/g, (character) => character.toUpperCase());
}

const FFWI_SCORE_HELP_TEXT = "Fire-weather danger score used to determine the Fire Danger level.";

export function FireDangerLayer({ areas }: FireDangerLayerProps) {
  if (areas.length === 0) {
    return null;
  }

  return (
    <>
      {areas.map((area) => {
        const assessment = area.assessment;
        const level = assessment?.level ?? null;
        const color = level !== null ? FIRE_DANGER_PRESENTATION[level].color : FIRE_DANGER_UNASSESSED_COLOR;

        return (
          <Circle
            key={area.area_id}
            center={[area.center.latitude, area.center.longitude]}
            radius={area.radius_km * METERS_PER_KM}
            pathOptions={{ color, fillColor: color, fillOpacity: FILL_OPACITY, weight: STROKE_WEIGHT }}
          >
            <Popup>
              <p>
                <small>Fire Danger Assessment</small>
              </p>
              <strong>{area.area_name}</strong>
              {assessment === null ? (
                <p>No assessment yet</p>
              ) : assessment.status !== "valid" || level === null ? (
                <p>Insufficient data for an assessment</p>
              ) : (
                <>
                  <p>Level: {toTitleCase(FIRE_DANGER_PRESENTATION[level].label)}</p>
                  {assessment.score !== null ? (
                    <p title={FFWI_SCORE_HELP_TEXT}>FFWI: {assessment.score.toFixed(1)}</p>
                  ) : null}
                  <p>
                    Assessed: <TimestampDisplay value={assessment.assessed_at} />
                  </p>
                </>
              )}
            </Popup>
          </Circle>
        );
      })}
    </>
  );
}
