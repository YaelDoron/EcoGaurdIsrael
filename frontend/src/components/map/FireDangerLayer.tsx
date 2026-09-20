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
 * Popup wording (clarity pass): explicitly labels the persisted danger
 * level as "Fire danger" (never "Severity" - that is a distinct FireEvent
 * concept) and the persisted score as "FFWI score" (the current
 * methodology is Fosberg Fire Weather Index) rather than a bare, ambiguous
 * "Score". Still no calculation of any kind - only presentation labels
 * around the exact same persisted `level`/`score`/`assessed_at`.
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
              <strong>{area.area_name}</strong>
              {assessment === null ? (
                <p>No assessment yet</p>
              ) : assessment.status !== "valid" || level === null ? (
                <p>Insufficient data for an assessment</p>
              ) : (
                <>
                  <p>Fire danger: {toTitleCase(FIRE_DANGER_PRESENTATION[level].label)}</p>
                  {assessment.score !== null ? (
                    <p title={FFWI_SCORE_HELP_TEXT}>FFWI score: {assessment.score.toFixed(1)}</p>
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
