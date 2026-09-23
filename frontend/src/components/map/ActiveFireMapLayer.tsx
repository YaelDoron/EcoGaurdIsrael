import { Marker, Popup } from "react-leaflet";
import { translateIfUntranslated } from "./stationTranslations";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import { SeverityBadge } from "../status/SeverityBadge";
import { StatusBadge } from "../status/StatusBadge";
import { STATUS_PRESENTATION } from "../status/presentation";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import { createActiveFireIcon } from "./activeFireIcon";
import { TONE_MAP_COLOR } from "./colors";
import "./activeFireIcon.css";

export interface ActiveFireMapLayerProps {
  activeFires: ActiveFireEvent[];
}

/**
 * Renders every A6 `active_fires` entry as a marker on the Operations map
 * (Task A8) - a dashboard-specific layer, independent of FireEventMarker
 * (Event Details' own single-event marker, left untouched to avoid any
 * conflict with that page). Marker color reuses the same
 * STATUS_PRESENTATION/TONE_MAP_COLOR lookup FireEventMarker already uses,
 * for visual consistency across the app.
 *
 * `emphasize` (a larger marker with a subtle halo) is pure presentation
 * over two already-persisted facts - FireEvent `status` and
 * `latest_severity.level` - never a new business score/rank, and never a
 * severity recalculation: a `null` severity is simply not emphasized, not
 * treated as "low".
 *
 * The popup shows the id, persisted status, persisted severity, persisted
 * coordinates, and (when present) `location_name` - optional,
 * read/presentation-only metadata the backend resolves from already-loaded
 * Fire Danger area geometries (see src/types/activeFireEvents.ts); never a
 * frontend-fabricated area/place name.
 *
 * Clarity pass: the status/severity badges get minimal text labels
 * ("Detection"/"Severity") so it is clear Suspected/Confirmed is the
 * detection status and High/Critical/etc. is the (distinct) Severity value -
 * never "Confidence: Suspected" (Suspected is not a numeric confidence).
 * Same compact popup, same badges, same colors - labels only.
 */
export function ActiveFireMapLayer({ activeFires }: ActiveFireMapLayerProps) {
  if (activeFires.length === 0) {
    return null;
  }

  return (
    <>
      {activeFires.map((fire) => {
        const presentation = STATUS_PRESENTATION[fire.status];
        const color = presentation ? TONE_MAP_COLOR[presentation.tone] : TONE_MAP_COLOR.neutral;
        const confirmed = fire.status === "confirmed";
        const emphasize = confirmed && (fire.severity?.level === "high" || fire.severity?.level === "critical");
        const icon = createActiveFireIcon(color, { emphasize, confirmed });

        return (
          <Marker key={fire.fire_event_id} position={[fire.latitude, fire.longitude]} icon={icon}>
            <Popup>
              <strong>Fire Event #{fire.fire_event_id}</strong>
              {fire.location_name !== null ? <p>{translateIfUntranslated(fire.location_name)}</p> : null}
              <p>
                Detection: <StatusBadge status={fire.status} />
              </p>
              <p>
                Severity: <SeverityBadge level={fire.severity?.level ?? null} />
              </p>
              <p>
                <CoordinateDisplay latitude={fire.latitude} longitude={fire.longitude} />
              </p>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
}
