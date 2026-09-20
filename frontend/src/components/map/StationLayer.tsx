import { Marker, Popup } from "react-leaflet";
import type { FireStation, StationSummary } from "../../types/eventDetails";
import { STATION_ALLOCATED_COLOR, STATION_MARKER_COLOR, TONE_MAP_COLOR } from "./colors";
import { createStationIcon } from "./divIcon";
import "./StationLayer.css";
import { translateStationAddress, translateStationName, translateStationType } from "./stationTranslations";

export interface StationLayerProps {
  stations: FireStation[];
  /**
   * Per-station resource counts and current-plan allocations
   * (`EventDetailsResult.station_summaries`). Optional so this layer stays
   * usable anywhere only `stations` is available; when a station has no
   * matching summary, the popup simply omits the resource section rather
   * than fabricating zero counts.
   */
  stationSummaries?: StationSummary[];
}

const STATION_ICON_SIZE = 26;

/**
 * Marker color for a station. A station that contributes trucks to the
 * current plan is the deep-blue action color (even when that leaves it with
 * no available trucks - it must not read as inactive). Every other station
 * keeps the default green, except one whose own counts show nothing
 * available: amber if some trucks are assigned, grey if none are usable.
 */
/**
 * Trucks still free under the current plan: the station's raw `available`
 * count minus the trucks the active global plan allocates from it, never
 * below zero. (The raw count is the DB state, which does not yet reflect this
 * plan's allocations.)
 */
export function dynamicAvailableCount(summary: StationSummary): number {
  return Math.max(0, summary.available - summary.current_global_plan_allocations.length);
}

function stationColor(summary: StationSummary | undefined): string {
  if (summary && summary.current_global_plan_allocations.length > 0) {
    return STATION_ALLOCATED_COLOR;
  }
  if (!summary || summary.total_resources === 0 || summary.available > 0) {
    return STATION_MARKER_COLOR;
  }
  if (summary.assigned_status > 0) {
    return TONE_MAP_COLOR.warning;
  }
  return TONE_MAP_COLOR.neutral;
}

function InventoryStats({ summary }: { summary: StationSummary }) {
  const dynamicAvailable = dynamicAvailableCount(summary);
  const allocated = summary.current_global_plan_allocations.length;
  return (
    <dl className="station-popup__stats">
      <div className="station-popup__stat">
        <dt>Total</dt>
        <dd>{summary.total_resources}</dd>
      </div>
      <div
        className={
          dynamicAvailable > 0
            ? "station-popup__stat station-popup__stat--available"
            : "station-popup__stat station-popup__stat--zero"
        }
      >
        <dt>Available</dt>
        <dd>{dynamicAvailable}</dd>
      </div>
      <div className="station-popup__stat station-popup__stat--allocated">
        <dt>Allocated</dt>
        <dd>{allocated}</dd>
      </div>
      <div
        className={
          summary.unavailable > 0
            ? "station-popup__stat station-popup__stat--unavailable"
            : "station-popup__stat station-popup__stat--zero"
        }
      >
        <dt>Unavailable</dt>
        <dd>{summary.unavailable}</dd>
      </div>
    </dl>
  );
}

function Allocations({ summary }: { summary: StationSummary }) {
  return (
    <section className="station-popup__allocations">
      <h4 className="station-popup__section-title">Allocated in current plan</h4>
      {summary.current_global_plan_allocations.length === 0 ? (
        <p className="station-popup__none">No allocated trucks</p>
      ) : (
        <ul className="station-popup__allocation-list">
          {summary.current_global_plan_allocations.map((allocation) => (
            <li key={allocation.resource_id} className="station-popup__allocation">
              {allocation.resource_id}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** Known fire stations, for spatial context on the map. */
export function StationLayer({ stations, stationSummaries = [] }: StationLayerProps) {
  if (stations.length === 0) {
    return null;
  }

  const summariesById = new Map(stationSummaries.map((summary) => [summary.station_id, summary]));

  return (
    <>
      {stations.map((station) => {
        const summary = summariesById.get(station.station_id);
        const icon = createStationIcon(stationColor(summary), { size: STATION_ICON_SIZE });
        return (
          <Marker key={station.station_id} position={[station.latitude, station.longitude]} icon={icon}>
            <Popup>
              <div className="station-popup" dir="ltr">
                <h3 className="station-popup__name">{translateStationName(station.name)}</h3>
                {station.address ? <p className="station-popup__address">{translateStationAddress(station.address)}</p> : null}
                {station.station_type ? (
                  <p className="station-popup__type">
                    <span className="station-popup__type-label">Type:</span>
                    <span className="station-popup__type-badge">{translateStationType(station.station_type)}</span>
                  </p>
                ) : null}
                {summary ? (
                  <>
                    <InventoryStats summary={summary} />
                    <hr className="station-popup__divider" />
                    <Allocations summary={summary} />
                  </>
                ) : null}
              </div>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
}
