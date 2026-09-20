import { useMemo } from "react";
import { ActiveFireMapLayer } from "../map/ActiveFireMapLayer";
import { DisableScrollWheelZoom } from "../map/DisableScrollWheelZoom";
import { FireDangerLayer } from "../map/FireDangerLayer";
import { MapView } from "../map/MapView";
import type { LatLngPoint } from "../map/mapTypes";
import { OperationsMapLegend } from "./OperationsMapLegend";
import type { ActiveFireEvent } from "../../types/activeFireEvents";
import type { FireDangerArea } from "../../types/fireDanger";
import "./OperationsMap.css";

export interface OperationsMapProps {
  fireDangerAreas: FireDangerArea[];
  activeFires: ActiveFireEvent[];
}

/**
 * A small, neutral Israel-wide viewport default - NOT FireEvent/incident
 * data, just a fallback bounding box so the map shows something meaningful
 * (Task A8, Part 21) immediately after a demo reset, before any Fire
 * Danger area or active fire has been persisted yet.
 */
const ISRAEL_DEFAULT_BOUNDS_POINTS: LatLngPoint[] = [
  { lat: 33.33, lng: 34.9 },
  { lat: 29.49, lng: 35.92 },
];

function buildBoundsPoints(fireDangerAreas: FireDangerArea[], activeFires: ActiveFireEvent[]): LatLngPoint[] {
  const points: LatLngPoint[] = [
    ...fireDangerAreas.map((area) => ({ lat: area.center.latitude, lng: area.center.longitude })),
    ...activeFires.map((fire) => ({ lat: fire.latitude, lng: fire.longitude })),
  ];
  return points.length > 0 ? points : ISRAEL_DEFAULT_BOUNDS_POINTS;
}

/**
 * The Operations Overview's national map (Task A8): Fire Danger regions and
 * active fires rendered together, always simultaneously - never a toggle
 * between "Risk Map" and "Fire Map". Purely a composition of existing/new
 * layer components over data the page already has from useOperationsOverview();
 * this component fetches nothing itself.
 */
export function OperationsMap({ fireDangerAreas, activeFires }: OperationsMapProps) {
  const boundsPoints = useMemo(
    () => buildBoundsPoints(fireDangerAreas, activeFires),
    [fireDangerAreas, activeFires],
  );

  return (
    <div className="operations-map">
      <MapView boundsPoints={boundsPoints} ariaLabel="National operations map" scrollWheelZoom={false}>
        <DisableScrollWheelZoom />
        <FireDangerLayer areas={fireDangerAreas} />
        <ActiveFireMapLayer activeFires={activeFires} />
      </MapView>
      <OperationsMapLegend />
    </div>
  );
}
