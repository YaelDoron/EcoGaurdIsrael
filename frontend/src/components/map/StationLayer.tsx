import { Marker, Popup } from "react-leaflet";
import type { FireStation } from "../../types/eventDetails";
import { STATION_MARKER_COLOR } from "./colors";
import { createSquareIcon } from "./divIcon";

export interface StationLayerProps {
  stations: FireStation[];
}

const STATION_ICON_SIZE = 16;

/** Known fire stations, for spatial context on the map. */
export function StationLayer({ stations }: StationLayerProps) {
  if (stations.length === 0) {
    return null;
  }

  const icon = createSquareIcon(STATION_MARKER_COLOR, { size: STATION_ICON_SIZE });

  return (
    <>
      {stations.map((station) => (
        <Marker key={station.station_id} position={[station.latitude, station.longitude]} icon={icon}>
          <Popup>
            <strong>{station.name}</strong>
            {station.station_type ? <p>Type: {station.station_type}</p> : null}
            {station.address ? <p>{station.address}</p> : null}
          </Popup>
        </Marker>
      ))}
    </>
  );
}
