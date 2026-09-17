import { Marker, Popup } from "react-leaflet";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import { STATUS_PRESENTATION } from "../status/presentation";
import type { FireEventSummary } from "../../types/eventDetails";
import { TONE_MAP_COLOR } from "./colors";
import { createCircleIcon } from "./divIcon";

export interface FireEventMarkerProps {
  fireEvent: FireEventSummary;
}

const MARKER_SIZE = 22;

/**
 * The current FireEvent's own location, always present (fire_event is a
 * required field on EventDetailsResult) - this is the one marker the map
 * always shows, regardless of layer-visibility toggles for the optional
 * layers below it.
 */
export function FireEventMarker({ fireEvent }: FireEventMarkerProps) {
  const presentation = STATUS_PRESENTATION[fireEvent.status];
  const color = presentation ? TONE_MAP_COLOR[presentation.tone] : TONE_MAP_COLOR.neutral;
  const icon = createCircleIcon(color, { size: MARKER_SIZE });

  return (
    <Marker position={[fireEvent.latitude, fireEvent.longitude]} icon={icon}>
      <Popup>
        <strong>Event #{fireEvent.fire_event_id}</strong>
        <p>Status: {presentation?.label ?? fireEvent.status}</p>
        <p>
          <CoordinateDisplay latitude={fireEvent.latitude} longitude={fireEvent.longitude} />
        </p>
      </Popup>
    </Marker>
  );
}
