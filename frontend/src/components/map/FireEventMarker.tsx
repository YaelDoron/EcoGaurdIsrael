import { Marker, Popup } from "react-leaflet";
import { CoordinateDisplay } from "../data/CoordinateDisplay";
import type { FireEventSummary } from "../../types/eventDetails";
import { FIRE_ICON_COLOR_ACTIVE, FIRE_ICON_COLOR_CONFIRMED, FIRE_ICON_COLOR_INACTIVE } from "./colors";
import { createFireIcon } from "./divIcon";

export interface FireEventMarkerProps {
  fireEvent: FireEventSummary;
}

const MARKER_SIZE = 32;

const STATUS_LABEL: Record<FireEventSummary["status"], string> = {
  suspected: "Suspected",
  confirmed: "Confirmed",
  resolved: "Resolved",
  dismissed: "Dismissed",
};

// An active wildfire (suspected/confirmed) is drawn in fiery orange/red; a
// resolved/dismissed one is muted so it never reads as still burning.
function fireColor(status: FireEventSummary["status"]): string {
  if (status === "confirmed") return FIRE_ICON_COLOR_CONFIRMED;
  if (status === "suspected") return FIRE_ICON_COLOR_ACTIVE;
  return FIRE_ICON_COLOR_INACTIVE;
}

/**
 * The current FireEvent's own location, always present (fire_event is a
 * required field on EventDetailsResult) - this is the one marker the map
 * always shows, regardless of layer-visibility toggles for the optional
 * layers below it.
 */
export function FireEventMarker({ fireEvent }: FireEventMarkerProps) {
  const icon = createFireIcon(fireColor(fireEvent.status), { size: MARKER_SIZE });

  return (
    <Marker position={[fireEvent.latitude, fireEvent.longitude]} icon={icon}>
      <Popup>
        <strong>Event #{fireEvent.fire_event_id}</strong>
        <p>Status: {STATUS_LABEL[fireEvent.status] ?? fireEvent.status}</p>
        <p>
          <CoordinateDisplay latitude={fireEvent.latitude} longitude={fireEvent.longitude} />
        </p>
      </Popup>
    </Marker>
  );
}
