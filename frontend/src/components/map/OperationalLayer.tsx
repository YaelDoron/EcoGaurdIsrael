import { Marker, Popup } from "react-leaflet";
import type { FireStation, FirefightingResource, ResourceStatus } from "../../types/eventDetails";
import { TONE_MAP_COLOR } from "./colors";
import { createCircleIcon } from "./divIcon";

export interface OperationalLayerProps {
  resources: FirefightingResource[];
  stations: FireStation[];
}

const MARKER_SIZE = 16;

export const RESOURCE_STATUS_LABEL: Record<ResourceStatus, string> = {
  available: "Available",
  assigned: "Assigned",
  unavailable: "Unavailable",
};

// Dominant status per station, in priority order (best-case status wins).
const STATUS_PRIORITY: ResourceStatus[] = ["available", "assigned", "unavailable"];
const STATUS_TONE: Record<ResourceStatus, keyof typeof TONE_MAP_COLOR> = {
  available: "success",
  assigned: "warning",
  unavailable: "neutral",
};

interface StationResourceGroup {
  station: FireStation;
  resources: FirefightingResource[];
}

function groupResourcesByStation(
  resources: FirefightingResource[],
  stations: FireStation[],
): StationResourceGroup[] {
  const stationsById = new Map(stations.map((station) => [station.station_id, station]));
  const groups = new Map<string, StationResourceGroup>();

  for (const resource of resources) {
    const station = stationsById.get(resource.station_id);
    if (!station) {
      // A resource whose station isn't in the fetched station list can't be
      // placed on the map - skipped rather than plotted at a guessed location.
      continue;
    }
    const existing = groups.get(resource.station_id);
    if (existing) {
      existing.resources.push(resource);
    } else {
      groups.set(resource.station_id, { station, resources: [resource] });
    }
  }

  return Array.from(groups.values());
}

function dominantStatus(resources: FirefightingResource[]): ResourceStatus {
  for (const status of STATUS_PRIORITY) {
    if (resources.some((resource) => resource.status === status)) {
      return status;
    }
  }
  return "unavailable";
}

/**
 * Firefighting resources, grouped and positioned at their station's known
 * coordinates (resources carry no coordinates of their own - see
 * FirefightingResourceResponse on the backend). Grouping/positioning is a
 * presentation join only, never an inference of a resource's real location.
 */
export function OperationalLayer({ resources, stations }: OperationalLayerProps) {
  const groups = groupResourcesByStation(resources, stations);

  if (groups.length === 0) {
    return null;
  }

  return (
    <>
      {groups.map(({ station, resources: stationResources }) => {
        const status = dominantStatus(stationResources);
        const icon = createCircleIcon(TONE_MAP_COLOR[STATUS_TONE[status]], { size: MARKER_SIZE });
        return (
          <Marker key={station.station_id} position={[station.latitude, station.longitude]} icon={icon}>
            <Popup>
              <strong>{station.name}</strong>
              <ul>
                {stationResources.map((resource) => (
                  <li key={resource.resource_id}>
                    {resource.resource_id}: {RESOURCE_STATUS_LABEL[resource.status]}
                  </li>
                ))}
              </ul>
            </Popup>
          </Marker>
        );
      })}
    </>
  );
}
