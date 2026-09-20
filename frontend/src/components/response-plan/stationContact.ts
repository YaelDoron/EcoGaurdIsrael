import registry from "../../../../backend/src/database/data/firefighting_stations.json";

/**
 * Station contact numbers, read from the fire-station registry
 * (backend/src/database/data/firefighting_stations.json) - never invented.
 * Each plan station is looked up individually by the name it carries
 * (`resource.station_name`, seeded from the registry's `Station` field).
 *
 * Only a number that belongs to that station alone counts. The registry
 * currently gives every one of its stations the same `PhoneNumber`
 * (*4964, the national call-centre hotline), which is not a station's local
 * number, so a number shared by more than one station is discarded rather
 * than shown as if it were theirs. A station with no name, no registry entry,
 * an empty number, or only a shared number has no contact: callers render
 * nothing (no icon, no text) and no fallback is substituted.
 */
export interface StationContact {
  /** As stored in the registry, e.g. "*4964". */
  display: string;
  /** `tel:` URI (RFC 3966 allows star codes such as *4964). */
  href: string;
}

export interface RegistryRecord {
  Station: string | null;
  PhoneNumber?: string | number | null;
}

/** Station name -> that station's own phone number (numbers shared across stations are dropped). */
export function buildStationPhoneLookup(records: RegistryRecord[]): ReadonlyMap<string, string> {
  const byName = new Map<string, string>();
  for (const record of records) {
    const name = record.Station?.trim();
    const phone = record.PhoneNumber === null || record.PhoneNumber === undefined ? "" : String(record.PhoneNumber).trim();
    if (name && phone) {
      byName.set(name, phone);
    }
  }
  const stationsPerPhone = new Map<string, number>();
  for (const phone of byName.values()) {
    stationsPerPhone.set(phone, (stationsPerPhone.get(phone) ?? 0) + 1);
  }
  return new Map([...byName].filter(([, phone]) => stationsPerPhone.get(phone) === 1));
}

const PHONE_BY_STATION_NAME = buildStationPhoneLookup(registry.result.records as RegistryRecord[]);

export function getStationContact(stationName: string | null): StationContact | null {
  if (stationName === null) {
    return null;
  }
  const phone = PHONE_BY_STATION_NAME.get(stationName.trim());
  if (phone === undefined) {
    return null;
  }
  return { display: phone, href: `tel:${phone}` };
}
