import { useEffect, useState } from "react";
import { reverseGeocode } from "../api/reverseGeocode";

export interface GeocodeTarget {
  id: number;
  latitude: number;
  longitude: number;
}

/**
 * Resolves each target's coordinates to a place name via reverse geocoding
 * (see `api/reverseGeocode` for caching and rate limiting). Returns a map of
 * target id -> name; an id is absent while its lookup is in flight and `null`
 * when the lookup failed or found no name - callers fall back to the parent
 * event's location rather than showing the raw target id.
 */
export function useTargetLocationNames(targets: GeocodeTarget[]): Record<number, string | null> {
  const [names, setNames] = useState<Record<number, string | null>>({});
  const signature = targets.map((t) => `${t.id}:${t.latitude},${t.longitude}`).join("|");

  useEffect(() => {
    let cancelled = false;
    for (const target of targets) {
      reverseGeocode(target.latitude, target.longitude).then((name) => {
        if (!cancelled) {
          setNames((previous) => (previous[target.id] === name ? previous : { ...previous, [target.id]: name }));
        }
      });
    }
    return () => {
      cancelled = true;
    };
    // `signature` captures the identity/coordinates of `targets`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature]);

  return names;
}
