/**
 * Client-side reverse geocoding via OpenStreetMap Nominatim: turns a
 * coordinate into a human-readable place name (city/town/village/region).
 *
 * Behaviour:
 * - Results are cached per coordinate (rounded to ~100 m), and an in-flight
 *   request is shared, so the same coordinate is never requested twice.
 * - Requests are serialized and spaced at least `MIN_INTERVAL_MS` apart, per
 *   Nominatim's usage policy (max 1 request/second).
 * - A failed request resolves to `null` and is NOT cached, so a later render
 *   can retry; callers fall back to another name rather than showing an error.
 */
const ENDPOINT = "https://nominatim.openstreetmap.org/reverse";
const MIN_INTERVAL_MS = 1100;

interface NominatimAddress {
  city?: string;
  town?: string;
  village?: string;
  hamlet?: string;
  municipality?: string;
  county?: string;
  state_district?: string;
  state?: string;
}

interface NominatimResponse {
  address?: NominatimAddress;
  name?: string;
}

/** Smallest meaningful settlement first, then wider regions. */
export function pickPlaceName(body: NominatimResponse): string | null {
  const a = body.address;
  if (!a) {
    return null;
  }
  return (
    a.city ?? a.town ?? a.village ?? a.hamlet ?? a.municipality ?? a.county ?? a.state_district ?? a.state ?? null
  );
}

const cache = new Map<string, Promise<string | null>>();
let lastStartedAt = 0;
let queue: Promise<unknown> = Promise.resolve();

function keyOf(latitude: number, longitude: number): string {
  return `${latitude.toFixed(3)},${longitude.toFixed(3)}`;
}

function schedule<T>(task: () => Promise<T>): Promise<T> {
  const run = queue.then(async () => {
    const wait = lastStartedAt + MIN_INTERVAL_MS - Date.now();
    if (wait > 0) {
      await new Promise((resolve) => setTimeout(resolve, wait));
    }
    lastStartedAt = Date.now();
    return task();
  });
  queue = run.catch(() => undefined);
  return run;
}

async function fetchPlaceName(latitude: number, longitude: number): Promise<string | null> {
  const url =
    `${ENDPOINT}?format=jsonv2&zoom=10&addressdetails=1&accept-language=en` +
    `&lat=${encodeURIComponent(latitude)}&lon=${encodeURIComponent(longitude)}`;
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(`Reverse geocoding failed with status ${response.status}`);
  }
  return pickPlaceName((await response.json()) as NominatimResponse);
}

export function reverseGeocode(latitude: number, longitude: number): Promise<string | null> {
  const key = keyOf(latitude, longitude);
  const existing = cache.get(key);
  if (existing) {
    return existing;
  }
  const request = schedule(() => fetchPlaceName(latitude, longitude)).catch(() => {
    cache.delete(key);
    return null;
  });
  cache.set(key, request);
  return request;
}

/** Test-only: clears the cache and rate-limiter state. */
export function resetReverseGeocodeForTests(): void {
  cache.clear();
  lastStartedAt = 0;
  queue = Promise.resolve();
}
