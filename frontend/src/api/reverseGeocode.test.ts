import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { pickPlaceName, resetReverseGeocodeForTests, reverseGeocode } from "./reverseGeocode";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" }, ...init });
}

describe("pickPlaceName", () => {
  it("prefers the smallest settlement, then wider regions", () => {
    expect(pickPlaceName({ address: { city: "Haifa", state: "Haifa District" } })).toBe("Haifa");
    expect(pickPlaceName({ address: { village: "Isfiya", county: "Haifa Subdistrict" } })).toBe("Isfiya");
    expect(pickPlaceName({ address: { state_district: "North", state: "Israel North" } })).toBe("North");
  });

  it("returns null when the response has no usable address", () => {
    expect(pickPlaceName({})).toBeNull();
    expect(pickPlaceName({ address: {} })).toBeNull();
  });
});

describe("reverseGeocode", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    resetReverseGeocodeForTests();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("calls the Nominatim reverse endpoint with the coordinates and returns the place name", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ address: { city: "Haifa" } }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(reverseGeocode(32.7323, 35.0373)).resolves.toBe("Haifa");

    const url = String(fetchMock.mock.calls[0][0]);
    expect(url).toContain("https://nominatim.openstreetmap.org/reverse");
    expect(url).toContain("lat=32.7323");
    expect(url).toContain("lon=35.0373");
    expect(url).toContain("format=jsonv2");
  });

  it("caches per coordinate: repeated and in-flight lookups share one request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ address: { city: "Haifa" } }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const [a, b] = await Promise.all([reverseGeocode(32.7323, 35.0373), reverseGeocode(32.7324, 35.0374)]);
    const c = await reverseGeocode(32.7323, 35.0373);

    expect([a, b, c]).toEqual(["Haifa", "Haifa", "Haifa"]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("resolves null on an HTTP failure and does not cache it, so a later call retries", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({}, { status: 500 }))
      .mockResolvedValueOnce(jsonResponse({ address: { town: "Nesher" } }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await expect(reverseGeocode(32.77, 35.04)).resolves.toBeNull();
    await expect(reverseGeocode(32.77, 35.04)).resolves.toBe("Nesher");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("resolves null on a network error instead of throwing", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError("offline")) as unknown as typeof fetch;

    await expect(reverseGeocode(1, 1)).resolves.toBeNull();
  });
});
