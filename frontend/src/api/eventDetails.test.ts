import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getEventDetails } from "./eventDetails";
import { ApiError } from "./errors";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("getEventDetails", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("requests the correct path for the given fire_event_id", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        as_of: "2026-09-17T14:00:00Z",
        fire_event: {
          fire_event_id: 12,
          status: "confirmed",
          latitude: 32.731,
          longitude: 35.046,
          detection_confidence: 0.91,
          detected_at: "2026-09-17T13:20:00Z",
          updated_at: "2026-09-17T13:28:00Z",
          methodology: "detector",
          methodology_version: "1.0",
        },
        severity: null,
        danger: null,
        spread_predictions: [],
        targets: [],
        stations: [],
        resources: [],
        current_response_plan: null,
      }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getEventDetails(12);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/fire-events/12/details",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns the typed response on success", async () => {
    const body = {
      as_of: "2026-09-17T14:00:00Z",
      fire_event: {
        fire_event_id: 12,
        status: "confirmed",
        latitude: 32.731,
        longitude: 35.046,
        detection_confidence: 0.91,
        detected_at: "2026-09-17T13:20:00Z",
        updated_at: "2026-09-17T13:28:00Z",
        methodology: "detector",
        methodology_version: "1.0",
      },
      severity: null,
      danger: null,
      spread_predictions: [],
      targets: [],
      stations: [],
      resources: [],
      current_response_plan: null,
    };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getEventDetails(12);

    expect(result).toEqual(body);
  });

  it("propagates an ApiError with status 404 when the event does not exist", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ error: { message: "FireEvent not found." } }, { status: 404 })) as unknown as typeof fetch;

    await expect(getEventDetails(999)).rejects.toMatchObject({ status: 404 });
    await expect(getEventDetails(999)).rejects.toBeInstanceOf(ApiError);
  });

  it("propagates an ApiError on a server failure", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ error: { message: "server error" } }, { status: 500 })) as unknown as typeof fetch;

    await expect(getEventDetails(12)).rejects.toBeInstanceOf(ApiError);
  });
});
