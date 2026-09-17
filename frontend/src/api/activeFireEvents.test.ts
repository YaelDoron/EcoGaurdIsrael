import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getActiveFireEvents } from "./activeFireEvents";
import { ApiError } from "./errors";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("getActiveFireEvents", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("requests the correct path", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ as_of: "2026-09-17T14:00:00Z", items: [] }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getActiveFireEvents();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/fire-events/active",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns the typed response on success", async () => {
    const body = {
      as_of: "2026-09-17T14:00:00Z",
      items: [
        {
          fire_event_id: 12,
          status: "confirmed",
          latitude: 32.731,
          longitude: 35.046,
          detection_confidence: 0.91,
          detected_at: "2026-09-17T13:20:00Z",
          updated_at: "2026-09-17T13:28:00Z",
          severity: null,
        },
      ],
    };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getActiveFireEvents();

    expect(result).toEqual(body);
  });

  it("propagates an ApiError on failure", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ error: { message: "server error" } }, { status: 500 })) as unknown as typeof fetch;

    await expect(getActiveFireEvents()).rejects.toBeInstanceOf(ApiError);
  });
});
