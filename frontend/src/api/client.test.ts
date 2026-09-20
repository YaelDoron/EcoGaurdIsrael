import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { apiGet, apiPost } from "./client";
import { ApiError } from "./errors";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("apiGet", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("prepends the configured base URL and issues a GET request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "ok" }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await apiGet("/health");

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/health",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("returns parsed JSON on success", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse({ status: "ok" })) as unknown as typeof fetch;

    const result = await apiGet<{ status: string }>("/health");

    expect(result).toEqual({ status: "ok" });
  });

  it("throws ApiError with the status code for a non-2xx response", async () => {
    globalThis.fetch = vi
      .fn()
      .mockImplementation(() => Promise.resolve(jsonResponse({ error: { message: "not found" } }, { status: 404 }))) as unknown as typeof fetch;

    await expect(apiGet("/missing")).rejects.toBeInstanceOf(ApiError);
    await expect(apiGet("/missing")).rejects.toMatchObject({ status: 404, message: "not found" });
  });

  it("preserves a backend-supplied error code when present", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse({ error: { code: "NOT_FOUND", message: "not found" } }, { status: 404 }),
    ) as unknown as typeof fetch;

    await expect(apiGet("/missing")).rejects.toMatchObject({ code: "NOT_FOUND" });
  });

  it("falls back to a safe generic message for a non-JSON error body", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response("<html><body>Internal Server Error</body></html>", {
        status: 500,
        headers: { "Content-Type": "text/html" },
      }),
    ) as unknown as typeof fetch;

    let caught: unknown;
    try {
      await apiGet("/broken");
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    const apiError = caught as ApiError;
    expect(apiError.status).toBe(500);
    expect(apiError.message).not.toContain("<html>");
    expect(apiError.message).not.toContain("Internal Server Error");
  });

  it("falls back to a safe generic message when an error body has no message field", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse({ detail: "unexpected shape" }, { status: 500 }),
    ) as unknown as typeof fetch;

    await expect(apiGet("/broken")).rejects.toMatchObject({
      status: 500,
      message: "Something went wrong while talking to the server.",
    });
  });

  it("wraps a network failure as an ApiError instead of throwing a raw exception", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    let caught: unknown;
    try {
      await apiGet("/health");
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect((caught as ApiError).status).toBe(0);
  });

  it("propagates an AbortSignal to fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ status: "ok" }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await apiGet("/health", { signal: controller.signal });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/health",
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("re-throws an AbortError unchanged instead of wrapping it as an ApiError", async () => {
    const abortError = new DOMException("The operation was aborted.", "AbortError");
    globalThis.fetch = vi.fn().mockRejectedValue(abortError) as unknown as typeof fetch;

    await expect(apiGet("/health")).rejects.toBe(abortError);
  });
});

describe("apiPost", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("prepends the configured base URL, issues a POST, and serializes the JSON body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ run_id: "run-1" }, { status: 202 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await apiPost("/api/v1/simulation/runs", { preset: "operations_demo", seed: 42, reset_demo_state: true });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/simulation/runs",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ preset: "operations_demo", seed: 42, reset_demo_state: true }),
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
      }),
    );
  });

  it("treats a 202 response as a success and returns the parsed body", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ run_id: "run-1", state: "preparing" }, { status: 202 })) as unknown as typeof fetch;

    const result = await apiPost<{ run_id: string; state: string }>("/api/v1/simulation/runs", {});

    expect(result).toEqual({ run_id: "run-1", state: "preparing" });
  });

  it("throws an ApiError with status 409 and the backend code for a conflict response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(
        { error: { code: "SIMULATION_ALREADY_RUNNING", message: "A simulation run is already running." } },
        { status: 409 },
      ),
    ) as unknown as typeof fetch;

    await expect(apiPost("/api/v1/simulation/runs", {})).rejects.toMatchObject({
      status: 409,
      code: "SIMULATION_ALREADY_RUNNING",
    });
  });

  it("propagates an AbortSignal to fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}, { status: 202 }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await apiPost("/api/v1/simulation/runs", {}, { signal: controller.signal });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/simulation/runs",
      expect.objectContaining({ signal: controller.signal }),
    );
  });
});
