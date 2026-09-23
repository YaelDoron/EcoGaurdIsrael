import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { askChatbot } from "./chatbot";
import { ApiError } from "./errors";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("chatbot API client", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("posts to the correct path with the exact question/history payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ answer: "Two active fires." }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await askChatbot({
      question: "What are the active fires?",
      history: [{ role: "user", content: "Hi" }],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/chatbot/ask",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          question: "What are the active fires?",
          history: [{ role: "user", content: "Hi" }],
        }),
      }),
    );
  });

  it("resolves with the typed answer", async () => {
    globalThis.fetch = vi
      .fn()
      .mockResolvedValue(jsonResponse({ answer: "The Galilee fire is confirmed." })) as unknown as typeof fetch;

    const result = await askChatbot({ question: "Status?", history: [] });

    expect(result).toEqual({ answer: "The Galilee fire is confirmed." });
  });

  it("preserves a 503 CHATBOT_UNAVAILABLE failure as a recognizable ApiError", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(
        { error: { code: "CHATBOT_UNAVAILABLE", message: "The chatbot is temporarily unavailable." } },
        { status: 503 },
      ),
    ) as unknown as typeof fetch;

    await expect(askChatbot({ question: "Status?", history: [] })).rejects.toMatchObject({
      status: 503,
      code: "CHATBOT_UNAVAILABLE",
    });
  });

  it("propagates an AbortSignal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ answer: "ok" }));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await askChatbot({ question: "Status?", history: [] }, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ signal: controller.signal }));
  });

  it("rejects with a generic ApiError on network failure, never a raw fetch error", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch")) as unknown as typeof fetch;

    await expect(askChatbot({ question: "Status?", history: [] })).rejects.toBeInstanceOf(ApiError);
  });
});
