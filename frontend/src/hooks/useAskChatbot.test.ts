import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { ChatbotAskResponse } from "../types/chatbot";
import { useAskChatbot } from "./useAskChatbot";

const { askChatbotMock } = vi.hoisted(() => ({
  askChatbotMock: vi.fn(),
}));

vi.mock("../api/chatbot", () => ({
  askChatbot: askChatbotMock,
}));

describe("useAskChatbot", () => {
  beforeEach(() => {
    askChatbotMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("sends the POST exactly once per call", async () => {
    askChatbotMock.mockResolvedValue({ answer: "ok" } satisfies ChatbotAskResponse);
    const { result } = renderHook(() => useAskChatbot());

    await act(async () => {
      await result.current.ask({ question: "Status?", history: [] });
    });

    expect(askChatbotMock).toHaveBeenCalledTimes(1);
  });

  it("sends the exact question/history payload", async () => {
    askChatbotMock.mockResolvedValue({ answer: "ok" } satisfies ChatbotAskResponse);
    const { result } = renderHook(() => useAskChatbot());

    await act(async () => {
      await result.current.ask({ question: "Status?", history: [{ role: "user", content: "Hi" }] });
    });

    expect(askChatbotMock).toHaveBeenCalledWith({ question: "Status?", history: [{ role: "user", content: "Hi" }] });
  });

  it("resolves with the typed answer", async () => {
    askChatbotMock.mockResolvedValue({ answer: "The Galilee fire is confirmed." } satisfies ChatbotAskResponse);
    const { result } = renderHook(() => useAskChatbot());

    let resolved: ChatbotAskResponse | undefined;
    await act(async () => {
      resolved = await result.current.ask({ question: "Status?", history: [] });
    });

    expect(resolved?.answer).toBe("The Galilee fire is confirmed.");
  });

  it("isSending reflects the in-flight request", async () => {
    let resolveAsk!: (value: ChatbotAskResponse) => void;
    askChatbotMock.mockReturnValue(
      new Promise<ChatbotAskResponse>((resolve) => {
        resolveAsk = resolve;
      }),
    );
    const { result } = renderHook(() => useAskChatbot());

    let askPromise!: Promise<ChatbotAskResponse>;
    act(() => {
      askPromise = result.current.ask({ question: "Status?", history: [] });
    });

    await waitFor(() => expect(result.current.isSending).toBe(true));

    await act(async () => {
      resolveAsk({ answer: "ok" });
      await askPromise;
    });

    expect(result.current.isSending).toBe(false);
  });

  it("rejects a second call while one is already in flight, never sending a duplicate request", async () => {
    askChatbotMock.mockReturnValue(new Promise<ChatbotAskResponse>(() => {}));
    const { result } = renderHook(() => useAskChatbot());

    act(() => {
      void result.current.ask({ question: "First?", history: [] });
    });
    await waitFor(() => expect(result.current.isSending).toBe(true));

    await act(async () => {
      await expect(result.current.ask({ question: "Second?", history: [] })).rejects.toBeInstanceOf(ApiError);
    });

    expect(askChatbotMock).toHaveBeenCalledTimes(1);
  });

  it("preserves a 503 CHATBOT_UNAVAILABLE failure as a typed, recognizable error", async () => {
    askChatbotMock.mockRejectedValue(new ApiError("The chatbot is temporarily unavailable.", 503, "CHATBOT_UNAVAILABLE"));
    const { result } = renderHook(() => useAskChatbot());

    await act(async () => {
      await expect(result.current.ask({ question: "Status?", history: [] })).rejects.toBeInstanceOf(ApiError);
    });

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(result.current.error?.status).toBe(503);
    expect(result.current.error?.code).toBe("CHATBOT_UNAVAILABLE");
    expect(result.current.isSending).toBe(false);
  });

  it("reset clears the error", async () => {
    askChatbotMock.mockRejectedValue(new ApiError("Unavailable.", 503, "CHATBOT_UNAVAILABLE"));
    const { result } = renderHook(() => useAskChatbot());

    await act(async () => {
      await expect(result.current.ask({ question: "Status?", history: [] })).rejects.toBeInstanceOf(ApiError);
    });
    await waitFor(() => expect(result.current.error).not.toBeNull());

    act(() => {
      result.current.reset();
    });

    expect(result.current.error).toBeNull();
  });
});
