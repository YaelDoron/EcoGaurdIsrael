import { useCallback, useRef, useState } from "react";
import { askChatbot } from "../api/chatbot";
import { ApiError } from "../api/errors";
import type { ChatbotAskRequest, ChatbotAskResponse } from "../types/chatbot";

const GENERIC_ERROR_MESSAGE = "Something went wrong while talking to the server.";

export interface UseAskChatbotResult {
  /** Sends the POST exactly once per call (a second call while one is in flight rejects immediately, never sending a duplicate request). */
  ask: (request: ChatbotAskRequest) => Promise<ChatbotAskResponse>;
  isSending: boolean;
  error: ApiError | null;
  reset: () => void;
}

/**
 * A reusable mutation for `POST /api/v1/chatbot/ask` (Task 7B), modeled
 * directly on `useStartSimulation`. Owns only request-in-flight/error
 * state - it never owns the visible conversation (see `ChatWindow`/
 * `AskAiChat`, which keep that separately) and never fabricates an answer
 * locally.
 */
export function useAskChatbot(): UseAskChatbotResult {
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const isSendingRef = useRef(false);

  const ask = useCallback((request: ChatbotAskRequest): Promise<ChatbotAskResponse> => {
    if (isSendingRef.current) {
      return Promise.reject(new ApiError("A chatbot request is already in flight.", 0));
    }
    isSendingRef.current = true;
    setIsSending(true);
    setError(null);

    return askChatbot(request)
      .then((response) => {
        setError(null);
        return response;
      })
      .catch((caught: unknown) => {
        const apiError = caught instanceof ApiError ? caught : new ApiError(GENERIC_ERROR_MESSAGE, 0);
        setError(apiError);
        throw apiError;
      })
      .finally(() => {
        isSendingRef.current = false;
        setIsSending(false);
      });
  }, []);

  const reset = useCallback(() => {
    setError(null);
  }, []);

  return { ask, isSending, error, reset };
}
