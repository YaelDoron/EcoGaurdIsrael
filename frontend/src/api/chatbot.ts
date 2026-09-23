import type { ChatbotAskRequest, ChatbotAskResponse } from "../types/chatbot";
import { apiPost } from "./client";

const CHATBOT_ASK_PATH = "/api/v1/chatbot/ask";

/**
 * Ask EcoGuard's AI decision-support chatbot a question. Thin wrapper over
 * the generic `apiPost` client - no fetch call, base-URL handling, retry,
 * or caching logic lives here (see client.ts for that).
 */
export function askChatbot(request: ChatbotAskRequest, signal?: AbortSignal): Promise<ChatbotAskResponse> {
  return apiPost<ChatbotAskResponse>(CHATBOT_ASK_PATH, request, { signal });
}
