/**
 * Frontend mirror of the chatbot API contract
 * (backend/src/api/schemas/chatbot.py). Exact property names and
 * nullability match that module - do not infer field names from task
 * descriptions, verify against the Pydantic source.
 *
 * `role` intentionally only accepts "user"/"assistant" - never Gemini's own
 * "model" turn role, which is an internal protocol detail the backend's
 * `ChatbotAgent` translates and never exposes at this boundary.
 */
export type ChatbotRole = "user" | "assistant";

export interface ChatbotHistoryMessage {
  role: ChatbotRole;
  content: string;
}

/** Request body for `POST /api/v1/chatbot/ask`. */
export interface ChatbotAskRequest {
  question: string;
  history: ChatbotHistoryMessage[];
}

/** Response body for `POST /api/v1/chatbot/ask` - the answer only (no `fire_event_id`, no metadata). */
export interface ChatbotAskResponse {
  answer: string;
}
