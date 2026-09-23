import { useCallback, useEffect, useState } from "react";
import type { ChatbotHistoryMessage, ChatbotRole } from "../../types/chatbot";
import { AskAiButton, type AskAiButtonPosition } from "./AskAiButton";
import { ChatWindow } from "./ChatWindow";

const STORAGE_KEY = "ecoguard-ai-chat-history";

function isChatbotRole(value: unknown): value is ChatbotRole {
  return value === "user" || value === "assistant";
}

function isChatbotHistoryMessage(value: unknown): value is ChatbotHistoryMessage {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as { role?: unknown; content?: unknown };
  return isChatbotRole(candidate.role) && typeof candidate.content === "string";
}

/**
 * Loads the persisted conversation from `sessionStorage`. Never throws: a
 * missing key, malformed JSON, or an unexpected shape all fall back to an
 * empty conversation rather than crashing the app (Task 7B, Part 7).
 */
function loadStoredMessages(): ChatbotHistoryMessage[] {
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isChatbotHistoryMessage) : [];
  } catch {
    return [];
  }
}

/** Persists only `{role, content}` - never an ApiError, Gemini metadata, or an event id. */
function persistMessages(messages: ChatbotHistoryMessage[]): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch {
    // sessionStorage may be unavailable (private browsing, quota) - the
    // conversation still works for this page view, it just won't survive
    // a refresh in that case.
  }
}

/**
 * Owns the EcoGuard AI chat's global, cross-route state (Task 7B): whether
 * the window is open, and the in-memory conversation (backed by
 * `sessionStorage` so it survives a same-tab refresh - see Part 7/8 of the
 * task). Mounted once by `AppShell` so this state - and therefore the
 * conversation - naturally survives route navigation without any router-
 * specific code here.
 *
 * A dedicated wrapper (rather than `AppShell` owning this state itself, or
 * `AskAiButton` owning it) keeps `AppShell` a pure layout shell and keeps
 * `AskAiButton`/`ChatWindow` both simple, stateless-of-each-other
 * presentational components that only need `onClick`/`onClose` +
 * `messages`/`onAppendMessage` props.
 *
 * `buttonPosition` (Task 7C) also lives here, not inside `AskAiButton`
 * itself: `AskAiButton` unmounts whenever the chat is open (see the
 * conditional render below), so any position state kept inside it would be
 * lost on every open/close cycle. Kept here, it survives that - the button
 * reappears wherever the user last dragged it - and, like `messages`,
 * naturally survives route navigation since `AskAiChat` stays mounted.
 * Deliberately NOT persisted to `sessionStorage` (only the conversation
 * is) and NOT coupled to `ChatWindow`'s own, independent drag position.
 */
export function AskAiChat() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatbotHistoryMessage[]>(loadStoredMessages);
  const [buttonPosition, setButtonPosition] = useState<AskAiButtonPosition | null>(null);

  useEffect(() => {
    persistMessages(messages);
  }, [messages]);

  const appendMessage = useCallback((message: ChatbotHistoryMessage) => {
    setMessages((previous) => [...previous, message]);
  }, []);

  if (!isOpen) {
    return (
      <AskAiButton
        onClick={() => setIsOpen(true)}
        position={buttonPosition}
        onPositionChange={setButtonPosition}
      />
    );
  }

  return <ChatWindow messages={messages} onAppendMessage={appendMessage} onClose={() => setIsOpen(false)} />;
}
