import { useEffect, useRef, useState, type AnchorHTMLAttributes, type KeyboardEvent, type PointerEvent } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { useAskChatbot } from "../../hooks/useAskChatbot";
import type { ApiError } from "../../api/errors";
import type { ChatbotHistoryMessage } from "../../types/chatbot";
import "./ChatWindow.css";

// Mirrors the backend's own authoritative cap
// (MAX_CHAT_HISTORY_MESSAGES in backend/src/agents/response/chatbot_agent.py)
// - ChatbotAgent trims independently regardless, but the frontend avoids
// sending an unbounded conversation on every request.
const MAX_HISTORY_MESSAGES = 8;

export interface ChatWindowProps {
  messages: ChatbotHistoryMessage[];
  onAppendMessage: (message: ChatbotHistoryMessage) => void;
  onClose: () => void;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * True when `target` is (or is inside) a button/link/input/textarea/select.
 * The header's drag handler must never start a drag - and must never call
 * `setPointerCapture` - for a press that landed on one of these (e.g. the
 * close button): once an ancestor captures a pointer, browsers redirect
 * that pointer's subsequent events (including the click a plain tap on the
 * close button would otherwise produce) to the capturing element, which is
 * exactly what made the close button unresponsive in the real browser.
 */
function isInteractiveElement(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest("button, a, input, textarea, select") !== null;
}

function friendlyErrorMessage(error: ApiError): string {
  if (error.code === "CHATBOT_UNAVAILABLE") {
    return "EcoGuard AI is temporarily unavailable. Please try again.";
  }
  return "EcoGuard AI could not complete the request. Please try again.";
}

/**
 * Any Markdown link opens in a new tab with `rel="noopener noreferrer"` -
 * `react-markdown` already strips dangerous URL schemes (e.g. `javascript:`)
 * from `href` by default, this only adds safe target/rel behavior on top.
 */
const MARKDOWN_COMPONENTS: Components = {
  a: ({ children, ...props }: AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

/**
 * One message bubble. Assistant answers are Gemini-generated Markdown
 * (headings/bold/lists - see ChatbotAgent's system instruction) and are
 * rendered as real elements via `react-markdown` - never raw HTML
 * (`rehypeRaw` is deliberately not used, and `dangerouslySetInnerHTML` is
 * never touched), so untrusted response text can only ever become safe,
 * known Markdown-derived elements. User messages are always shown as
 * plain text exactly as typed, never Markdown-interpreted.
 */
function MessageBubble({ message }: { message: ChatbotHistoryMessage }) {
  if (message.role === "assistant") {
    return (
      <div className="chat-bubble chat-bubble--assistant">
        <div className="chat-bubble__markdown" dir="auto">
          <ReactMarkdown components={MARKDOWN_COMPONENTS}>{message.content}</ReactMarkdown>
        </div>
      </div>
    );
  }
  return (
    <div className="chat-bubble chat-bubble--user">
      <p dir="auto">{message.content}</p>
    </div>
  );
}

/**
 * The floating EcoGuard AI chat window (Task 7B). Draggable on desktop from
 * its header only (native Pointer Events - no drag dependency); becomes a
 * fixed, non-draggable bottom sheet below 960px, matching
 * `OperationsActivityDrawer`'s own existing responsive pattern.
 *
 * Owns only transient/interaction state (drag position, draft input text,
 * in-flight send/error via `useAskChatbot`) - the persisted conversation
 * itself is owned by the parent (`AskAiChat`) so it survives this
 * component unmounting when the window is closed.
 */
export function ChatWindow({ messages, onAppendMessage, onClose }: ChatWindowProps) {
  const { ask, isSending, error } = useAskChatbot();
  const [draftText, setDraftText] = useState("");
  const [position, setPosition] = useState<{ top: number; left: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const windowRef = useRef<HTMLDivElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dragOffsetRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    const node = messagesRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages, isSending]);

  function handleHeaderPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (isInteractiveElement(event.target)) {
      // A press on the close button (or any other interactive descendant)
      // must never start a drag or capture the pointer - see
      // isInteractiveElement's docstring.
      return;
    }
    const rect = windowRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    dragOffsetRef.current = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    setIsDragging(true);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture isn't supported everywhere (e.g. some test
      // environments) - dragging still works via move/up below either way.
    }
  }

  function handleHeaderPointerMove(event: PointerEvent<HTMLDivElement>) {
    const offset = dragOffsetRef.current;
    if (!offset) {
      return;
    }
    const rect = windowRef.current?.getBoundingClientRect();
    const width = rect?.width ?? 0;
    const height = rect?.height ?? 0;
    const maxLeft = Math.max(0, window.innerWidth - width);
    const maxTop = Math.max(0, window.innerHeight - height);
    setPosition({
      left: clamp(event.clientX - offset.x, 0, maxLeft),
      top: clamp(event.clientY - offset.y, 0, maxTop),
    });
  }

  function endDrag(event: PointerEvent<HTMLDivElement>) {
    dragOffsetRef.current = null;
    setIsDragging(false);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // See handleHeaderPointerDown.
    }
  }

  function handleSend() {
    const trimmed = draftText.trim();
    if (trimmed === "" || isSending) {
      return;
    }

    // Built from the conversation BEFORE the new question is appended, so
    // the current question can never end up duplicated inside `history`.
    const history = messages.slice(-MAX_HISTORY_MESSAGES);
    setDraftText("");
    onAppendMessage({ role: "user", content: trimmed });

    ask({ question: trimmed, history })
      .then((response) => {
        onAppendMessage({ role: "assistant", content: response.answer });
      })
      .catch(() => {
        // `error` (from useAskChatbot) already reflects this - the user's
        // message above stays visible so they can retry/rephrase.
      });
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  const canSend = draftText.trim() !== "" && !isSending;

  return (
    <div
      ref={windowRef}
      className={`chat-window${isDragging ? " chat-window--dragging" : ""}`}
      role="dialog"
      aria-labelledby="chat-window-title"
      style={position ? { top: position.top, left: position.left } : undefined}
    >
      <div
        className="chat-window__header"
        onPointerDown={handleHeaderPointerDown}
        onPointerMove={handleHeaderPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        <div className="chat-window__title-group">
          <svg className="chat-window__icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 2 L14.5 9.5 L22 12 L14.5 14.5 L12 22 L9.5 14.5 L2 12 L9.5 9.5 Z" fill="currentColor" />
          </svg>
          <div>
            <h2 id="chat-window-title" className="chat-window__title">
              EcoGuard AI
            </h2>
            <p className="chat-window__subtitle">AI decision support based on current EcoGuard data</p>
          </div>
        </div>
        <button
          type="button"
          className="chat-window__close"
          onClick={onClose}
          onPointerDown={(event) => event.stopPropagation()}
          aria-label="Close EcoGuard AI chat"
        >
          &times;
        </button>
      </div>

      <div className="chat-window__messages" ref={messagesRef}>
        {messages.length === 0 ? (
          <p className="chat-window__empty">
            Ask EcoGuard AI about active fire events, severity, spread predictions, or response plans.
          </p>
        ) : (
          messages.map((message, index) => <MessageBubble key={index} message={message} />)
        )}
        {isSending ? (
          <p className="chat-window__status" role="status" aria-live="polite">
            Thinking&hellip;
          </p>
        ) : null}
      </div>

      {error ? (
        <p className="chat-window__error" role="alert">
          {friendlyErrorMessage(error)}
        </p>
      ) : null}

      <div className="chat-window__input-row">
        <textarea
          ref={textareaRef}
          className="chat-window__input"
          aria-label="Your question"
          value={draftText}
          onChange={(event) => setDraftText(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question in Hebrew or English…"
          rows={2}
        />
        <button type="button" className="chat-window__send" onClick={handleSend} disabled={!canSend}>
          Send
        </button>
      </div>
    </div>
  );
}
