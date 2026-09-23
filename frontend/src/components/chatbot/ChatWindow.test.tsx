import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/errors";
import type { ChatbotHistoryMessage } from "../../types/chatbot";
import { ChatWindow } from "./ChatWindow";

const { useAskChatbotMock } = vi.hoisted(() => ({ useAskChatbotMock: vi.fn() }));

vi.mock("../../hooks/useAskChatbot", () => ({
  useAskChatbot: useAskChatbotMock,
}));

function mockHook(overrides: Partial<ReturnType<typeof useAskChatbotMock>> = {}) {
  useAskChatbotMock.mockReturnValue({
    ask: vi.fn().mockResolvedValue({ answer: "ok" }),
    isSending: false,
    error: null,
    reset: vi.fn(),
    ...overrides,
  });
}

/** A thin stateful harness mirroring AskAiChat's own message ownership, so
 * ChatWindow's onAppendMessage -> re-render loop can be exercised like the
 * real app does, without pulling in AskAiChat/sessionStorage. */
function Harness({ initialMessages = [] as ChatbotHistoryMessage[] }) {
  const [messages, setMessages] = useState<ChatbotHistoryMessage[]>(initialMessages);
  return (
    <ChatWindow
      messages={messages}
      onAppendMessage={(message) => setMessages((previous) => [...previous, message])}
      onClose={vi.fn()}
    />
  );
}

describe("ChatWindow", () => {
  beforeEach(() => {
    mockHook();
    Element.prototype.setPointerCapture = vi.fn();
    Element.prototype.releasePointerCapture = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the empty-state introduction when there are no messages", () => {
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    expect(
      screen.getByText("Ask EcoGuard AI about active fire events, severity, spread predictions, or response plans."),
    ).toBeInTheDocument();
  });

  it("renders the EcoGuard AI title and a close button", () => {
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "EcoGuard AI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close EcoGuard AI chat" })).toBeInTheDocument();
  });

  it("close button calls onClose", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: "Close EcoGuard AI chat" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("lets the user type a question", async () => {
    const user = userEvent.setup();
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    await user.type(textarea, "What are the active fires?");

    expect(textarea).toHaveValue("What are the active fires?");
  });

  it("disables Send while the input is blank", () => {
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("disables Send while the input is whitespace-only", async () => {
    const user = userEvent.setup();
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    await user.type(screen.getByLabelText("Your question"), "   ");

    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("Enter sends the question", async () => {
    const askMock = vi.fn().mockResolvedValue({ answer: "ok" });
    mockHook({ ask: askMock });
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(screen.getByLabelText("Your question"), "Status?{Enter}");

    expect(askMock).toHaveBeenCalledWith({ question: "Status?", history: [] });
  });

  it("Shift+Enter inserts a newline instead of sending", async () => {
    const askMock = vi.fn().mockResolvedValue({ answer: "ok" });
    mockHook({ ask: askMock });
    const user = userEvent.setup();
    render(<Harness />);

    const textarea = screen.getByLabelText("Your question");
    await user.type(textarea, "line1{Shift>}{Enter}{/Shift}line2");

    expect(askMock).not.toHaveBeenCalled();
    expect(textarea).toHaveValue("line1\nline2");
  });

  it("clears the textarea after sending", async () => {
    mockHook({ ask: vi.fn().mockResolvedValue({ answer: "ok" }) });
    const user = userEvent.setup();
    render(<Harness />);

    const textarea = screen.getByLabelText("Your question");
    await user.type(textarea, "Status?{Enter}");

    expect(textarea).toHaveValue("");
  });

  it("shows a Thinking status while a request is in flight", () => {
    mockHook({ isSending: true });
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    expect(screen.getByRole("status")).toHaveTextContent("Thinking…");
  });

  it("disables Send while a request is already in flight (no duplicate submissions)", async () => {
    const askMock = vi.fn();
    mockHook({ isSending: true, ask: askMock });
    const user = userEvent.setup();
    render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

    const textarea = screen.getByLabelText("Your question");
    await user.type(textarea, "Status?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(askMock).not.toHaveBeenCalled();
  });

  it("renders the assistant's answer after a successful send", async () => {
    mockHook({ ask: vi.fn().mockResolvedValue({ answer: "The Galilee fire is confirmed." }) });
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(screen.getByLabelText("Your question"), "Status?{Enter}");

    expect(await screen.findByText("The Galilee fire is confirmed.")).toBeInTheDocument();
  });

  it("renders a Hebrew assistant message with dir=auto on its content container", () => {
    render(
      <ChatWindow
        messages={[{ role: "assistant", content: "מה מצב השריפה בחיפה?" }]}
        onAppendMessage={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    const textNode = screen.getByText("מה מצב השריפה בחיפה?");
    expect(textNode.closest(".chat-bubble__markdown")).toHaveAttribute("dir", "auto");
  });

  it("renders an English message with dir=auto", () => {
    render(
      <ChatWindow
        messages={[{ role: "user", content: "What are the active fires?" }]}
        onAppendMessage={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText("What are the active fires?")).toHaveAttribute("dir", "auto");
  });

  it("never renders fire_event_id or other internal metadata from a message", () => {
    render(
      <ChatWindow
        messages={[{ role: "assistant", content: "The fire near Haifa is confirmed." }]}
        onAppendMessage={vi.fn()}
        onClose={vi.fn()}
      />,
    );

    expect(screen.queryByText(/fire_event_id/i)).not.toBeInTheDocument();
  });

  describe("assistant Markdown rendering", () => {
    it("renders a ### heading as a real heading, not the literal ### characters", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "### Event summary" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      expect(screen.getByRole("heading", { name: "Event summary" })).toBeInTheDocument();
      expect(screen.queryByText(/###/)).not.toBeInTheDocument();
    });

    it("renders **bold** as strong text, not the literal ** characters", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "**Severity:** critical" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      const strong = screen.getByText("Severity:");
      expect(strong.tagName).toBe("STRONG");
      expect(screen.queryByText(/\*\*/)).not.toBeInTheDocument();
    });

    it("renders an unordered list as real list items", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "* Satellite detection\n* Active response plan" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      const list = screen.getByRole("list");
      expect(list.tagName).toBe("UL");
      expect(screen.getAllByRole("listitem")).toHaveLength(2);
      expect(screen.getByText("Satellite detection")).toBeInTheDocument();
      expect(screen.queryByText(/^\*/)).not.toBeInTheDocument();
    });

    it("renders an ordered list correctly", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "1. First step\n2. Second step" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      const list = screen.getByRole("list");
      expect(list.tagName).toBe("OL");
      expect(screen.getAllByRole("listitem")).toHaveLength(2);
    });

    it("never Markdown-renders user messages - they stay literal plain text", () => {
      render(
        <ChatWindow
          messages={[{ role: "user", content: "### not a heading and **not bold**" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      expect(screen.getByText("### not a heading and **not bold**")).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: /not a heading/ })).not.toBeInTheDocument();
    });

    it("renders a Hebrew Markdown response (heading + bold + list) correctly", () => {
      render(
        <ChatWindow
          messages={[
            {
              role: "assistant",
              content: "### אזור הגליל\n**הערכת חומרה:** קריטית\n\n* תצפית לוויין\n* תוכנית מענה פעילה",
            },
          ]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      expect(screen.getByRole("heading", { name: "אזור הגליל" })).toBeInTheDocument();
      const strong = screen.getByText("הערכת חומרה:");
      expect(strong.tagName).toBe("STRONG");
      expect(screen.getAllByRole("listitem")).toHaveLength(2);
      expect(screen.getByText("תצפית לוויין")).toBeInTheDocument();
      expect(screen.queryByText(/###|\*\*/)).not.toBeInTheDocument();
    });

    it("keeps dir=auto on the assistant Markdown content container", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "### Heading\n\nSome text." }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      const heading = screen.getByRole("heading", { name: "Heading" });
      expect(heading.closest(".chat-bubble__markdown")).toHaveAttribute("dir", "auto");
    });

    it("never uses dangerouslySetInnerHTML or renders raw HTML tags from response text", () => {
      render(
        <ChatWindow
          messages={[{ role: "assistant", content: "<script>window.__pwned = true;</script>" }]}
          onAppendMessage={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      expect((window as unknown as { __pwned?: boolean }).__pwned).toBeUndefined();
      expect(document.querySelector("script")).not.toBeInTheDocument();
    });
  });

  describe("history sent to the backend", () => {
    function historyMessages(count: number): ChatbotHistoryMessage[] {
      return Array.from({ length: count }, (_, index) => ({
        role: index % 2 === 0 ? "user" : "assistant",
        content: `message ${index}`,
      }));
    }

    it("sends the current question separately from history, never duplicated inside it", async () => {
      const askMock = vi.fn().mockResolvedValue({ answer: "ok" });
      mockHook({ ask: askMock });
      const user = userEvent.setup();
      render(<Harness initialMessages={historyMessages(3)} />);

      await user.type(screen.getByLabelText("Your question"), "New question{Enter}");

      const [request] = askMock.mock.calls[0] as [{ question: string; history: ChatbotHistoryMessage[] }];
      expect(request.question).toBe("New question");
      expect(request.history.some((message) => message.content === "New question")).toBe(false);
    });

    it("sends only the latest 8 previous messages as history", async () => {
      const askMock = vi.fn().mockResolvedValue({ answer: "ok" });
      mockHook({ ask: askMock });
      const user = userEvent.setup();
      render(<Harness initialMessages={historyMessages(10)} />);

      await user.type(screen.getByLabelText("Your question"), "New question{Enter}");

      const [request] = askMock.mock.calls[0] as [{ history: ChatbotHistoryMessage[] }];
      expect(request.history).toHaveLength(8);
      expect(request.history[0].content).toBe("message 2");
      expect(request.history[7].content).toBe("message 9");
    });

    it("preserves message order in history", async () => {
      const askMock = vi.fn().mockResolvedValue({ answer: "ok" });
      mockHook({ ask: askMock });
      const user = userEvent.setup();
      render(<Harness initialMessages={historyMessages(3)} />);

      await user.type(screen.getByLabelText("Your question"), "New question{Enter}");

      const [request] = askMock.mock.calls[0] as [{ history: ChatbotHistoryMessage[] }];
      expect(request.history.map((message) => message.content)).toEqual(["message 0", "message 1", "message 2"]);
    });
  });

  describe("errors", () => {
    it("shows a friendly message for a 503 CHATBOT_UNAVAILABLE error", () => {
      mockHook({ error: new ApiError("raw backend detail", 503, "CHATBOT_UNAVAILABLE") });
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      expect(screen.getByRole("alert")).toHaveTextContent("EcoGuard AI is temporarily unavailable. Please try again.");
    });

    it("shows a generic safe message for any other server error", () => {
      mockHook({ error: new ApiError("raw backend detail", 500, "CHATBOT_CONFIGURATION_ERROR") });
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      expect(screen.getByRole("alert")).toHaveTextContent(
        "EcoGuard AI could not complete the request. Please try again.",
      );
    });

    it("never renders the raw backend error message", () => {
      mockHook({ error: new ApiError("raw backend detail", 500, "CHATBOT_CONFIGURATION_ERROR") });
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      expect(screen.queryByText("raw backend detail")).not.toBeInTheDocument();
    });
  });

  describe("dragging", () => {
    it("pointer interaction on the header updates the window's position", () => {
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      const dialog = screen.getByRole("dialog");
      const header = dialog.querySelector(".chat-window__header") as HTMLElement;

      fireEvent.pointerDown(header, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(header, { pointerId: 1, clientX: 150, clientY: 130 });

      expect(dialog.style.left).not.toBe("");
      expect(dialog.style.top).not.toBe("");
    });

    it("pointer interaction on the message/input area does not move the window", () => {
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      const dialog = screen.getByRole("dialog");
      const textarea = screen.getByLabelText("Your question");

      fireEvent.pointerDown(textarea, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(textarea, { pointerId: 1, clientX: 150, clientY: 130 });

      expect(dialog.style.left).toBe("");
      expect(dialog.style.top).toBe("");
    });

    it("a pointer press on the close button never starts a drag (regression: pointer capture was swallowing the close click)", () => {
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      const dialog = screen.getByRole("dialog");
      const closeButton = screen.getByRole("button", { name: "Close EcoGuard AI chat" });

      fireEvent.pointerDown(closeButton, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(closeButton, { pointerId: 1, clientX: 150, clientY: 130 });

      expect(dialog.style.left).toBe("");
      expect(dialog.style.top).toBe("");
      expect(dialog).not.toHaveClass("chat-window--dragging");
    });
  });

  describe("close button (regression)", () => {
    it("clicking the close button always calls onClose", async () => {
      const user = userEvent.setup();
      const onClose = vi.fn();
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={onClose} />);

      await user.click(screen.getByRole("button", { name: "Close EcoGuard AI chat" }));

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("a full pointerdown-then-click sequence on the close button (as a real browser fires it) still calls onClose", () => {
      const onClose = vi.fn();
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={onClose} />);

      const closeButton = screen.getByRole("button", { name: "Close EcoGuard AI chat" });
      fireEvent.pointerDown(closeButton, { pointerId: 1, clientX: 380, clientY: 12 });
      fireEvent.pointerUp(closeButton, { pointerId: 1, clientX: 380, clientY: 12 });
      fireEvent.click(closeButton);

      expect(onClose).toHaveBeenCalledTimes(1);
    });

    it("the rest of the header still starts a drag normally after the close-button fix", () => {
      render(<ChatWindow messages={[]} onAppendMessage={vi.fn()} onClose={vi.fn()} />);

      const dialog = screen.getByRole("dialog");
      const header = dialog.querySelector(".chat-window__header") as HTMLElement;

      fireEvent.pointerDown(header, { pointerId: 1, clientX: 50, clientY: 12 });
      fireEvent.pointerMove(header, { pointerId: 1, clientX: 90, clientY: 20 });

      expect(dialog.style.left).not.toBe("");
      expect(dialog.style.top).not.toBe("");
    });
  });
});
