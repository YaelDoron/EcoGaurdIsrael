import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AskAiChat } from "./AskAiChat";

const { askChatbotMock } = vi.hoisted(() => ({ askChatbotMock: vi.fn() }));

vi.mock("../../api/chatbot", () => ({
  askChatbot: askChatbotMock,
}));

const STORAGE_KEY = "ecoguard-ai-chat-history";

describe("AskAiChat", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    askChatbotMock.mockReset();
    Element.prototype.setPointerCapture = vi.fn();
    Element.prototype.releasePointerCapture = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the ASK AI button when closed", () => {
    render(<AskAiChat />);

    expect(screen.getByRole("button", { name: "Open EcoGuard AI chat" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens the chat window and hides the trigger button on click", async () => {
    const user = userEvent.setup();
    render(<AskAiChat />);

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Open EcoGuard AI chat" })).not.toBeInTheDocument();
  });

  it("closing the chat shows the trigger button again", async () => {
    const user = userEvent.setup();
    render(<AskAiChat />);
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    await user.click(screen.getByRole("button", { name: "Close EcoGuard AI chat" }));

    expect(screen.getByRole("button", { name: "Open EcoGuard AI chat" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("preserves the conversation across close and reopen", async () => {
    askChatbotMock.mockResolvedValue({ answer: "The Galilee fire is confirmed." });
    const user = userEvent.setup();
    render(<AskAiChat />);

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    await user.type(screen.getByLabelText("Your question"), "Status?{Enter}");
    await screen.findByText("The Galilee fire is confirmed.");

    await user.click(screen.getByRole("button", { name: "Close EcoGuard AI chat" }));
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByText("Status?")).toBeInTheDocument();
    expect(screen.getByText("The Galilee fire is confirmed.")).toBeInTheDocument();
  });

  it("restores a conversation from valid sessionStorage on mount", async () => {
    window.sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify([
        { role: "user", content: "Status?" },
        { role: "assistant", content: "Two active fires." },
      ]),
    );
    const user = userEvent.setup();
    render(<AskAiChat />);

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByText("Status?")).toBeInTheDocument();
    expect(screen.getByText("Two active fires.")).toBeInTheDocument();
  });

  it("falls back to an empty conversation when sessionStorage is malformed, without crashing", async () => {
    window.sessionStorage.setItem(STORAGE_KEY, "{not valid json");
    const user = userEvent.setup();

    expect(() => render(<AskAiChat />)).not.toThrow();
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(
      screen.getByText("Ask EcoGuard AI about active fire events, severity, spread predictions, or response plans."),
    ).toBeInTheDocument();
  });

  it("falls back to an empty conversation when sessionStorage holds an unexpected shape", async () => {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ not: "an array" }));
    const user = userEvent.setup();
    render(<AskAiChat />);

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(
      screen.getByText("Ask EcoGuard AI about active fire events, severity, spread predictions, or response plans."),
    ).toBeInTheDocument();
  });

  it("writes the conversation to sessionStorage under the documented key after a new message", async () => {
    askChatbotMock.mockResolvedValue({ answer: "Two active fires." });
    const user = userEvent.setup();
    render(<AskAiChat />);

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    await user.type(screen.getByLabelText("Your question"), "Status?{Enter}");
    await screen.findByText("Two active fires.");

    const stored = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "[]");
    expect(stored).toEqual([
      { role: "user", content: "Status?" },
      { role: "assistant", content: "Two active fires." },
    ]);
  });

  it("the ASK AI button's dragged position survives closing and reopening the chat", async () => {
    const user = userEvent.setup();
    render(<AskAiChat />);

    const button = screen.getByRole("button", { name: "Open EcoGuard AI chat" });
    fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(button, { pointerId: 1, clientX: 260, clientY: 180 });
    fireEvent.pointerUp(button, { pointerId: 1, clientX: 260, clientY: 180 });

    const draggedLeft = button.style.left;
    const draggedTop = button.style.top;
    expect(draggedLeft).not.toBe("");

    // The drag above suppresses its own click (see AskAiButton) - open via
    // a fresh, separate click, matching how a real user would afterwards.
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    await user.click(screen.getByRole("button", { name: "Close EcoGuard AI chat" }));

    const reopenedButton = screen.getByRole("button", { name: "Open EcoGuard AI chat" });
    expect(reopenedButton.style.left).toBe(draggedLeft);
    expect(reopenedButton.style.top).toBe(draggedTop);
  });
});
