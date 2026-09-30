import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../../api/errors";
import { SimulationControl } from "../dashboard/SimulationControl";
import { AskAiChat } from "./AskAiChat";

const { askChatbotMock, startSimulationMock } = vi.hoisted(() => ({
  askChatbotMock: vi.fn(),
  startSimulationMock: vi.fn(),
}));

vi.mock("../../api/chatbot", () => ({
  askChatbot: askChatbotMock,
}));

vi.mock("../../api/simulation", () => ({
  startSimulation: startSimulationMock,
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

describe("AskAiChat conversation reset on a new simulation run", () => {
  const PREVIOUS_RUN_HISTORY = [
    { role: "user", content: "What is the severity in Galilee?" },
    { role: "assistant", content: "The Galilee fire is critical." },
  ];
  let runCounter = 0;

  function acceptedRun(runId: string) {
    return {
      run_id: runId,
      state: "preparing",
      preset_id: "presentation_demo",
      seed: 1,
      mode: "automatic",
      simulation_duration_seconds: 1800,
      events_total: 70,
      events_completed: 0,
      events_succeeded: 0,
      events_failed: 0,
      current_event: null,
      started_at: null,
      completed_at: null,
      wall_clock_elapsed_seconds: 0,
      last_message: null,
      error: null,
    };
  }

  /** AskAiChat (mounted once by AppShell) alongside the dashboard's real Start Simulation control. */
  function renderChatWithSimulationControl() {
    return render(
      <>
        <SimulationControl run={null} enabled onRequestOverviewRefresh={vi.fn()} />
        <AskAiChat />
      </>,
    );
  }

  beforeEach(() => {
    window.sessionStorage.clear();
    startSimulationMock.mockReset();
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(PREVIOUS_RUN_HISTORY));
    runCounter += 1;
  });

  it("shows the previous run's history before any new simulation is started", async () => {
    const user = userEvent.setup();
    renderChatWithSimulationControl();

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByText("What is the severity in Galilee?")).toBeInTheDocument();
    expect(screen.getByText("The Galilee fire is critical.")).toBeInTheDocument();
  });

  it("a successful Start Simulation clears the visible messages and removes the stored history", async () => {
    startSimulationMock.mockResolvedValue(acceptedRun(`run-success-${runCounter}`));
    const user = userEvent.setup();
    renderChatWithSimulationControl();
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    expect(screen.getByText("The Galilee fire is critical.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(screen.queryByText("The Galilee fire is critical.")).not.toBeInTheDocument());
    expect(screen.queryByText("What is the severity in Galilee?")).not.toBeInTheDocument();
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("a failed Start Simulation keeps the existing conversation and stored history", async () => {
    startSimulationMock.mockRejectedValue(
      new ApiError("A simulation is already running.", 409, "SIMULATION_ALREADY_RUNNING"),
    );
    const user = userEvent.setup();
    renderChatWithSimulationControl();
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    await user.click(screen.getByRole("button", { name: "Start Simulation" }));

    await waitFor(() => expect(startSimulationMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByRole("button", { name: "Start Simulation" })).toBeEnabled());
    expect(screen.getByText("The Galilee fire is critical.")).toBeInTheDocument();
    expect(JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "[]")).toEqual(PREVIOUS_RUN_HISTORY);
  });

  it("a conversation started after the new run began is kept, even when the chat remounts", async () => {
    startSimulationMock.mockResolvedValue(acceptedRun(`run-remount-${runCounter}`));
    askChatbotMock.mockResolvedValue({ answer: "Jerusalem Forest is 34.2°C." });
    const user = userEvent.setup();
    const { unmount } = renderChatWithSimulationControl();
    await user.click(screen.getByRole("button", { name: "Start Simulation" }));
    await waitFor(() => expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull());

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    await user.type(screen.getByLabelText("Your question"), "Weather in Jerusalem?{Enter}");
    await screen.findByText("Jerusalem Forest is 34.2°C.");

    // Same run still pending: a remount (e.g. layout re-render) must not clear it again.
    unmount();
    renderChatWithSimulationControl();
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByText("Weather in Jerusalem?")).toBeInTheDocument();
    expect(screen.getByText("Jerusalem Forest is 34.2°C.")).toBeInTheDocument();
  });

  it("a page refresh / navigation remount without a new simulation start preserves the history", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<AskAiChat />);
    unmount();

    render(<AskAiChat />);
    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));

    expect(screen.getByText("What is the severity in Galilee?")).toBeInTheDocument();
    expect(JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "[]")).toEqual(PREVIOUS_RUN_HISTORY);
    expect(startSimulationMock).not.toHaveBeenCalled();
  });
});
