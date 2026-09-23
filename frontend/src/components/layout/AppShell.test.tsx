import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

const { askChatbotMock } = vi.hoisted(() => ({ askChatbotMock: vi.fn() }));

vi.mock("../../api/chatbot", () => ({
  askChatbot: askChatbotMock,
}));

function renderShell() {
  return render(
    <MemoryRouter initialEntries={["/events"]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/events" element={<p>content</p>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

function renderShellWithTwoRoutes() {
  return render(
    <MemoryRouter initialEntries={["/events"]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/events" element={<Link to="/other">Go to other</Link>} />
          <Route path="/other" element={<Link to="/events">Go to events</Link>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("AppShell", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    askChatbotMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows the EcoGuard Israel brand", () => {
    renderShell();

    expect(screen.getByText("EcoGuard Israel")).toBeInTheDocument();
    expect(screen.getByText("Wildfire Operations")).toBeInTheDocument();
  });

  it("has an Operations Overview navigation link", () => {
    renderShell();

    const link = screen.getByRole("link", { name: "Operations Overview" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/events");
  });

  it("does not render the old History/Monitoring placeholder items", () => {
    renderShell();

    for (const forbidden of [/history/i, /monitoring/i, /coming soon/i]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });

  it("renders exactly one navigation link", () => {
    renderShell();

    expect(screen.getAllByRole("link")).toHaveLength(1);
  });

  it("uses a semantic nav landmark for navigation", () => {
    renderShell();

    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
  });

  it("renders routed content inside main", () => {
    renderShell();

    expect(screen.getByRole("main")).toHaveTextContent("content");
  });

  it("mounts the EcoGuard AI chat widget once via the shell, available on every page", () => {
    renderShell();

    expect(screen.getByRole("button", { name: "Open EcoGuard AI chat" })).toBeInTheDocument();
  });

  it("keeps the chat conversation when navigating between routes", async () => {
    askChatbotMock.mockResolvedValue({ answer: "Two active fires." });
    const user = userEvent.setup();
    renderShellWithTwoRoutes();

    await user.click(screen.getByRole("button", { name: "Open EcoGuard AI chat" }));
    await user.type(screen.getByLabelText("Your question"), "Status?{Enter}");
    await screen.findByText("Two active fires.");

    await user.click(screen.getByRole("link", { name: "Go to other" }));

    expect(screen.getByText("Status?")).toBeInTheDocument();
    expect(screen.getByText("Two active fires.")).toBeInTheDocument();
  });

  it("keeps the ASK AI button's dragged position when navigating between routes", async () => {
    const user = userEvent.setup();
    Element.prototype.setPointerCapture = vi.fn();
    Element.prototype.releasePointerCapture = vi.fn();
    renderShellWithTwoRoutes();

    const button = screen.getByRole("button", { name: "Open EcoGuard AI chat" });
    fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(button, { pointerId: 1, clientX: 220, clientY: 160 });
    fireEvent.pointerUp(button, { pointerId: 1, clientX: 220, clientY: 160 });
    const draggedLeft = button.style.left;
    expect(draggedLeft).not.toBe("");

    await user.click(screen.getByRole("link", { name: "Go to other" }));

    expect(screen.getByRole("button", { name: "Open EcoGuard AI chat" }).style.left).toBe(draggedLeft);
  });
});
