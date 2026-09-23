import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AskAiButton } from "./AskAiButton";

function renderButton(overrides: { onClick?: () => void; onPositionChange?: (p: { top: number; left: number }) => void } = {}) {
  return render(
    <AskAiButton
      onClick={overrides.onClick ?? vi.fn()}
      position={null}
      onPositionChange={overrides.onPositionChange ?? vi.fn()}
    />,
  );
}

const originalInnerWidth = window.innerWidth;

function setViewportWidth(width: number): void {
  Object.defineProperty(window, "innerWidth", { writable: true, configurable: true, value: width });
}

describe("AskAiButton", () => {
  beforeEach(() => {
    Element.prototype.setPointerCapture = vi.fn();
    Element.prototype.releasePointerCapture = vi.fn();
  });

  afterEach(() => {
    setViewportWidth(originalInnerWidth);
  });

  it("renders a real button", () => {
    renderButton();

    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("shows the ASK AI label", () => {
    renderButton();

    expect(screen.getByText("ASK AI")).toBeInTheDocument();
  });

  it("has a useful accessible label", () => {
    renderButton();

    expect(screen.getByRole("button", { name: "Open EcoGuard AI chat" })).toBeInTheDocument();
  });

  it("calls onClick on a normal click", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    renderButton({ onClick });

    await user.click(screen.getByRole("button"));

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("calls onClick on keyboard activation (Enter), which never touches the pointer handlers", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    renderButton({ onClick });

    screen.getByRole("button").focus();
    await user.keyboard("{Enter}");

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("dragging past the threshold moves the button without opening the chat", () => {
    const onClick = vi.fn();
    const onPositionChange = vi.fn();
    renderButton({ onClick, onPositionChange });
    const button = screen.getByRole("button");

    fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(button, { pointerId: 1, clientX: 140, clientY: 100 }); // 40px > threshold
    fireEvent.pointerUp(button, { pointerId: 1, clientX: 140, clientY: 100 });
    fireEvent.click(button);

    expect(onPositionChange).toHaveBeenCalled();
    expect(onClick).not.toHaveBeenCalled();
  });

  it("small movement below the threshold is still treated as a click", () => {
    const onClick = vi.fn();
    const onPositionChange = vi.fn();
    renderButton({ onClick, onPositionChange });
    const button = screen.getByRole("button");

    fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
    fireEvent.pointerMove(button, { pointerId: 1, clientX: 102, clientY: 101 }); // well under the 5px threshold
    fireEvent.pointerUp(button, { pointerId: 1, clientX: 102, clientY: 101 });
    fireEvent.click(button);

    expect(onPositionChange).not.toHaveBeenCalled();
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("clamps the dragged position within the viewport", () => {
    const onPositionChange = vi.fn();
    renderButton({ onPositionChange });
    const button = screen.getByRole("button");

    fireEvent.pointerDown(button, { pointerId: 1, clientX: 10, clientY: 10 });
    // Drag far past the top-left corner of the viewport.
    fireEvent.pointerMove(button, { pointerId: 1, clientX: -500, clientY: -500 });

    const [position] = onPositionChange.mock.calls.at(-1) as [{ top: number; left: number }];
    expect(position.left).toBeGreaterThanOrEqual(0);
    expect(position.top).toBeGreaterThanOrEqual(0);
  });

  it("renders at the supplied position", () => {
    render(<AskAiButton onClick={vi.fn()} position={{ top: 120, left: 240 }} onPositionChange={vi.fn()} />);

    const button = screen.getByRole("button");
    expect(button.style.top).toBe("120px");
    expect(button.style.left).toBe("240px");
  });

  describe("below the 960px desktop-drag breakpoint", () => {
    beforeEach(() => {
      setViewportWidth(500);
    });

    it("pointer movement never triggers a drag or onPositionChange", () => {
      const onPositionChange = vi.fn();
      renderButton({ onPositionChange });
      const button = screen.getByRole("button");

      fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(button, { pointerId: 1, clientX: 200, clientY: 200 }); // well past the 5px threshold

      expect(onPositionChange).not.toHaveBeenCalled();
      expect(button).not.toHaveClass("ask-ai-button--dragging");
    });

    it("pointer movement followed by click still opens the chat normally", () => {
      const onClick = vi.fn();
      const onPositionChange = vi.fn();
      renderButton({ onClick, onPositionChange });
      const button = screen.getByRole("button");

      fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(button, { pointerId: 1, clientX: 200, clientY: 200 });
      fireEvent.pointerUp(button, { pointerId: 1, clientX: 200, clientY: 200 });
      fireEvent.click(button);

      expect(onPositionChange).not.toHaveBeenCalled();
      expect(onClick).toHaveBeenCalledTimes(1);
    });
  });

  describe("at/above the 960px desktop-drag breakpoint", () => {
    beforeEach(() => {
      setViewportWidth(1280);
    });

    it("drag behavior is unchanged: movement past the threshold moves the button without opening the chat", () => {
      const onClick = vi.fn();
      const onPositionChange = vi.fn();
      renderButton({ onClick, onPositionChange });
      const button = screen.getByRole("button");

      fireEvent.pointerDown(button, { pointerId: 1, clientX: 100, clientY: 100 });
      fireEvent.pointerMove(button, { pointerId: 1, clientX: 200, clientY: 200 });
      fireEvent.pointerUp(button, { pointerId: 1, clientX: 200, clientY: 200 });
      fireEvent.click(button);

      expect(onPositionChange).toHaveBeenCalled();
      expect(onClick).not.toHaveBeenCalled();
    });
  });
});
