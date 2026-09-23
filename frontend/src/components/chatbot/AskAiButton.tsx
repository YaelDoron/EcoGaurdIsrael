import { useRef, useState, type PointerEvent } from "react";
import "./AskAiButton.css";

// A press that never moves more than this many pixels is a click/tap, not
// a drag - keeps a slightly shaky tap from being misread as a drag, and
// (via didDragRef in handleClick) stops a real drag's release from also
// opening the chat.
const DRAG_THRESHOLD_PX = 5;

// Matches the project's existing 960px responsive breakpoint (see
// OperationsActivityDrawer.css/ChatWindow.css). Free dragging is
// desktop-only; read live via window.innerWidth at interaction time (no
// resize listener, no stored/persisted breakpoint state) so a touch/small
// viewport never starts tracking a drag in the first place.
const DESKTOP_DRAG_BREAKPOINT_PX = 960;

function isDesktopViewport(): boolean {
  return window.innerWidth >= DESKTOP_DRAG_BREAKPOINT_PX;
}

export interface AskAiButtonPosition {
  top: number;
  left: number;
}

export interface AskAiButtonProps {
  onClick: () => void;
  /** Current position, or `null` to use the default bottom-right CSS position. Owned by the parent (`AskAiChat`) so it survives this component unmounting while the chat is open. */
  position: AskAiButtonPosition | null;
  onPositionChange: (position: AskAiButtonPosition) => void;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Global floating trigger for the EcoGuard AI chat (Task 7B/7C). Fixed by
 * default, bottom-right on every screen; draggable with native Pointer
 * Events while the chat is closed (no dependency). A drag release never
 * also opens the chat, and a plain click/tap (including keyboard
 * activation, which never touches these pointer handlers at all) always
 * does.
 */
export function AskAiButton({ onClick, position, onPositionChange }: AskAiButtonProps) {
  const [isDragging, setIsDragging] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const pointerStateRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    offsetX: number;
    offsetY: number;
  } | null>(null);
  const didDragRef = useRef(false);

  function handlePointerDown(event: PointerEvent<HTMLButtonElement>) {
    if (!isDesktopViewport()) {
      // Below the breakpoint the button is a plain, non-draggable button -
      // never start tracking, so pointer movement can never be mistaken
      // for a drag and a tap always opens the chat via the browser's own
      // native click.
      return;
    }
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    didDragRef.current = false;
    pointerStateRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture isn't supported everywhere - dragging still works via move/up below either way.
    }
  }

  function handlePointerMove(event: PointerEvent<HTMLButtonElement>) {
    const state = pointerStateRef.current;
    if (!state || state.pointerId !== event.pointerId) {
      return;
    }
    const dx = event.clientX - state.startX;
    const dy = event.clientY - state.startY;
    if (!didDragRef.current && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) {
      return;
    }
    didDragRef.current = true;
    setIsDragging(true);

    const rect = buttonRef.current?.getBoundingClientRect();
    const width = rect?.width ?? 0;
    const height = rect?.height ?? 0;
    const maxLeft = Math.max(0, window.innerWidth - width);
    const maxTop = Math.max(0, window.innerHeight - height);
    onPositionChange({
      left: clamp(event.clientX - state.offsetX, 0, maxLeft),
      top: clamp(event.clientY - state.offsetY, 0, maxTop),
    });
  }

  function endDrag(event: PointerEvent<HTMLButtonElement>) {
    pointerStateRef.current = null;
    setIsDragging(false);
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // See handlePointerDown.
    }
  }

  function handleClick() {
    // A drag's pointerup still fires a native click right after - swallow
    // exactly that one so dragging never accidentally opens the chat.
    if (didDragRef.current) {
      didDragRef.current = false;
      return;
    }
    onClick();
  }

  return (
    <button
      ref={buttonRef}
      type="button"
      className={`ask-ai-button${isDragging ? " ask-ai-button--dragging" : ""}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onClick={handleClick}
      aria-label="Open EcoGuard AI chat"
      style={position ? { top: position.top, left: position.left, right: "auto", bottom: "auto" } : undefined}
    >
      <svg className="ask-ai-button__icon" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 2 L14.5 9.5 L22 12 L14.5 14.5 L12 22 L9.5 14.5 L2 12 L9.5 9.5 Z" fill="currentColor" />
      </svg>
      <span>ASK AI</span>
    </button>
  );
}
