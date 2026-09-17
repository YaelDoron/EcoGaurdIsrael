import type { ReactNode } from "react";
import "./FeedbackStates.css";

export interface EmptyStateProps {
  title: string;
  message?: string;
  action?: ReactNode;
}

/**
 * Generic, feature-agnostic "nothing to show" presentation. Performs no
 * business calculation - the caller decides what counts as empty and
 * supplies the title/message accordingly.
 */
export function EmptyState({ title, message, action }: EmptyStateProps) {
  return (
    <div className="feedback-state feedback-state--empty">
      <h2 className="feedback-state__title">{title}</h2>
      {message ? <p className="feedback-state__message">{message}</p> : null}
      {action ? <div className="feedback-state__action-slot">{action}</div> : null}
    </div>
  );
}
