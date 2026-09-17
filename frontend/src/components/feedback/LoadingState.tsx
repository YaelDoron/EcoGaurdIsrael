import "./FeedbackStates.css";

export interface LoadingStateProps {
  message?: string;
}

const DEFAULT_MESSAGE = "Loading…";

/**
 * Generic, feature-agnostic loading indicator. `role="status"` (a polite
 * live region) lets assistive tech announce `message` without stealing
 * focus; the text itself always stays visible, never icon/spinner-only.
 */
export function LoadingState({ message = DEFAULT_MESSAGE }: LoadingStateProps) {
  return (
    <div className="feedback-state feedback-state--loading" role="status" aria-live="polite">
      <span className="feedback-state__spinner" aria-hidden="true" />
      <p className="feedback-state__message">{message}</p>
    </div>
  );
}
