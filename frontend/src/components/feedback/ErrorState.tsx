import "./FeedbackStates.css";

export interface ErrorStateProps {
  title: string;
  message?: string;
  onRetry?: () => void;
  retryLabel?: string;
}

const DEFAULT_RETRY_LABEL = "Retry";

/**
 * Generic, feature-agnostic error presentation. Only ever renders the
 * `title`/`message` strings it is given - it never inspects an Error
 * object, parses a backend error body, or renders a stack trace. Choosing
 * a safe message is the caller's responsibility.
 */
export function ErrorState({ title, message, onRetry, retryLabel = DEFAULT_RETRY_LABEL }: ErrorStateProps) {
  return (
    <div className="feedback-state feedback-state--error" role="alert">
      <h2 className="feedback-state__title">{title}</h2>
      {message ? <p className="feedback-state__message">{message}</p> : null}
      {onRetry ? (
        <button type="button" className="feedback-state__action" onClick={onRetry}>
          {retryLabel}
        </button>
      ) : null}
    </div>
  );
}
