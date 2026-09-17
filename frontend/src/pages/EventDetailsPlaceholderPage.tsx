import { useParams } from "react-router-dom";

/**
 * Reserved for US 6.2 (Event Details). This route/page intentionally
 * renders no real content yet.
 */
export function EventDetailsPlaceholderPage() {
  const { fireEventId } = useParams<{ fireEventId: string }>();

  return (
    <section aria-labelledby="event-details-heading">
      <h1 id="event-details-heading">Event Details</h1>
      <p>This view is not implemented yet. It is reserved for User Story 6.2.</p>
      {fireEventId ? <p>Fire event: {fireEventId}</p> : null}
    </section>
  );
}
