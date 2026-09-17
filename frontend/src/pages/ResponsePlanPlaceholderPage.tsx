import { useParams } from "react-router-dom";

/**
 * Reserved for US 6.3 (Response Plan). Shared by both
 * `/events/:fireEventId/plan` and `/plans/:planId` - this route/page
 * intentionally renders no real content yet.
 */
export function ResponsePlanPlaceholderPage() {
  const { fireEventId, planId } = useParams<{ fireEventId?: string; planId?: string }>();

  return (
    <section aria-labelledby="response-plan-heading">
      <h1 id="response-plan-heading">Response Plan</h1>
      <p>This view is not implemented yet. It is reserved for User Story 6.3.</p>
      {fireEventId ? <p>Fire event: {fireEventId}</p> : null}
      {planId ? <p>Plan: {planId}</p> : null}
    </section>
  );
}
