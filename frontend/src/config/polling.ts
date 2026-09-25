/**
 * Centralized polling configuration (Task A7).
 *
 * No frontend config module existed before this task - this is the one
 * place a poll interval is defined; do not scatter a literal `5000`/similar
 * across components or hooks.
 *
 * `OPERATIONS_OVERVIEW_POLL_INTERVAL_MS` defaults to 5 seconds because a
 * real `GET /api/v1/operations/overview` request was measured at
 * 3.1-4.6 seconds against the demo database (Task A6 live verification) -
 * this is primarily remote DB round-trip latency, not a fixable frontend
 * concern. A naive fixed 2-second poll would frequently overlap with the
 * previous still-in-flight request; see useOperationsOverview.ts for how
 * overlap is prevented regardless (the interval alone is not the only
 * safeguard).
 */
export const OPERATIONS_OVERVIEW_POLL_INTERVAL_MS = 5000;

/**
 * How long useEventDetails waits after one Event Details request settles
 * before fetching again, while the event is still active (suspected or
 * confirmed) - so a status change (e.g. "Monitoring" -> "Pending") and a
 * newly generated response plan show up without navigating away and back.
 */
export const EVENT_DETAILS_POLL_INTERVAL_MS = 5000;

/**
 * How long useGlobalResponsePlan waits after one request settles before
 * fetching the current Global Response Plan again, so a newly materialized
 * generation (e.g. after new events are confirmed) shows up without a
 * manual page reload.
 */
export const GLOBAL_RESPONSE_PLAN_POLL_INTERVAL_MS = 5000;
