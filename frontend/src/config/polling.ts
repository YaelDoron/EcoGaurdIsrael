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
