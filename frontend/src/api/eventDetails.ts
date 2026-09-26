import type { EventDetailsResult } from "../types/eventDetails";
import { apiGet } from "./client";

/**
 * Fetch one wildfire event's full details. Thin wrapper over the generic
 * `apiGet` client - no fetch call, base-URL handling, retry, or caching
 * logic lives here (see client.ts for that). A 404 (unknown fire_event_id)
 * surfaces as a rejected promise carrying an `ApiError` with `status === 404`
 * - callers (see useEventDetails) distinguish that from other failures.
 */
export function getEventDetails(fireEventId: number, signal?: AbortSignal): Promise<EventDetailsResult> {
  return apiGet<EventDetailsResult>(`/api/v1/fire-events/${fireEventId}/details`, { signal });
}
