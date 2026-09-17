import type { ActiveFireEventsResponse } from "../types/activeFireEvents";
import { apiGet } from "./client";

const ACTIVE_FIRE_EVENTS_PATH = "/api/v1/fire-events/active";

/**
 * Fetch the currently active wildfire events. Thin wrapper over the
 * generic `apiGet` client - no fetch call, base-URL handling, retry, or
 * caching logic lives here (see Task 4's client.ts for that).
 */
export function getActiveFireEvents(): Promise<ActiveFireEventsResponse> {
  return apiGet<ActiveFireEventsResponse>(ACTIVE_FIRE_EVENTS_PATH);
}
