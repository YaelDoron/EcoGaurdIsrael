import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";
import { clearEventLocationNameCache } from "../hooks/eventLocationNameCache";
import { endDemoRunStart } from "../hooks/demoRunStart";
import { clearOverviewSnapshot } from "../hooks/operationsOverviewSnapshot";

// RTL's auto-cleanup relies on detecting global test hooks; with vitest's
// `globals: false` those aren't injected, so it is registered explicitly.
afterEach(() => {
  cleanup();
  // Session-lifetime name cache (hooks/eventLocationNameCache.ts) - isolate tests.
  clearEventLocationNameCache();
  clearOverviewSnapshot();
  endDemoRunStart();
});
