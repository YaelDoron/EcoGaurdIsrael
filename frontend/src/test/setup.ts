import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";
import "@testing-library/jest-dom/vitest";

// RTL's auto-cleanup relies on detecting global test hooks; with vitest's
// `globals: false` those aren't injected, so it is registered explicitly.
afterEach(() => {
  cleanup();
});
