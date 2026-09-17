import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

/**
 * Task 6 guard: production dashboard code must render only data returned by
 * the API - it must never contain a literal demo/example incident baked in.
 * Test fixtures (this suite included) are allowed to use these values;
 * production source files are not.
 */
const PRODUCTION_FILES = [
  "src/pages/ActiveWildfiresPage.tsx",
  "src/components/fire-events/ActiveFireEventCard.tsx",
  "src/hooks/useActiveFireEvents.ts",
  "src/api/activeFireEvents.ts",
];

const FORBIDDEN_LITERALS = ["Carmel", "Golan", "Jerusalem Forest", "Event #12", "32.731", "35.046"];

describe("no hardcoded demo FireEvent data in production dashboard code", () => {
  it.each(PRODUCTION_FILES)("%s contains no forbidden demo literals", (relativePath) => {
    const source = readFileSync(relativePath, "utf-8");

    for (const forbidden of FORBIDDEN_LITERALS) {
      expect(source).not.toContain(forbidden);
    }
  });
});
