import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The ASK AI trigger is a compact, single-line pill on every screen. jsdom
 * does not lay out CSS, so this guards the stylesheet rules that keep it so.
 */
const css = readFileSync(join(__dirname, "AskAiButton.css"), "utf8");

function baseRule(): string {
  const match = css.match(/\n\.ask-ai-button\s*\{([^}]*)\}/);
  if (!match) {
    throw new Error(".ask-ai-button rule not found");
  }
  return match[1];
}

describe("AskAiButton styles", () => {
  it("never wraps the ASK AI label", () => {
    expect(baseRule()).toMatch(/white-space:\s*nowrap/);
    expect(baseRule()).toMatch(/width:\s*max-content/);
  });

  it("uses the compact padding and font size", () => {
    expect(baseRule()).toMatch(/padding:\s*var\(--space-2\)\s+var\(--space-4\)/);
    expect(baseRule()).toMatch(/font-size:\s*0\.875rem/);
  });
});
