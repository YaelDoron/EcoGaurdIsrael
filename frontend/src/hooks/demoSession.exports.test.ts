import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import * as demoSession from "./demoSession";

/**
 * Export contract for hooks/demoSession.ts. A named import of a name the module
 * does not export is a hard ES-module link error in the browser that blanks the
 * whole app (Vite does not type-check), so every `import { ... } from
 * ".../demoSession"` in production source must resolve against the REAL module.
 */
const SRC_DIR = join(__dirname, "..");
const IMPORT_PATTERN = /import\s*\{([^}]*)\}\s*from\s*["'][./]+(?:hooks\/)?demoSession["']/g;

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      return sourceFiles(path);
    }
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [path] : [];
  });
}

function namedImports(source: string): string[] {
  return [...source.matchAll(IMPORT_PATTERN)].flatMap((match) =>
    match[1]
      .split(",")
      .map((part) => part.trim().replace(/^type\s+/, "").split(/\s+as\s+/)[0])
      .filter((name) => name.length > 0),
  );
}

describe("demoSession export contract", () => {
  it("exports the demo-session API used by the pages", () => {
    expect(typeof demoSession.useCurrentDemoSimulation).toBe("function");
    expect(typeof demoSession.useDemoDataVisibility).toBe("function");
    expect(typeof demoSession.isDemoDataVisible).toBe("function");
    expect(typeof demoSession.rememberDemoRun).toBe("function");
    expect(typeof demoSession.getRememberedDemoRunId).toBe("function");
  });

  it("every production import from demoSession names a real export", () => {
    const typeOnly = new Set(["DemoSimulationSummary"]);
    const missing: string[] = [];
    let importers = 0;
    for (const file of sourceFiles(SRC_DIR)) {
      const names = namedImports(readFileSync(file, "utf8"));
      importers += names.length > 0 ? 1 : 0;
      for (const name of names) {
        if (!typeOnly.has(name) && !(name in demoSession)) {
          missing.push(`${relative(SRC_DIR, file)}: ${name}`);
        }
      }
    }
    expect(importers).toBeGreaterThanOrEqual(3);
    expect(missing).toEqual([]);
  });
});
