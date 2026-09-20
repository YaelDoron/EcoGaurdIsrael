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
  "src/pages/EventDetailsPage.tsx",
  "src/hooks/useEventDetails.ts",
  "src/api/eventDetails.ts",
  "src/components/map/MapView.tsx",
  "src/components/map/FitBoundsToPoints.tsx",
  "src/components/map/FireEventMarker.tsx",
  "src/components/map/SpreadLayer.tsx",
  "src/components/map/ResponseTargetLayer.tsx",
  "src/components/map/StationLayer.tsx",
  "src/components/map/OperationalLayer.tsx",
  "src/pages/ResponsePlanPage.tsx",
  "src/components/response-plan/ResponsePlanMapLayer.tsx",
  "src/api/operations.ts",
  "src/api/simulation.ts",
  "src/hooks/useOperationsOverview.ts",
  "src/hooks/useOperationsActivityDetail.ts",
  "src/hooks/useStartSimulation.ts",
  // Task A8
  "src/components/status/fireDangerPresentation.ts",
  "src/components/map/FireDangerLayer.tsx",
  "src/components/map/activeFireIcon.ts",
  "src/components/map/ActiveFireMapLayer.tsx",
  "src/components/dashboard/OperationsMap.tsx",
  "src/components/dashboard/OperationsMapLegend.tsx",
  "src/components/dashboard/ActiveFiresPanel.tsx",
  "src/components/dashboard/activityFeedPresentation.ts",
  "src/components/dashboard/OperationsActivityItem.tsx",
  "src/components/dashboard/OperationsActivityFeed.tsx",
  "src/components/dashboard/OperationsActivityDrawer.tsx",
  "src/components/dashboard/OperationsStatusHeader.tsx",
  "src/components/dashboard/simulationStatusPresentation.ts",
  // Task A9
  "src/config/demoSimulation.ts",
  "src/components/dashboard/SimulationControl.tsx",
  // Visual polish pass
  "src/components/dashboard/ActivityTimestamp.tsx",
  // Interaction polish pass
  "src/components/dashboard/ViewResponsePlanAction.tsx",
  "src/components/map/DisableScrollWheelZoom.tsx",
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

/**
 * Task A8, Part 40: architecture guards for the redesigned Operations
 * Overview dashboard. These assert absence of things the task explicitly
 * forbids re-implementing or reintroducing on the frontend - not behavior,
 * which is covered by the component-level tests elsewhere.
 */
// Strips line and block comments so guard checks below only see real code,
// not documentation that legitimately names a forbidden concept to explain
// its absence.
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

describe("Task A8 architecture guards", () => {
  const combinedSource = PRODUCTION_FILES.map((path) => stripComments(readFileSync(path, "utf-8"))).join("\n");

  it("never maps a coordinate to a fabricated area/place name", () => {
    // The only source of an area's name is the backend's own `area_name`
    // field (FireDangerArea) - production code must never carry its own
    // coordinate -> place-name lookup table.
    for (const forbiddenIdentifier of ["coordinateToArea", "coordinatesToPlace", "latLngToArea", "AREA_NAME_BY_COORDINATE"]) {
      expect(combinedSource).not.toContain(forbiddenIdentifier);
    }
  });

  it("never re-implements Fire Danger or Fire Severity scoring/threshold logic", () => {
    // "FFWI" itself is now a legitimate, explicit UI label ("FFWI score" -
    // see FireDangerLayer's popup, Part C of this task's clarity pass) for
    // the persisted score's methodology name - never recalculated. These
    // patterns instead target actual re-implementation risk specifically.
    for (const forbidden of [
      "FFWICalculator",
      "calculateFFWI",
      "computeFFWI",
      "ffwiFormula",
      "Fosberg",
      "calculateFireDanger",
      "calculateSeverity",
      "computeFireDanger",
      "computeSeverity",
      "deriveLevel",
      "deriveSeverityLevel",
    ]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it('never introduces a "system_update" or other invented Activity Feed pseudo-category', () => {
    expect(combinedSource.toLowerCase()).not.toContain("system_update");
    expect(combinedSource.toLowerCase()).not.toContain("system update");
  });

  it("never adds a second dashboard route or page component", () => {
    const routerSource = readFileSync("src/router/AppRouter.tsx", "utf-8");
    for (const forbidden of ["OperationsOverviewPage", "OperationsDashboardV2", "DashboardNew", "ActiveWildfiresPage2"]) {
      expect(routerSource).not.toContain(forbidden);
    }
    // Exactly one element renders at "/events" (via the redirect from "/").
    expect(routerSource.match(/path="\/events"/g)).toHaveLength(1);
  });

  it("never opens a WebSocket or Server-Sent Events connection", () => {
    for (const forbidden of ["WebSocket(", "new WebSocket", "EventSource("]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never invents a frontend priority/risk/criticality score", () => {
    for (const forbidden of ["priorityScore", "riskRank", "criticalityScore"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });
});

/**
 * Task A9 architecture guards: the Start Simulation/Run Again workflow is
 * frontend wiring around the existing A3 Simulation Control API and A6
 * overview polling - it must never grow its own reset implementation,
 * cancellation controls, fake wall-clock ETA, or anomaly-detection
 * thresholds.
 */
describe("Task A9 architecture guards", () => {
  const combinedSource = PRODUCTION_FILES.map((path) => stripComments(readFileSync(path, "utf-8"))).join("\n");

  it("never implements a frontend demo-state reset - only the backend's own reset service does", () => {
    for (const forbidden of ["resetDemoState(", "/reset", "DELETE_DEMO", "clearDemoState"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never renders a Stop/Cancel/Pause simulation control", () => {
    for (const forbidden of ["Stop Simulation", "Cancel Simulation", "Pause Simulation"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never fabricates a wall-clock ETA/remaining-time estimate", () => {
    expect(combinedSource).not.toMatch(/\bETA\b/i);
    for (const forbidden of ["remaining", "seconds remaining", "minutes remaining"]) {
      expect(combinedSource.toLowerCase()).not.toContain(forbidden.toLowerCase());
    }
  });

  it("never derives an anomaly/critical/urgent classification from FRP, confidence, or news text", () => {
    for (const forbidden of [
      "isAnomalous",
      "isCritical",
      "isUrgent",
      "anomalyThreshold",
      "FRP_THRESHOLD",
      "CONFIDENCE_THRESHOLD",
      "classifyReport",
    ]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never exposes a preset picker, seed field, or reset checkbox in the UI", () => {
    for (const forbidden of ["<select", "presetOptions", "seedInput", "resetCheckbox"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("centralizes the demo request in one config constant, never duplicated inline", () => {
    // Every occurrence of the three demo-request literals together must be
    // the one DEMO_SIMULATION_REQUEST declaration - not re-typed elsewhere.
    const occurrences = combinedSource.match(/preset:\s*"operations_demo"/g) ?? [];
    expect(occurrences).toHaveLength(1);
  });
});

/**
 * Interaction/visual polish pass architecture guards: removing the
 * Activity Feed's textual type-abbreviation badges and Global Planning row
 * must not silently come back as source-level regressions.
 */
describe("Interaction polish pass architecture guards", () => {
  const combinedSource = PRODUCTION_FILES.map((path) => stripComments(readFileSync(path, "utf-8"))).join("\n");

  it("never renders a textual activity-type abbreviation badge (SEV/SAT/NEWS/FE/GP) as visible content", () => {
    // These were real glyphs on a REMOVED lookup table (ACTIVITY_TYPE_META) -
    // this guards against that table (or an equivalent) coming back.
    for (const forbidden of ['"SEV"', '"SAT"', '"NEWS"', '"FE"', '"GP"', "ACTIVITY_TYPE_META"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never gives the Activity Feed its own second scroll/grid presentation", () => {
    expect(combinedSource).not.toMatch(/grid-template-columns/);
  });

  it("never fabricates a global_planning_run response_plan_id instead of using the real persisted A5 value", () => {
    for (const forbidden of ["fabricatePlanId", "response_plan_ids[0] +", "response_plan_ids[0]+"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("never polls A5 GlobalPlanningRun detail on a timer - only a click can trigger it", () => {
    const viewResponsePlanSource = readFileSync("src/components/dashboard/ViewResponsePlanAction.tsx", "utf-8");
    expect(viewResponsePlanSource).not.toMatch(/setInterval|setTimeout/);
  });

  it("never renders a fire_severity item in the Activity Feed (Severity already lives on Active Fires cards)", () => {
    for (const forbidden of ["— Critical severity", "— High severity", "— Moderate severity", "— Low severity"]) {
      expect(combinedSource).not.toContain(forbidden);
    }
  });

  it("the feed's eligibility filter excludes fire_event, fire_severity, and global_planning_run", () => {
    const source = readFileSync("src/components/dashboard/OperationsActivityFeed.tsx", "utf-8");
    const filterFn = source.match(/function isEligibleForFeed[\s\S]*?\n}/)?.[0] ?? "";

    expect(filterFn).toContain('"global_planning_run"');
    expect(filterFn).toContain('"fire_severity"');
    expect(filterFn).toContain('"fire_event"');
  });
});

/**
 * Task 4, Part J: Activity Feed presentation strings we own must use a
 * normal ASCII hyphen-minus, never an em/en-style long dash. This is a
 * generated-string guard only - it must never be satisfied by mangling real
 * news headline text, which OperationsActivityItem.tsx passes straight
 * through unmodified (see the dedicated headline-passthrough test below).
 */
describe("Task 4: Activity Feed presentation strings use ASCII hyphens, never em/en-dashes", () => {
  it("no production dashboard file contains an em-dash (—) or en-dash (–)", () => {
    for (const path of PRODUCTION_FILES) {
      const source = readFileSync(path, "utf-8");
      expect(source).not.toContain("—");
      expect(source).not.toContain("–");
    }
  });

  it("never fabricates an availability timestamp - ActivityTimestamp is always given available_at, not occurred_at", () => {
    const source = readFileSync("src/components/dashboard/OperationsActivityItem.tsx", "utf-8");
    expect(source).toContain("item.available_at");
    expect(source).not.toContain("<ActivityTimestamp value={item.occurred_at}");
  });
});

/**
 * Map wheel-interaction architecture guards (Part A): the fix must use
 * Leaflet's own supported Handler API, never a hand-rolled DOM wheel
 * listener, `preventDefault()`, or CSS `pointer-events` trick.
 */
describe("Map wheel-interaction architecture guards", () => {
  it("disables scroll-wheel zoom only through Leaflet's own scrollWheelZoom.disable() API", () => {
    const source = stripComments(readFileSync("src/components/map/DisableScrollWheelZoom.tsx", "utf-8"));

    expect(source).toContain("scrollWheelZoom.disable()");
    expect(source).not.toMatch(/preventDefault/);
    expect(source).not.toMatch(/pointer-events/);
    expect(source).not.toMatch(/addEventListener\(\s*["']wheel["']/);
    expect(source).not.toMatch(/window\.scrollBy/);
  });

  it("never disables dragging or the zoom control on the Operations map", () => {
    const source = readFileSync("src/components/dashboard/OperationsMap.tsx", "utf-8");

    expect(source).not.toMatch(/dragging=\{?false/);
    expect(source).not.toMatch(/zoomControl=\{?false/);
  });

  it("only Operations Overview opts into disabled scroll-wheel zoom - MapView still defaults to enabled", () => {
    const mapViewSource = readFileSync("src/components/map/MapView.tsx", "utf-8");
    expect(mapViewSource).toContain("scrollWheelZoom = true");

    const eventDetailsSource = readFileSync("src/pages/EventDetailsPage.tsx", "utf-8");
    const responsePlanSource = readFileSync("src/pages/ResponsePlanPage.tsx", "utf-8");
    expect(eventDetailsSource).not.toMatch(/scrollWheelZoom/);
    expect(responsePlanSource).not.toMatch(/scrollWheelZoom/);
  });
});

/**
 * Satellite hotspot location enrichment guards (Part C): the frontend must
 * never derive a geographic name itself - only display a real persisted
 * `location_name` (or its absence) exactly as A6 provides it.
 */
describe("Satellite hotspot location presentation guards", () => {
  it("never contains a coordinate-to-area-name lookup table or threshold in the frontend", () => {
    const source = readFileSync("src/components/dashboard/OperationsActivityItem.tsx", "utf-8");
    for (const forbidden of [
      "AREA_NAME_BY_COORDINATE",
      "coordinateToArea",
      "latLngToArea",
      "haversine",
      "Haversine",
    ]) {
      expect(source).not.toContain(forbidden);
    }
  });
});

/**
 * Active Fire location guards: the same rule as satellite hotspots -
 * ActiveFireEventCard/ActiveFireMapLayer display only a real persisted
 * `location_name` (or its absence), never a frontend coordinate-to-name
 * derivation.
 */
describe("Active Fire location presentation guards", () => {
  it("never contains a coordinate-to-area-name lookup table, geocoding call, or haversine math in ActiveFireEventCard/ActiveFireMapLayer", () => {
    for (const path of ["src/components/fire-events/ActiveFireEventCard.tsx", "src/components/map/ActiveFireMapLayer.tsx"]) {
      const source = readFileSync(path, "utf-8");
      for (const forbidden of [
        "AREA_NAME_BY_COORDINATE",
        "coordinateToArea",
        "latLngToArea",
        "haversine",
        "Haversine",
        "geocode",
        "Geocode",
      ]) {
        expect(source).not.toContain(forbidden);
      }
    }
  });
});

/**
 * Timezone display guards: satellite hotspot timestamps were ~3h off
 * because a naive backend value lost its UTC offset before reaching the
 * frontend (see backend fix: satellite_hotspots.detected_at is now
 * TIMESTAMPTZ). The frontend fix must never be "add +3 hours in React" -
 * these guards prove no such arithmetic exists anywhere in the dashboard.
 */
describe("No hardcoded timezone-offset correction anywhere in the dashboard", () => {
  it("never hardcodes an Israel/manual-offset timezone correction in any production dashboard file", () => {
    // Note: bare "IDT"/"IST" are deliberately excluded here - they collide
    // with ordinary English substrings (WIDTH, EXIST, CONSIST, ASSIST...)
    // and would false-positive; "Asia/Jerusalem" and the Date API offset
    // getters are unambiguous signals of a manual timezone correction.
    for (const path of PRODUCTION_FILES) {
      const source = readFileSync(path, "utf-8");
      for (const forbidden of ["Asia/Jerusalem", "getTimezoneOffset", "utcOffset"]) {
        expect(source).not.toContain(forbidden);
      }
    }
  });
});

/**
 * FireEvent "Opened" time guards: the operator-facing Opened timestamp must
 * come from the backend's own persisted `created_at` - never a frontend
 * `Date.now()`/`new Date()` invented at render/first-seen time.
 */
describe("No fabricated FireEvent creation/opened time in the frontend", () => {
  it("never calls Date.now() or constructs a no-argument Date() in any production dashboard file", () => {
    for (const path of PRODUCTION_FILES) {
      const source = stripComments(readFileSync(path, "utf-8"));
      expect(source).not.toContain("Date.now()");
      // A bare `new Date()` (no argument) would be "now" - every legitimate
      // use in this codebase parses a real ISO string argument instead.
      expect(source).not.toMatch(/new Date\(\s*\)/);
    }
  });

  it("ActiveFireEventCard's Opened row is driven by event.created_at, not a locally computed value", () => {
    const source = readFileSync("src/components/fire-events/ActiveFireEventCard.tsx", "utf-8");
    expect(source).toContain("event.created_at");
  });
});

/**
 * Weather Activity Feed wording guards (Part L): the LOW/MODERATE vs HIGH+
 * wording split must be driven purely by the persisted FireDangerLevel
 * enum value - never a frontend-invented numeric temperature/humidity/wind
 * threshold (that would be a second, undeclared danger methodology).
 */
describe("Weather Activity Feed wording guards", () => {
  it("never derives the weather wording from a numeric temperature/humidity/wind threshold", () => {
    const source = readFileSync("src/components/dashboard/OperationsActivityItem.tsx", "utf-8");
    // The wording lookup must be keyed by the FireDangerLevel string enum,
    // never by comparing temperature_c/relative_humidity_pct/wind_speed_kmh
    // against a literal number.
    expect(source).not.toMatch(/temperature_c\s*[<>]=?\s*\d/);
    expect(source).not.toMatch(/relative_humidity_pct\s*[<>]=?\s*\d/);
    expect(source).not.toMatch(/wind_speed_kmh\s*[<>]=?\s*\d/);
  });
});
