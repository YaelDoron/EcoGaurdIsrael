import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FireEventStatus, FireSeverityAssessmentStatus, FireSeverityLevel } from "../types/fireEvent";
import type { FireDangerAssessmentStatus, FireDangerLevel } from "../types/fireDanger";
import type { OperationsActivityDetailResponse, OperationsActivityType } from "../types/operationsActivity";
import type { OperationsOverviewResponse } from "../types/operationsOverview";
import { getOperationsActivityDetail, getOperationsOverview } from "./operations";
import { ApiError } from "./errors";

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

function emptyOverview(): OperationsOverviewResponse {
  return {
    generated_at: "2026-09-20T12:00:00Z",
    simulation: { enabled: false, run: null },
    fire_danger_areas: [],
    active_fires: [],
    activity_feed: { items: [], limit: 30 },
  };
}

describe("getOperationsOverview", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("builds the correct URL with no query string when activityLimit is omitted", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(emptyOverview()));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getOperationsOverview();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/operations/overview",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("forwards a custom activityLimit as ?activity_limit=", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(emptyOverview()));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getOperationsOverview({ activityLimit: 10 });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/operations/overview?activity_limit=10",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("propagates an AbortSignal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(emptyOverview()));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await getOperationsOverview({ signal: controller.signal });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("returns the response without losing null fields (severity=null, assessment=null, run=null)", async () => {
    const body: OperationsOverviewResponse = {
      generated_at: "2026-09-20T12:00:00Z",
      simulation: { enabled: true, run: null },
      fire_danger_areas: [
        {
          area_id: "area-carmel",
          area_name: "Carmel",
          center: { latitude: 32.731, longitude: 35.046 },
          radius_km: 5,
          assessment: null,
        },
      ],
      active_fires: [
        {
          fire_event_id: 1,
          status: "suspected",
          latitude: 32.7,
          longitude: 35.0,
          detection_confidence: 0.6,
          detected_at: "2026-09-20T11:00:00Z",
          updated_at: "2026-09-20T11:05:00Z",
          created_at: "2026-09-20T11:00:30Z",
          severity: null,
          location_name: null,
          ml_summary: null,
        },
      ],
      activity_feed: { items: [], limit: 30 },
    };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getOperationsOverview();

    expect(result).toEqual(body);
    expect(result.fire_danger_areas[0].assessment).toBeNull();
    expect(result.active_fires[0].severity).toBeNull();
    expect(result.simulation.run).toBeNull();
  });

  it("supports all five Fire Danger levels and both statuses round-tripped unchanged", async () => {
    const levels: FireDangerLevel[] = ["low", "moderate", "high", "very_high", "extreme"];
    const statuses: FireDangerAssessmentStatus[] = ["valid", "insufficient_data"];
    const body = emptyOverview();
    body.fire_danger_areas = levels.map((level, index) => ({
      area_id: `area-${index}`,
      area_name: `Area ${index}`,
      center: { latitude: 32.0, longitude: 35.0 },
      radius_km: 5,
      assessment: {
        assessment_id: index + 1,
        status: statuses[index % statuses.length],
        score: statuses[index % statuses.length] === "valid" ? 10 : null,
        level: statuses[index % statuses.length] === "valid" ? level : null,
        assessed_at: "2026-09-20T11:00:00Z",
        age_seconds: 60,
        methodology: "FOSBERG_FFWI",
        methodology_version: "1.0",
      },
    }));
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getOperationsOverview();

    expect(result.fire_danger_areas).toEqual(body.fire_danger_areas);
  });

  it("supports all four FireEvent statuses returned by the API", async () => {
    const statuses: FireEventStatus[] = ["suspected", "confirmed", "resolved", "dismissed"];
    const body = emptyOverview();
    body.active_fires = statuses.map((status, index) => ({
      fire_event_id: index + 1,
      status,
      latitude: 32.0,
      longitude: 35.0,
      detection_confidence: 0.5,
      detected_at: "2026-09-20T11:00:00Z",
      updated_at: "2026-09-20T11:05:00Z",
      created_at: "2026-09-20T11:00:30Z",
      severity: null,
      location_name: null,
      ml_summary: null,
    }));
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getOperationsOverview();

    expect(result.active_fires.map((item) => item.status)).toEqual(statuses);
  });

  it("supports all four Severity levels and all three Severity statuses returned by the API", async () => {
    const levels: FireSeverityLevel[] = ["low", "moderate", "high", "critical"];
    const statuses: FireSeverityAssessmentStatus[] = ["valid", "insufficient_data", "inactive_event"];
    const body = emptyOverview();
    body.active_fires = levels.map((level, index) => ({
      fire_event_id: index + 1,
      status: "confirmed",
      latitude: 32.0,
      longitude: 35.0,
      detection_confidence: 0.5,
      detected_at: "2026-09-20T11:00:00Z",
      updated_at: "2026-09-20T11:05:00Z",
      created_at: "2026-09-20T11:00:30Z",
      severity: {
        assessment_id: index + 1,
        status: statuses[index % statuses.length],
        score: statuses[index % statuses.length] === "valid" ? 10 : null,
        level: statuses[index % statuses.length] === "valid" ? level : null,
        assessed_at: "2026-09-20T11:00:00Z",
      },
      location_name: null,
      ml_summary: null,
    }));
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getOperationsOverview();

    expect(result.active_fires.map((item) => item.severity)).toEqual(body.active_fires.map((item) => item.severity));
  });

  it("supports all seven activity types in the activity feed", async () => {
    const types: OperationsActivityType[] = [
      "fire_danger",
      "satellite_hotspot",
      "news_report",
      "fire_event",
      "fire_severity",
      "global_planning_run",
      "weather_conditions",
    ];
    const body = emptyOverview();
    body.activity_feed = {
      limit: 30,
      items: types.map((activity_type, index) => ({
        activity_id: `${activity_type}:${index + 1}`,
        activity_type,
        entity_id: index + 1,
        occurred_at: "2026-09-20T11:00:00Z",
        available_at: "2026-09-20T11:00:00Z",
        title: `Item ${index}`,
        location: null,
        preview: {} as never,
      })),
    };
    globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

    const result = await getOperationsOverview();

    expect(result.activity_feed.items.map((item) => item.activity_type)).toEqual(types);
  });
});

describe("getOperationsActivityDetail", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.stubEnv("VITE_API_BASE_URL", "http://api.example.test");
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.unstubAllEnvs();
  });

  it("builds the correct URL for each activity type", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        activity_type: "fire_event",
        entity_id: 42,
        occurred_at: "2026-09-20T11:00:00Z",
        title: "Fire Event #42",
        location: null,
        details: {},
      }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    await getOperationsActivityDetail("fire_event", 42);

    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.example.test/api/v1/operations/activity/fire_event/42",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("propagates an AbortSignal", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        activity_type: "satellite_hotspot",
        entity_id: 7,
        occurred_at: "2026-09-20T11:00:00Z",
        title: "Satellite Hotspot",
        location: null,
        details: {},
      }),
    );
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const controller = new AbortController();

    await getOperationsActivityDetail("satellite_hotspot", 7, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it("returns the typed discriminated response for each activity type", async () => {
    const cases: Array<[OperationsActivityType, OperationsActivityDetailResponse]> = [
      [
        "fire_danger",
        {
          activity_type: "fire_danger",
          entity_id: 1,
          occurred_at: "2026-09-20T11:00:00Z",
          title: "Fire Danger Assessment — Carmel",
          location: { latitude: 32.7, longitude: 35.0 },
          details: {
            assessment_id: 1,
            area_id: "area-carmel",
            area_name: "Carmel",
            center: { latitude: 32.7, longitude: 35.0 },
            radius_km: 5,
            status: "valid",
            score: 42.5,
            level: "very_high",
            assessed_at: "2026-09-20T11:00:00Z",
            age_seconds: 60,
            methodology: "FOSBERG_FFWI",
            methodology_version: "1.0",
            weather_inputs: [],
          },
        },
      ],
      [
        "global_planning_run",
        {
          activity_type: "global_planning_run",
          entity_id: 5,
          occurred_at: "2026-09-20T11:00:00Z",
          title: "Global Response Plan",
          location: null,
          details: {
            global_planning_run_id: 5,
            status: "completed",
            trigger: "weather_event",
            started_at: "2026-09-20T11:00:00Z",
            completed_at: "2026-09-20T11:05:00Z",
            methodology: "legacy_per_event_orchestration",
            methodology_version: "1.0",
            fire_event_ids: [1, 2],
            response_plan_ids: [10, 11],
            coverage_score: 100,
            average_eta_seconds: 300,
            shortage_total_required: 4,
            shortage_total_desired: 6,
            shortage_total_assigned: 6,
            shortage_unmet_required: 0,
            shortage_unmet_desired: 0,
            ga_population_size: 24,
            ga_generation_count: 40,
            ga_mutation_rate: 0.08,
            ga_crossover_rate: 0.75,
            members: [],
          },
        },
      ],
      [
        "weather_conditions",
        {
          activity_type: "weather_conditions",
          entity_id: 1,
          occurred_at: "2026-09-20T11:00:00Z",
          title: "Weather Conditions — Galilee Demo Area",
          location: { latitude: 32.965, longitude: 35.381 },
          details: {
            fire_danger_assessment_id: 1,
            area_name: "Galilee Demo Area",
            fire_danger_level: "very_high",
            assessed_at: "2026-09-20T11:00:00Z",
            readings: [
              {
                station_id: 3,
                station_name: "Galilee Station",
                observation_id: 10,
                observed_at: "2026-09-20T10:55:00Z",
                temperature: 34,
                relative_humidity: 19,
                wind_speed: 28,
                wind_gust: null,
              },
            ],
          },
        },
      ],
    ];

    for (const [activityType, body] of cases) {
      globalThis.fetch = vi.fn().mockResolvedValue(jsonResponse(body)) as unknown as typeof fetch;

      const result = await getOperationsActivityDetail(activityType, body.entity_id);

      expect(result).toEqual(body);
    }
  });

  it("propagates an ApiError with status 404 for an unknown entity", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      jsonResponse(
        { error: { code: "OPERATIONS_ACTIVITY_NOT_FOUND", message: "No fire_event activity exists with id=999." } },
        { status: 404 },
      ),
    ) as unknown as typeof fetch;

    await expect(getOperationsActivityDetail("fire_event", 999)).rejects.toMatchObject({
      status: 404,
      code: "OPERATIONS_ACTIVITY_NOT_FOUND",
    });
    await expect(getOperationsActivityDetail("fire_event", 999)).rejects.toBeInstanceOf(ApiError);
  });
});
