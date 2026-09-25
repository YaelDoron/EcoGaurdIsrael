import { act, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EventDetailsResult } from "../types/eventDetails";
import type { LatLngPoint } from "../components/map/mapTypes";
import { EventDetailsPage } from "./EventDetailsPage";

const { getEventDetailsMock, reverseGeocodeMock, fitBoundsPointsSeen } = vi.hoisted(() => ({
  getEventDetailsMock: vi.fn(),
  reverseGeocodeMock: vi.fn(),
  fitBoundsPointsSeen: [] as unknown[],
}));

vi.mock("../api/eventDetails", () => ({ getEventDetails: getEventDetailsMock }));
vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: reverseGeocodeMock }));
vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));
// FitBoundsToPoints re-frames the map whenever its `points` identity changes;
// recording what it receives is the direct way to observe that.
vi.mock("../components/map/FitBoundsToPoints", () => ({
  FitBoundsToPoints: ({ points }: { points: LatLngPoint[] }) => {
    fitBoundsPointsSeen.push(points);
    return null;
  },
}));

function makeResult(overrides: Partial<EventDetailsResult> = {}): EventDetailsResult {
  return {
    as_of: "2026-09-17T14:00:00Z",
    fire_event: {
      fire_event_id: 12,
      status: "suspected",
      latitude: 32.731,
      longitude: 35.046,
      detection_confidence: 0.91,
      detected_at: "2026-09-17T13:20:00Z",
      updated_at: "2026-09-17T13:28:00Z",
      methodology: "detector",
      methodology_version: "1.0",
    },
    severity: null,
    ml_assessment: null,
    danger: null,
    detection_evidence: { satellite: [], news: [] },
    spread_predictions: [],
    targets: [],
    stations: [],
    resources: [],
    station_summaries: [],
    current_response_plan: null,
    ...overrides,
  };
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("EventDetailsPage polling", () => {
  beforeEach(() => {
    getEventDetailsMock.mockReset();
    reverseGeocodeMock.mockReset();
    reverseGeocodeMock.mockResolvedValue(null);
    fitBoundsPointsSeen.length = 0;
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  function renderPage() {
    return render(
      <MemoryRouter initialEntries={["/events/12"]}>
        <Routes>
          <Route path="/events/:fireEventId" element={<EventDetailsPage />} />
        </Routes>
      </MemoryRouter>,
    );
  }

  it("shows the updated status after a poll without navigating away", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeResult())
      .mockResolvedValue(
        makeResult({ as_of: "2026-09-17T14:00:05Z", fire_event: { ...makeResult().fire_event, status: "confirmed" } }),
      );

    renderPage();
    await advance(0);
    expect(screen.getByText(/monitoring/i)).toBeInTheDocument();

    await advance(5000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/monitoring/i)).not.toBeInTheDocument();
  });

  it("does not re-fit the map on a poll that returns the same geometry", async () => {
    getEventDetailsMock.mockImplementation(async () => makeResult({ as_of: new Date().toISOString() }));

    renderPage();
    await advance(0);
    const firstPoints = fitBoundsPointsSeen.at(-1);
    expect(firstPoints).toBeDefined();

    await advance(5000);
    await advance(5000);

    expect(getEventDetailsMock).toHaveBeenCalledTimes(3);
    expect(new Set(fitBoundsPointsSeen).size).toBe(1);
    expect(fitBoundsPointsSeen.at(-1)).toBe(firstPoints);
  });

  it("re-fits the map when a poll returns different geometry", async () => {
    getEventDetailsMock
      .mockResolvedValueOnce(makeResult())
      .mockResolvedValue(makeResult({ fire_event: { ...makeResult().fire_event, latitude: 33.0 } }));

    renderPage();
    await advance(0);
    const firstPoints = fitBoundsPointsSeen.at(-1);

    await advance(5000);

    expect(fitBoundsPointsSeen.at(-1)).not.toBe(firstPoints);
  });
});
