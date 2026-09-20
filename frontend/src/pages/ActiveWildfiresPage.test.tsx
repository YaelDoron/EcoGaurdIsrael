import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/errors";
import type { ActiveFireEvent, ActiveFireEventsResponse } from "../types/activeFireEvents";
import { ActiveWildfiresPage } from "./ActiveWildfiresPage";

const { getActiveFireEventsMock } = vi.hoisted(() => ({
  getActiveFireEventsMock: vi.fn(),
}));

vi.mock("../api/activeFireEvents", () => ({
  getActiveFireEvents: getActiveFireEventsMock,
}));

function makeEvent(overrides: Partial<ActiveFireEvent> = {}): ActiveFireEvent {
  return {
    fire_event_id: 12,
    status: "confirmed",
    latitude: 32.731,
    longitude: 35.046,
    detection_confidence: 0.91,
    detected_at: "2026-09-17T13:20:00Z",
    updated_at: "2026-09-17T13:28:00Z",
    severity: null,
    ...overrides,
  };
}

function makeResponse(items: ActiveFireEvent[], asOf = "2026-09-17T14:00:00Z"): ActiveFireEventsResponse {
  return { as_of: asOf, items };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ActiveWildfiresPage />
    </MemoryRouter>,
  );
}

describe("ActiveWildfiresPage", () => {
  beforeEach(() => {
    getActiveFireEventsMock.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows LoadingState on initial render", () => {
    getActiveFireEventsMock.mockReturnValue(new Promise<never>(() => {}));

    renderPage();

    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it("shows correct summary metrics and both cards for one confirmed + one suspected event", async () => {
    getActiveFireEventsMock.mockResolvedValue(
      makeResponse([
        makeEvent({ fire_event_id: 1, status: "confirmed" }),
        makeEvent({ fire_event_id: 2, status: "suspected" }),
      ]),
    );

    renderPage();
    await screen.findByText("Event #1");

    const metrics = within(document.querySelector(".active-wildfires-page__metrics") as HTMLElement);
    expect(metrics.getByText("Active Events").closest("dl")).toHaveTextContent("2");
    expect(metrics.getByText("Confirmed").closest("dl")).toHaveTextContent("1");
    expect(metrics.getByText("Suspected").closest("dl")).toHaveTextContent("1");
    expect(screen.getByText("Event #1")).toBeInTheDocument();
    expect(screen.getByText("Event #2")).toBeInTheDocument();
  });

  it("preserves the backend item order in the rendered card order", async () => {
    getActiveFireEventsMock.mockResolvedValue(
      makeResponse([makeEvent({ fire_event_id: 12 }), makeEvent({ fire_event_id: 3 })]),
    );

    renderPage();
    await screen.findByText("Event #12");

    const headings = screen.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(["Event #12", "Event #3"]);
  });

  it("displays identity, status, confidence, coordinates, and timestamps on each card", async () => {
    getActiveFireEventsMock.mockResolvedValue(
      makeResponse([
        makeEvent({
          fire_event_id: 7,
          status: "confirmed",
          latitude: 32.731,
          longitude: 35.046,
          detection_confidence: 0.91,
          detected_at: "2026-09-17T13:20:00Z",
          updated_at: "2026-09-17T13:28:00Z",
        }),
      ]),
    );

    renderPage();
    const heading = await screen.findByText("Event #7");
    const card = heading.closest("article") as HTMLElement;
    const scoped = within(card);

    expect(scoped.getByText("Confirmed")).toBeInTheDocument();
    expect(scoped.getByText("91%")).toBeInTheDocument();
    expect(scoped.getByText("32.7310, 35.0460")).toBeInTheDocument();

    const times = card.querySelectorAll("time");
    expect(times).toHaveLength(2);
    expect(times[0]).toHaveAttribute("dateTime", "2026-09-17T13:20:00Z");
    expect(times[1]).toHaveAttribute("dateTime", "2026-09-17T13:28:00Z");
  });

  it("shows severity level and score when the latest assessment is valid", async () => {
    getActiveFireEventsMock.mockResolvedValue(
      makeResponse([
        makeEvent({
          severity: {
            assessment_id: 44,
            status: "valid",
            score: 81.4,
            level: "critical",
            assessed_at: "2026-09-17T13:27:00Z",
          },
        }),
      ]),
    );

    renderPage();
    await screen.findByText("Event #12");

    expect(screen.getByText("Critical")).toBeInTheDocument();
    expect(screen.getByText("Severity score: 81.4")).toBeInTheDocument();
  });

  it('shows "Not available" when there is no severity assessment at all', async () => {
    getActiveFireEventsMock.mockResolvedValue(makeResponse([makeEvent({ severity: null })]));

    renderPage();
    await screen.findByText("Event #12");

    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("does not fabricate a severity level when the assessment is insufficient_data", async () => {
    getActiveFireEventsMock.mockResolvedValue(
      makeResponse([
        makeEvent({
          severity: {
            assessment_id: 45,
            status: "insufficient_data",
            score: null,
            level: null,
            assessed_at: "2026-09-17T13:27:00Z",
          },
        }),
      ]),
    );

    renderPage();
    await screen.findByText("Event #12");

    expect(screen.getByText("Not available")).toBeInTheDocument();
    expect(screen.getByText("Insufficient data for a severity assessment.")).toBeInTheDocument();
    for (const label of ["Low", "Moderate", "High", "Critical"]) {
      expect(screen.queryByText(label)).not.toBeInTheDocument();
    }
  });

  it("shows EmptyState with zeroed metrics when there are no active events", async () => {
    getActiveFireEventsMock.mockResolvedValue(makeResponse([]));

    renderPage();
    await screen.findByText("No active wildfire events");

    expect(
      screen.getByText("There are currently no suspected or confirmed wildfire events."),
    ).toBeInTheDocument();
    const metrics = within(document.querySelector(".active-wildfires-page__metrics") as HTMLElement);
    expect(metrics.getByText("Active Events").closest("dl")).toHaveTextContent("0");
    expect(metrics.getByText("Confirmed").closest("dl")).toHaveTextContent("0");
    expect(metrics.getByText("Suspected").closest("dl")).toHaveTextContent("0");
  });

  it("shows ErrorState with Retry when the initial load fails", async () => {
    getActiveFireEventsMock.mockRejectedValue(new ApiError("boom", 500));

    renderPage();

    expect(await screen.findByRole("heading", { name: "Unable to load active wildfire events." })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("recovers after Retry when the second attempt succeeds", async () => {
    const user = userEvent.setup();
    getActiveFireEventsMock
      .mockRejectedValueOnce(new ApiError("boom", 500))
      .mockResolvedValueOnce(makeResponse([makeEvent({ fire_event_id: 9 })]));

    renderPage();
    await screen.findByRole("button", { name: "Retry" });
    await user.click(screen.getByRole("button", { name: "Retry" }));

    await screen.findByText("Event #9");
    expect(screen.queryByRole("heading", { name: "Unable to load active wildfire events." })).not.toBeInTheDocument();
  });

  it("updates the dashboard after a successful Refresh", async () => {
    const user = userEvent.setup();
    getActiveFireEventsMock
      .mockResolvedValueOnce(makeResponse([makeEvent({ fire_event_id: 1 })]))
      .mockResolvedValueOnce(
        makeResponse([makeEvent({ fire_event_id: 1 }), makeEvent({ fire_event_id: 2, status: "suspected" })]),
      );

    renderPage();
    await screen.findByText("Event #1");
    expect(screen.queryByText("Event #2")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Refresh" }));

    await screen.findByText("Event #2");
  });

  it("keeps existing valid data visible when a refresh fails, without wiping the page", async () => {
    const user = userEvent.setup();
    getActiveFireEventsMock
      .mockResolvedValueOnce(makeResponse([makeEvent({ fire_event_id: 1 })]))
      .mockRejectedValueOnce(new ApiError("boom", 500));

    renderPage();
    await screen.findByText("Event #1");

    await user.click(screen.getByRole("button", { name: "Refresh" }));

    await screen.findByRole("alert");
    expect(screen.getByText("Event #1")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Unable to load active wildfire events." })).not.toBeInTheDocument();
  });

  it("links View Event to /events/{fire_event_id}", async () => {
    getActiveFireEventsMock.mockResolvedValue(makeResponse([makeEvent({ fire_event_id: 55 })]));

    renderPage();
    const link = await screen.findByRole("link", { name: "View Event" });

    expect(link).toHaveAttribute("href", "/events/55");
  });

  it("displays the API as_of timestamp", async () => {
    getActiveFireEventsMock.mockResolvedValue(makeResponse([], "2026-09-17T14:00:00Z"));

    renderPage();
    await screen.findByText("No active wildfire events");

    expect(screen.getByText(/data as of/i)).toBeInTheDocument();
    const time = document.querySelector("time");
    expect(time).toHaveAttribute("dateTime", "2026-09-17T14:00:00Z");
  });

  it("offers a prominent Global Response Plan link to /response-plan with no focus filter", async () => {
    getActiveFireEventsMock.mockResolvedValue({ as_of: "2026-09-17T14:00:00Z", items: [] });

    renderPage();

    const link = await screen.findByRole("link", { name: "Global Response Plan" });
    expect(link).toHaveAttribute("href", "/response-plan");
  });
});
