import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppRouter } from "./AppRouter";

const { getCurrentResponsePlanMock, getResponsePlanByIdMock } = vi.hoisted(() => ({
  getCurrentResponsePlanMock: vi.fn(),
  getResponsePlanByIdMock: vi.fn(),
}));

vi.mock("../api/responsePlans", () => ({
  getCurrentResponsePlan: getCurrentResponsePlanMock,
  getResponsePlanById: getResponsePlanByIdMock,
}));

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRouter />
    </MemoryRouter>,
  );
}

describe("AppRouter", () => {
  beforeEach(() => {
    getCurrentResponsePlanMock.mockReset();
    getResponsePlanByIdMock.mockReset();
    // Both routes render their PageHeader synchronously, before the plan
    // fetch resolves - these routing tests only need that heading, so an
    // always-pending promise keeps each test from depending on fetch timing.
    getCurrentResponsePlanMock.mockReturnValue(new Promise<never>(() => {}));
    getResponsePlanByIdMock.mockReturnValue(new Promise<never>(() => {}));
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("redirects / to /events", () => {
    renderAt("/");

    expect(screen.getByRole("heading", { name: "Operations Overview" })).toBeInTheDocument();
  });

  it("renders the Operations Overview dashboard at /events", () => {
    renderAt("/events");

    expect(screen.getByRole("heading", { name: "Operations Overview" })).toBeInTheDocument();
  });

  it("renders the Event Details page at /events/:fireEventId", () => {
    renderAt("/events/123");

    expect(screen.getByRole("heading", { name: "Loading Event…" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(/loading/i);
  });

  it("renders ResponsePlanPage at /events/:fireEventId/plan", () => {
    renderAt("/events/123/plan");

    expect(screen.getByRole("heading", { name: "Loading Response Plan…" })).toBeInTheDocument();
    expect(getCurrentResponsePlanMock).toHaveBeenCalledWith(123, expect.any(AbortSignal));
  });

  it("renders ResponsePlanPage at /plans/:planId", () => {
    renderAt("/plans/42");

    expect(screen.getByRole("heading", { name: "Loading Response Plan…" })).toBeInTheDocument();
    expect(getResponsePlanByIdMock).toHaveBeenCalledWith(42, expect.any(AbortSignal));
  });

  it("renders Not Found for an unknown route", () => {
    renderAt("/this-route-does-not-exist");

    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /return to active wildfires/i })).toHaveAttribute(
      "href",
      "/events",
    );
  });
});
