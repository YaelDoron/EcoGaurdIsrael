import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ResponsePlanPage } from "./ResponsePlanPage";

const { getCurrentResponsePlanMock } = vi.hoisted(() => ({ getCurrentResponsePlanMock: vi.fn() }));

vi.mock("../api/responsePlans", () => ({ getCurrentResponsePlan: getCurrentResponsePlanMock, getResponsePlanById: vi.fn() }));
vi.mock("../api/eventDetails", () => ({ getEventDetails: vi.fn().mockReturnValue(new Promise(() => {})) }));
vi.mock("../api/reverseGeocode", () => ({ reverseGeocode: vi.fn().mockResolvedValue(null) }));
vi.mock("react-leaflet", async () => import("../test/reactLeafletStub"));

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/events/:fireEventId/plan" element={<ResponsePlanPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ResponsePlanPage plan lifecycle", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("a CONFIRMED event whose plan is not persisted yet shows Generating, not No response plan", async () => {
    getCurrentResponsePlanMock.mockResolvedValue({ plan: null, plan_status: "generating" });

    renderAt("/events/5/plan");

    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();
    expect(screen.queryByText("No response plan available")).not.toBeInTheDocument();
  });

  it("keeps checking quietly while the plan is being generated", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getCurrentResponsePlanMock.mockResolvedValue({ plan: null, plan_status: "generating" });

    renderAt("/events/5/plan");
    expect(await screen.findByText("Generating response plan...")).toBeInTheDocument();
    expect(getCurrentResponsePlanMock).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(5000);

    await vi.waitFor(() => expect(getCurrentResponsePlanMock).toHaveBeenCalledTimes(2));
    expect(screen.getByText("Generating response plan...")).toBeInTheDocument(); // no loading flash
  });

  it("a SUSPECTED (monitoring-only) event never implies a plan is being generated", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    getCurrentResponsePlanMock.mockResolvedValue({ plan: null, plan_status: "not_applicable" });

    renderAt("/events/6/plan");

    expect(await screen.findByText("No response plan available")).toBeInTheDocument();
    expect(screen.queryByText(/Generating/)).not.toBeInTheDocument();
    await vi.advanceTimersByTimeAsync(15000);
    expect(getCurrentResponsePlanMock).toHaveBeenCalledTimes(1); // no polling for a plan that is not coming
  });
});
