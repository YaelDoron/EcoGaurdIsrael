import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppRouter } from "./AppRouter";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRouter />
    </MemoryRouter>,
  );
}

describe("AppRouter", () => {
  it("redirects / to /events", () => {
    renderAt("/");

    expect(screen.getByRole("heading", { name: "Active Wildfires" })).toBeInTheDocument();
  });

  it("renders the Active Wildfires placeholder at /events", () => {
    renderAt("/events");

    expect(screen.getByRole("heading", { name: "Active Wildfires" })).toBeInTheDocument();
  });

  it("renders the Event Details placeholder at /events/:fireEventId", () => {
    renderAt("/events/123");

    expect(screen.getByRole("heading", { name: "Event Details" })).toBeInTheDocument();
    expect(screen.getByText(/not implemented yet/i)).toBeInTheDocument();
    expect(screen.getByText(/123/)).toBeInTheDocument();
  });

  it("renders the Response Plan placeholder at /events/:fireEventId/plan", () => {
    renderAt("/events/123/plan");

    expect(screen.getByRole("heading", { name: "Response Plan" })).toBeInTheDocument();
    expect(screen.getByText(/123/)).toBeInTheDocument();
  });

  it("renders the Response Plan placeholder at /plans/:planId", () => {
    renderAt("/plans/42");

    expect(screen.getByRole("heading", { name: "Response Plan" })).toBeInTheDocument();
    expect(screen.getByText(/42/)).toBeInTheDocument();
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
