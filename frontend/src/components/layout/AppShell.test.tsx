import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "./AppShell";

function renderShell() {
  return render(
    <MemoryRouter initialEntries={["/events"]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/events" element={<p>content</p>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("AppShell", () => {
  it("shows the EcoGuard Israel brand", () => {
    renderShell();

    expect(screen.getByText("EcoGuard Israel")).toBeInTheDocument();
    expect(screen.getByText("Wildfire Operations")).toBeInTheDocument();
  });

  it("has an Active Wildfires navigation link", () => {
    renderShell();

    const link = screen.getByRole("link", { name: "Active Wildfires" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/events");
  });

  it("marks History as unavailable, not as a clickable link", () => {
    renderShell();

    expect(screen.queryByRole("link", { name: /history/i })).not.toBeInTheDocument();
    const history = screen.getByText("History", { exact: false });
    expect(history.closest(".nav-link")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("History", { exact: false }).closest("li")).toHaveTextContent(/coming soon/i);
  });

  it("marks Monitoring as unavailable, not as a clickable link", () => {
    renderShell();

    expect(screen.queryByRole("link", { name: /monitoring/i })).not.toBeInTheDocument();
    const monitoring = screen.getByText("Monitoring", { exact: false });
    expect(monitoring.closest(".nav-link")).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Monitoring", { exact: false }).closest("li")).toHaveTextContent(/coming soon/i);
  });

  it("uses a semantic nav landmark for navigation", () => {
    renderShell();

    expect(screen.getByRole("navigation", { name: "Main" })).toBeInTheDocument();
  });

  it("renders routed content inside main", () => {
    renderShell();

    expect(screen.getByRole("main")).toHaveTextContent("content");
  });
});
