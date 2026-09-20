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

  it("has an Operations Overview navigation link", () => {
    renderShell();

    const link = screen.getByRole("link", { name: "Operations Overview" });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/events");
  });

  it("does not render the old History/Monitoring placeholder items", () => {
    renderShell();

    for (const forbidden of [/history/i, /monitoring/i, /coming soon/i]) {
      expect(screen.queryByText(forbidden)).not.toBeInTheDocument();
    }
  });

  it("renders exactly one navigation link", () => {
    renderShell();

    expect(screen.getAllByRole("link")).toHaveLength(1);
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
