import { readFileSync } from "node:fs";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ActivityTimestamp } from "./ActivityTimestamp";

describe("ActivityTimestamp", () => {
  it("renders HH:MM:SS so items within the same minute are still distinguishable", () => {
    render(<ActivityTimestamp value="2026-09-20T12:00:07Z" />);

    const time = screen.getByText((_, element) => element?.tagName.toLowerCase() === "time");
    expect(time).toHaveTextContent(/^\d{2}:\d{2}:\d{2}$/);
    expect(time).toHaveAttribute("dateTime", "2026-09-20T12:00:07Z");
  });

  it("shows two items in the same minute with different displayed seconds", () => {
    const { unmount } = render(<ActivityTimestamp value="2026-09-20T12:00:05Z" />);
    const first = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time").textContent;
    unmount();

    render(<ActivityTimestamp value="2026-09-20T12:00:47Z" />);
    const second = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time").textContent;

    expect(first).not.toBe(second);
  });

  it("shows a safe fallback for an invalid timestamp, never crashing", () => {
    render(<ActivityTimestamp value="not-a-real-timestamp" />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
  });

  it("renders a UTC 'Z' timestamp identically to its equivalent '+00:00' offset form (no hardcoded offset math)", () => {
    const { unmount } = render(<ActivityTimestamp value="2026-09-20T10:12:00Z" />);
    const zForm = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time").textContent;
    unmount();

    render(<ActivityTimestamp value="2026-09-20T10:12:00+00:00" />);
    const offsetForm = screen.getByText((_, el) => el?.tagName.toLowerCase() === "time").textContent;

    expect(zForm).toBe(offsetForm);
  });

  it("contains no hardcoded Israel/+3/IDT/IST timezone offset arithmetic", () => {
    const source = readFileSync("src/components/dashboard/ActivityTimestamp.tsx", "utf-8");
    for (const forbidden of ["+3", "10800", "Asia/Jerusalem", "IDT", "IST", "getTimezoneOffset"]) {
      expect(source).not.toContain(forbidden);
    }
  });
});
