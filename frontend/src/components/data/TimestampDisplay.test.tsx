import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TimestampDisplay } from "./TimestampDisplay";

describe("TimestampDisplay", () => {
  it("renders a formatted, human-readable timestamp", () => {
    render(<TimestampDisplay value="2026-09-17T14:28:00Z" />);

    // Locale-formatted text varies with the environment's ICU data, but the
    // date/year/hour components must all be present somewhere in the output.
    const time = document.querySelector("time");
    expect(time).not.toBeNull();
    expect(time?.textContent).toMatch(/17/);
    expect(time?.textContent).toMatch(/2026/);
  });

  it("uses a <time> element", () => {
    render(<TimestampDisplay value="2026-09-17T14:28:00Z" />);

    expect(document.querySelector("time")).not.toBeNull();
  });

  it("preserves the original ISO value in dateTime", () => {
    render(<TimestampDisplay value="2026-09-17T14:28:00Z" />);

    expect(document.querySelector("time")).toHaveAttribute("dateTime", "2026-09-17T14:28:00Z");
  });

  it("renders a safe fallback for a null value instead of throwing", () => {
    render(<TimestampDisplay value={null} />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
    expect(document.querySelector("time")).toBeNull();
  });

  it("renders a safe fallback for an invalid timestamp string", () => {
    render(<TimestampDisplay value="not-a-real-timestamp" />);

    expect(screen.getByText("Not available")).toBeInTheDocument();
    expect(document.querySelector("time")).toBeNull();
  });
});
