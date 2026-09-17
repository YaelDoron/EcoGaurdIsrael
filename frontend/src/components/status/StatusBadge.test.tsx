import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { FireEventStatus } from "../../types/fireEvent";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  const cases: Array<[FireEventStatus, string]> = [
    ["suspected", "Suspected"],
    ["confirmed", "Confirmed"],
    ["resolved", "Resolved"],
    ["dismissed", "Dismissed"],
  ];

  it.each(cases)("renders the readable label for status=%s", (status, label) => {
    render(<StatusBadge status={status} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("gives different statuses visually distinct classes, not color alone", () => {
    const { container: suspected } = render(<StatusBadge status="suspected" />);
    const { container: confirmed } = render(<StatusBadge status="confirmed" />);

    const suspectedBadge = suspected.querySelector(".badge");
    const confirmedBadge = confirmed.querySelector(".badge");
    expect(suspectedBadge?.className).not.toBe(confirmedBadge?.className);
    // Visible text differs regardless of styling - the real "not color alone" guarantee.
    expect(suspectedBadge).toHaveTextContent("Suspected");
    expect(confirmedBadge).toHaveTextContent("Confirmed");
  });
});
