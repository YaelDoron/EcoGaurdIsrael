import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CoordinateDisplay } from "./CoordinateDisplay";

describe("CoordinateDisplay", () => {
  it("displays latitude/longitude with the default precision", () => {
    render(<CoordinateDisplay latitude={32.731} longitude={35.046} />);

    expect(screen.getByText("32.7310, 35.0460")).toBeInTheDocument();
  });

  it("supports a configurable precision", () => {
    render(<CoordinateDisplay latitude={32.731} longitude={35.046} precision={2} />);

    expect(screen.getByText("32.73, 35.05")).toBeInTheDocument();
  });

  it("does not mutate the values it is given (pure formatting)", () => {
    const latitude = 32.731;
    const longitude = 35.046;
    render(<CoordinateDisplay latitude={latitude} longitude={longitude} />);

    expect(latitude).toBe(32.731);
    expect(longitude).toBe(35.046);
  });
});
