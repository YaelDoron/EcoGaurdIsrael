import { describe, expect, it } from "vitest";
import { createCircleIcon, createSquareIcon } from "./divIcon";

describe("createCircleIcon", () => {
  it("embeds the given color as the filled background by default", () => {
    const icon = createCircleIcon("rgb(1, 2, 3)");

    expect(icon.options.html).toContain("background:rgb(1, 2, 3)");
    expect(icon.options.html).toContain("border-radius:50%");
  });

  it("renders a hollow ring (transparent fill, colored border) when filled is false", () => {
    const icon = createCircleIcon("rgb(1, 2, 3)", { filled: false, borderStyle: "dashed" });

    expect(icon.options.html).toContain("background:transparent");
    expect(icon.options.html).toContain("border:3px dashed rgb(1, 2, 3)");
  });

  it("uses the given size for both the icon box and anchor", () => {
    const icon = createCircleIcon("red", { size: 30 });

    expect(icon.options.iconSize).toEqual([30, 30]);
    expect(icon.options.iconAnchor).toEqual([15, 15]);
  });
});

describe("createSquareIcon", () => {
  it("embeds the given color as the background with a square (rounded) shape", () => {
    const icon = createSquareIcon("blue");

    expect(icon.options.html).toContain("background:blue");
    expect(icon.options.html).toContain("border-radius:3px");
  });
});
