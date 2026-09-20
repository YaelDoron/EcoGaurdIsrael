import { describe, expect, it } from "vitest";
import { createCircleIcon, createFireIcon, createSquareIcon, createStationIcon } from "./divIcon";

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

  it("embeds the given opacity when provided", () => {
    const icon = createCircleIcon("red", { opacity: 0.3 });

    expect(icon.options.html).toContain("opacity:0.3");
  });

  it("omits opacity entirely when not given", () => {
    const icon = createCircleIcon("red");

    expect(icon.options.html).not.toContain("opacity:");
  });
});

describe("createSquareIcon", () => {
  it("embeds the given color as the background with a square (rounded) shape", () => {
    const icon = createSquareIcon("blue");

    expect(icon.options.html).toContain("background:blue");
    expect(icon.options.html).toContain("border-radius:3px");
  });
});

describe("createStationIcon", () => {
  it("renders the Bootstrap geo-alt-fill pin glyph in the given color", () => {
    const icon = createStationIcon("rgb(1, 2, 3)");

    expect(icon.options.html).toContain("<svg");
    expect(icon.options.html).toContain("color:rgb(1, 2, 3)");
    expect(icon.options.html).toContain("M8 16s6-5.686");
  });

  it("anchors the pin's tip, not its center, on the coordinate", () => {
    const icon = createStationIcon("red", { size: 30 });

    expect(icon.options.iconAnchor).toEqual([15, 30]);
  });

  it("embeds opacity only when given", () => {
    expect(createStationIcon("red", { opacity: 0.25 }).options.html).toContain("opacity:0.25");
    expect(createStationIcon("red").options.html).not.toContain("opacity:");
  });
});

describe("createFireIcon", () => {
  it("renders the Bootstrap fire glyph in the given color, centered on the coordinate", () => {
    const icon = createFireIcon("#ff4500", { size: 30 });

    expect(icon.options.html).toContain("data-fire-icon");
    expect(icon.options.html).toContain("color:#ff4500");
    expect(icon.options.html).toContain("M8 16c3.314 0 6-2");
    expect(icon.options.iconAnchor).toEqual([15, 15]);
  });

  it("embeds opacity only when given", () => {
    expect(createFireIcon("#ff4500", { opacity: 0.25 }).options.html).toContain("opacity:0.25");
    expect(createFireIcon("#ff4500").options.html).not.toContain("opacity:");
  });
});
