import { describe, expect, it } from "vitest";
import { createActiveFireIcon } from "./activeFireIcon";

const AMBER = "#b45309";
const RED = "#b91c1c";

describe("createActiveFireIcon", () => {
  it("renders SUSPECTED as a hollow marker (lighter center, colored ring) - not a solid dot", () => {
    const icon = createActiveFireIcon(AMBER, { confirmed: false });

    expect(icon.options.className).toContain("active-fire-icon--suspected");
    expect(icon.options.className).not.toContain("active-fire-icon--confirmed");
    const html = icon.options.html as string;
    expect(html).toContain(`border:3px solid ${AMBER}`);
    expect(html).toContain("background:#ffffff");
  });

  it("renders CONFIRMED as a solid filled marker", () => {
    const icon = createActiveFireIcon(RED, { confirmed: true });

    expect(icon.options.className).toContain("active-fire-icon--confirmed");
    expect(icon.options.className).not.toContain("active-fire-icon--suspected");
    const html = icon.options.html as string;
    expect(html).toContain(`background:${RED}`);
    expect(html).toContain("border:2px solid #ffffff");
  });

  it("gives SUSPECTED and CONFIRMED visually distinct markup even when hypothetically given the same color", () => {
    const suspected = createActiveFireIcon(RED, { confirmed: false });
    const confirmed = createActiveFireIcon(RED, { confirmed: true });

    expect(suspected.options.html).not.toEqual(confirmed.options.html);
  });

  it("adds a halo and the emphasized class only when explicitly requested (CONFIRMED + HIGH/CRITICAL)", () => {
    const plain = createActiveFireIcon(RED, { confirmed: true, emphasize: false });
    const emphasized = createActiveFireIcon(RED, { confirmed: true, emphasize: true });

    expect(plain.options.className).not.toContain("active-fire-icon--emphasized");
    expect(plain.options.html).not.toContain("active-fire-icon__halo");
    expect(emphasized.options.className).toContain("active-fire-icon--emphasized");
    expect(emphasized.options.html).toContain("active-fire-icon__halo");
  });

  it("gives the emphasized CONFIRMED marker distinct markup from a plain CONFIRMED marker", () => {
    const plain = createActiveFireIcon(RED, { confirmed: true, emphasize: false });
    const emphasized = createActiveFireIcon(RED, { confirmed: true, emphasize: true });

    expect(plain.options.html).not.toEqual(emphasized.options.html);
  });
});
