import { readFileSync } from "node:fs";
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DisableScrollWheelZoom } from "./DisableScrollWheelZoom";
import { MapInstanceContext, createFakeMap } from "../../test/reactLeafletStub";

vi.mock("react-leaflet", async () => import("../../test/reactLeafletStub"));

function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

describe("DisableScrollWheelZoom", () => {
  it("calls map.scrollWheelZoom.disable() on the live Leaflet map instance", () => {
    const disable = vi.fn();
    const fakeMap = createFakeMap({ scrollWheelZoom: { enable: vi.fn(), disable } });

    render(
      <MapInstanceContext.Provider value={fakeMap}>
        <DisableScrollWheelZoom />
      </MapInstanceContext.Provider>,
    );

    expect(disable).toHaveBeenCalledTimes(1);
  });

  it("never calls enable() - it only ever disables", () => {
    const enable = vi.fn();
    const fakeMap = createFakeMap({ scrollWheelZoom: { enable, disable: vi.fn() } });

    render(
      <MapInstanceContext.Provider value={fakeMap}>
        <DisableScrollWheelZoom />
      </MapInstanceContext.Provider>,
    );

    expect(enable).not.toHaveBeenCalled();
  });

  it("renders no visible DOM - it is a pure side-effect controller", () => {
    const fakeMap = createFakeMap();

    const { container } = render(
      <MapInstanceContext.Provider value={fakeMap}>
        <DisableScrollWheelZoom />
      </MapInstanceContext.Provider>,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("does not call preventDefault-style DOM wiring - it only uses Leaflet's own Handler API", () => {
    // Source-level guard: this component must never touch a raw wheel
    // event listener or CSS pointer-events - only Leaflet's supported
    // scrollWheelZoom.disable() API (see the component's own docstring).
    const source = stripComments(readFileSync("src/components/map/DisableScrollWheelZoom.tsx", "utf-8"));

    expect(source).not.toMatch(/preventDefault/);
    expect(source).not.toMatch(/pointer-events/);
    expect(source).not.toMatch(/addEventListener\(\s*["']wheel["']/);
  });
});
