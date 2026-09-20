import { describe, expect, it } from "vitest";
import { buildStationPhoneLookup, getStationContact } from "./stationContact";

describe("buildStationPhoneLookup", () => {
  it("keeps a number that belongs to exactly one station", () => {
    const lookup = buildStationPhoneLookup([
      { Station: "A", PhoneNumber: "04-820-1111" },
      { Station: "B", PhoneNumber: "04-820-2222" },
    ]);

    expect(lookup.get("A")).toBe("04-820-1111");
    expect(lookup.get("B")).toBe("04-820-2222");
  });

  it("drops a number shared by several stations (a hotline, not a local number)", () => {
    const lookup = buildStationPhoneLookup([
      { Station: "A", PhoneNumber: "*4964" },
      { Station: "B", PhoneNumber: "*4964" },
      { Station: "C", PhoneNumber: "04-820-3333" },
    ]);

    expect(lookup.has("A")).toBe(false);
    expect(lookup.has("B")).toBe(false);
    expect(lookup.get("C")).toBe("04-820-3333");
  });

  it("ignores records with no name or an empty number", () => {
    const lookup = buildStationPhoneLookup([
      { Station: null, PhoneNumber: "04-820-4444" },
      { Station: "D", PhoneNumber: "" },
      { Station: "E", PhoneNumber: null },
      { Station: "F" },
    ]);

    expect(lookup.size).toBe(0);
  });
});

describe("getStationContact with the real registry", () => {
  it("returns no contact for a dispatched station, because the registry only holds the shared hotline", () => {
    for (const name of ["נשר", "עוספיא", "פארק הכרמל"]) {
      expect(getStationContact(name)).toBeNull();
    }
    expect(getStationContact(null)).toBeNull();
  });
});
