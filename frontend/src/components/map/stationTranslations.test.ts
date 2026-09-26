import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  translateIfUntranslated,
  translateStationAddress,
  translateStationName,
  translateStationType,
  untranslatedWords,
} from "./stationTranslations";

const HEBREW = /[֐-׿]/;

describe("stationTranslations", () => {
  it("translateIfUntranslated leaves already-English text completely unchanged", () => {
    // The normal case, always: backend LLM translation already succeeded -
    // this must never re-process or second-guess it.
    expect(translateIfUntranslated("Jerusalem Hills")).toBe("Jerusalem Hills");
    expect(translateIfUntranslated(null)).toBeNull();
    expect(translateIfUntranslated("")).toBe("");
  });

  it("translateIfUntranslated keeps readable Hebrew free text instead of romanizing it (the failed-translation safety net)", () => {
    // Mirrors the real "TRANSLATION FALLBACK TRIGGERED" case: backend
    // ingestion translation failed and persisted the original Hebrew.
    // Letter-by-letter romanization ("Dyvchym Rashvnym Al Mvkdy Ash...")
    // is unreadable, so free text the dictionary cannot fully cover stays Hebrew.
    const headline = "דיווחים ראשוניים על מוקדי אש באזור הרי יהודה";
    expect(translateIfUntranslated(headline)).toBe(headline);
    expect(translateIfUntranslated("דיווח ראשוני על עשן ולהבות באזור יער ירושלים")).toBe(
      "דיווח ראשוני על עשן ולהבות באזור יער ירושלים",
    );
    expect(translateIfUntranslated(headline)).not.toMatch(/Dyvchym|Rashvnym|Mvkdy/);
  });

  it("translateIfUntranslated gives every simulator demo area its English name, as a whole-string match only", () => {
    expect(translateIfUntranslated("הכרמל")).toBe("Carmel");
    expect(translateIfUntranslated("יער ירושלים")).toBe("Jerusalem Forest");
    expect(translateIfUntranslated("הגליל")).toBe("Galilee");
    expect(translateIfUntranslated("רמת הגולן")).toBe("Golan Heights");
    expect(translateIfUntranslated("הרי יהודה")).toBe("Judean Hills");
    // Station vocabulary containing these words is unaffected.
    expect(translateStationName("נוף הגליל")).toBe("Nof HaGalil Station");
    // Free-text headlines are never partially rewritten.
    expect(translateIfUntranslated("דיווח ראשוני על עשן ולהבות באזור הגליל")).toBe("דיווח ראשוני על עשן ולהבות באזור הגליל");
  });

  it("translateIfUntranslated still translates a fully dictionary-covered place name", () => {
    expect(translateIfUntranslated("יער ירושלים")).toBe("Jerusalem Forest");
    expect(translateIfUntranslated("רמת הגולן")).toBe("Golan Heights");
  });

  it("translateIfUntranslated is idempotent - running it twice does not change the result further", () => {
    const once = translateIfUntranslated("הרי ירושלים");
    const twice = translateIfUntranslated(once);
    expect(twice).toBe(once);
  });

  it("translates names, addresses and types", () => {
    expect(translateStationName("מפרץ חיפה")).toBe("Haifa Bay Station");
    expect(translateStationAddress("החרש 1 חיפה")).toBe("Haifa, HaCharash 1");
    expect(translateStationType("אזורית")).toBe("Regional");
  });

  it("uses clean City-first overrides for intersections and industrial zones", () => {
    expect(translateStationAddress("סולטן באשא אל אטראש פינת מופק דיאב שפרעם")).toBe(
      "Shefa-Amr, Sultan Pasha Al Atrash / Mufaq Diab",
    );
    expect(translateStationAddress("תכלת 1 פארק תעשיה משגב-תרדיון")).toBe("Misgav (Tardiyon), Tekhelet 1");
    // whitespace differences in the source do not defeat an override
    expect(translateStationAddress("  שלומציון  	 בקעת הירדן ")).toBe("Bikat HaYarden, Shlomtzion");
  });

  it("flips unknown comma-separated addresses city-first without stray commas or spaces", () => {
    const out = translateStationAddress("הזית , , ירושלים");
    expect(out).toBe("Jerusalem, HaZayit");
    expect(out).not.toMatch(/,s*,|s{2,}|s,/);
  });

  it("never leaves Hebrew script behind, even for unknown words (romanized fallback)", () => {
    expect(translateStationName("זזזזז")).not.toMatch(HEBREW);
    expect(untranslatedWords("זזזזז")).toEqual(["זזזזז"]);
  });

  it("covers every station in the seed data", () => {
    const path = resolve(__dirname, "../../../../backend/src/database/data/firefighting_stations.json");
    const records: { Station: string | null; Address: string | null }[] = JSON.parse(readFileSync(path, "utf-8")).result.records;
    const missing = records.flatMap((r) => [...untranslatedWords(r.Station ?? ""), ...untranslatedWords(r.Address ?? "")]);
    expect([...new Set(missing)]).toEqual([]);
    for (const r of records) {
      expect(translateStationAddress(r.Address ?? "")).not.toMatch(/,s*,|s{2,}|^,|,$/);
      expect(translateStationName(r.Station ?? "x")).not.toMatch(HEBREW);
      expect(translateStationAddress(r.Address ?? "")).not.toMatch(HEBREW);
    }
  });

  it("covers every District/Regional_Station region name in the seed data too, not just Station/Address", () => {
    // Regression guard: these two fields name real Israeli regions but
    // aren't exercised by any Station/Address in the seed data, which is
    // exactly how the "רמת הגולן" gap went unnoticed - District and
    // Regional_Station are unrelated real-world region names, not derived
    // from any Station/Address string.
    const path = resolve(__dirname, "../../../../backend/src/database/data/firefighting_stations.json");
    const records: { District: string | null; Regional_Station: string | null }[] = JSON.parse(
      readFileSync(path, "utf-8"),
    ).result.records;
    const missing = records.flatMap((r) => [
      ...untranslatedWords(r.District ?? ""),
      ...untranslatedWords(r.Regional_Station ?? ""),
    ]);
    // 'מב"ת' (a control-center abbreviation, not a place name) is the one
    // accepted gap - everything else must be a real dictionary hit.
    const ABBREVIATION = 'מב"ת';
    expect([...new Set(missing)].filter((word) => word !== ABBREVIATION)).toEqual([]);
  });
});
