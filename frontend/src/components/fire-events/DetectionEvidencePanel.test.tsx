import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DetectionEvidencePanel } from "./DetectionEvidencePanel";
import type { DetectionEvidence, NewsEvidence, SatelliteEvidence } from "../../types/eventDetails";

function makeSatellite(overrides: Partial<SatelliteEvidence> = {}): SatelliteEvidence {
  return {
    id: 501,
    detected_at: "2026-09-17T13:15:00Z",
    latitude: 32.7,
    longitude: 35.0,
    confidence: "high",
    frp: 15.2,
    brightness: 310.5,
    satellite: "Terra",
    instrument: "MODIS",
    day_night: "D",
    ...overrides,
  };
}

function makeNews(overrides: Partial<NewsEvidence> = {}): NewsEvidence {
  return {
    id: 701,
    title: "Blaze reported near reserve",
    summary: "A wildfire was reported near the nature reserve.",
    source: "haaretz",
    observed_at: "2026-09-17T13:10:00Z",
    location_name: "Modiin",
    latitude: 31.9,
    longitude: 35.0,
    ...overrides,
  };
}

function makeEvidence(overrides: Partial<DetectionEvidence> = {}): DetectionEvidence {
  return { satellite: [], news: [], ...overrides };
}

describe("DetectionEvidencePanel", () => {
  it("shows an empty state when there is no evidence at all", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence()} />);

    expect(screen.getByText("No detection evidence")).toBeInTheDocument();
  });

  it("renders satellite hotspot details", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite()] })} />);

    expect(screen.getByText("Satellite Hotspots")).toBeInTheDocument();
    expect(screen.getByText("Terra")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
    expect(screen.getByText("MODIS")).toBeInTheDocument();
    expect(screen.getByText("Day")).toBeInTheDocument();
    expect(screen.getByText("15.2")).toBeInTheDocument();
    expect(screen.getByText("310.5")).toBeInTheDocument();
  });

  it.each([
    ["n", "Nominal (n)"],
    ["N", "Nominal (n)"],
    ["h", "High (h)"],
    ["H", "High (h)"],
    ["l", "Low (l)"],
    ["L", "Low (l)"],
  ])("translates raw confidence %s into %s", (raw, expected) => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite({ confidence: raw })] })} />);

    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it("passes an unknown confidence value through unchanged", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite({ confidence: "87" })] })} />);

    expect(screen.getByText("87")).toBeInTheDocument();
  });

  it("shows a source descriptor above the satellite data", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite()] })} />);

    expect(screen.getByText("Source: Meteorological Satellite")).toBeInTheDocument();
  });

  it("omits satellite fields the backend sent as null, without fabricating placeholders", () => {
    render(
      <DetectionEvidencePanel
        evidence={makeEvidence({
          satellite: [
            makeSatellite({ confidence: null, instrument: null, day_night: null, frp: null, brightness: null }),
          ],
        })}
      />,
    );

    expect(screen.queryByText("Confidence")).not.toBeInTheDocument();
    expect(screen.queryByText("Instrument")).not.toBeInTheDocument();
    expect(screen.queryByText("Day / Night")).not.toBeInTheDocument();
    expect(screen.queryByText("FRP")).not.toBeInTheDocument();
    expect(screen.queryByText("Brightness")).not.toBeInTheDocument();
  });

  it("falls back to a generic label when satellite name is null", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite({ satellite: null })] })} />);

    expect(screen.getByText("Satellite hotspot")).toBeInTheDocument();
  });

  it("renders news report details", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ news: [makeNews()] })} />);

    expect(screen.getByText("News Reports")).toBeInTheDocument();
    expect(screen.getByText("Blaze reported near reserve")).toBeInTheDocument();
    expect(screen.getByText("A wildfire was reported near the nature reserve.")).toBeInTheDocument();
    expect(screen.getByText("haaretz")).toBeInTheDocument();
    expect(screen.getByText("Modiin")).toBeInTheDocument();
  });

  it("omits the location fact when the backend sent no location_name", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ news: [makeNews({ location_name: null })] })} />);

    expect(screen.queryByText("Location")).not.toBeInTheDocument();
  });

  it("renders both groups together when both kinds of evidence exist", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite()], news: [makeNews()] })} />);

    expect(screen.getByText("Satellite Hotspots")).toBeInTheDocument();
    expect(screen.getByText("News Reports")).toBeInTheDocument();
  });

  it("omits the satellite group entirely when there is only news evidence", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ news: [makeNews()] })} />);

    expect(screen.queryByText("Satellite Hotspots")).not.toBeInTheDocument();
  });

  it("does not repeat coordinates on satellite cards (they are in the Fire Event card)", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite()] })} />);

    expect(screen.queryByText(/32\.7000/)).not.toBeInTheDocument();
  });

  it("condenses instrument, day/night, FRP and brightness into small muted pills", () => {
    render(<DetectionEvidencePanel evidence={makeEvidence({ satellite: [makeSatellite()] })} />);

    const pills = screen.getByRole("list", { name: "Sensor details" });
    expect(pills.querySelectorAll("li")).toHaveLength(4);
    expect(within(pills).getByText("MODIS")).toBeInTheDocument();
    // Confidence stays a primary fact, outside the pills.
    expect(within(pills).queryByText("Confidence")).not.toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
  });

  it("renders no pill list when the sensor sent none of the technical fields", () => {
    render(
      <DetectionEvidencePanel
        evidence={makeEvidence({
          satellite: [makeSatellite({ instrument: null, day_night: null, frp: null, brightness: null })],
        })}
      />,
    );

    expect(screen.queryByRole("list", { name: "Sensor details" })).not.toBeInTheDocument();
  });
});
