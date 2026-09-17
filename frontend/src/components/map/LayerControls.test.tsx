import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LayerControls } from "./LayerControls";
import type { LayerToggle } from "./mapTypes";

const LAYERS: LayerToggle[] = [
  { id: "spread", label: "Predicted spread", count: 5 },
  { id: "stations", label: "Fire stations" },
];

describe("LayerControls", () => {
  it("renders one checkbox per layer, labeled and checked according to visibility", () => {
    render(<LayerControls layers={LAYERS} visibility={{ spread: true, stations: false }} onToggle={vi.fn()} />);

    const spreadCheckbox = screen.getByRole("checkbox", { name: /predicted spread/i });
    const stationsCheckbox = screen.getByRole("checkbox", { name: /fire stations/i });
    expect(spreadCheckbox).toBeChecked();
    expect(stationsCheckbox).not.toBeChecked();
  });

  it("treats a layer missing from visibility as visible by default", () => {
    render(<LayerControls layers={LAYERS} visibility={{}} onToggle={vi.fn()} />);

    expect(screen.getByRole("checkbox", { name: /predicted spread/i })).toBeChecked();
  });

  it("shows the optional item count next to a layer's label", () => {
    render(<LayerControls layers={LAYERS} visibility={{}} onToggle={vi.fn()} />);

    expect(screen.getByText("5")).toBeInTheDocument();
  });

  it("calls onToggle with the layer id when its checkbox is clicked", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    render(<LayerControls layers={LAYERS} visibility={{ spread: true }} onToggle={onToggle} />);

    await user.click(screen.getByRole("checkbox", { name: /predicted spread/i }));

    expect(onToggle).toHaveBeenCalledWith("spread");
  });
});
