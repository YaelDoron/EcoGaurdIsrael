import type { LayerToggle, LayerVisibility } from "./mapTypes";
import "./LayerControls.css";

export interface LayerControlsProps {
  layers: LayerToggle[];
  visibility: LayerVisibility;
  onToggle: (layerId: string) => void;
}

/**
 * Generic map-layer visibility panel: a checkbox per `LayerToggle`. Holds no
 * knowledge of what a layer draws - the page owns the visibility state and
 * decides which layer components to render based on it. Reusable as-is for
 * US 6.3's routing layers.
 */
export function LayerControls({ layers, visibility, onToggle }: LayerControlsProps) {
  return (
    <fieldset className="layer-controls">
      <legend className="layer-controls__legend">Map layers</legend>
      <div className="layer-controls__list">
        {layers.map((layer) => {
          const isVisible = visibility[layer.id] ?? true;
          return (
            <label key={layer.id} className="layer-controls__item">
              <input
                type="checkbox"
                checked={isVisible}
                onChange={() => onToggle(layer.id)}
                className="layer-controls__checkbox"
              />
              <span className="layer-controls__label">{layer.label}</span>
              {typeof layer.count === "number" ? (
                <span className="layer-controls__count">{layer.count}</span>
              ) : null}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
