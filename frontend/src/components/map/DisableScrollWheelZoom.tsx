import { useEffect } from "react";
import { useMap } from "react-leaflet";

/**
 * Belt-and-suspenders fix for the Operations Overview map (manual browser
 * testing showed the mouse wheel/trackpad still zoomed the map despite
 * `MapView`'s `scrollWheelZoom={false}` prop).
 *
 * Root cause: react-leaflet v5's `MapContainer` only applies handler props
 * like `scrollWheelZoom` ONCE, at the moment it constructs the underlying
 * Leaflet `Map` instance (see react-leaflet's `MapContainer.js`: the
 * `mapRef` callback is created via `useCallback(fn, [])` - an empty
 * dependency array - so it captures whatever `options.scrollWheelZoom` was
 * on the very first call and never reads it again). There is no reactive
 * prop-sync path afterward, so any timing where the map gets constructed
 * before this is fully settled leaves Leaflet's own ScrollWheelZoom
 * `Handler` enabled regardless of the prop.
 *
 * The fix calls Leaflet's own supported, imperative Handler API directly on
 * the live map instance: `map.scrollWheelZoom.disable()` (see
 * leaflet/src/map/Map.js - `Map.addInitHook('addHandler', 'scrollWheelZoom',
 * ScrollWheelZoom)` registers exactly this enable()/disable() pair).
 * Disabling the handler this way means Leaflet's own wheel listener simply
 * stops reacting to the event - it never calls `preventDefault()` and never
 * touches CSS `pointer-events` - so the browser's native wheel-scroll
 * behavior (scrolling the page) is completely unaffected, and dragging,
 * the +/-  zoom controls, and marker/popup clicks are untouched (those are
 * separate Leaflet handlers, never disabled by this component).
 *
 * Must be rendered as a child of `MapContainer` (via `MapView`'s
 * `children`), since `useMap()` only works inside that context - see
 * OperationsMap.tsx for the one place this is actually used. Event
 * Details/Response Plan do not render this, so their maps keep the
 * ordinary wheel-zoom behavior.
 */
export function DisableScrollWheelZoom() {
  const map = useMap();

  useEffect(() => {
    map.scrollWheelZoom.disable();
  }, [map]);

  return null;
}
