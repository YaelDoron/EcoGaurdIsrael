import { useEffect } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";
import type { LatLngPoint } from "./mapTypes";

export interface FitBoundsToPointsProps {
  points: LatLngPoint[];
  /** Zoom used when there is exactly one point (fitBounds needs a range of at least two). */
  singlePointZoom?: number;
  padding?: number;
}

const DEFAULT_SINGLE_POINT_ZOOM = 13;
const DEFAULT_PADDING = 48;

/**
 * Fits the map's viewport to whatever points it is given, every time they
 * change. This is the ONLY place the viewport's center/zoom is decided -
 * no layer or page sets a fixed center of its own, so the map always frames
 * real fetched data and never a hardcoded demo coordinate.
 */
export function FitBoundsToPoints({
  points,
  singlePointZoom = DEFAULT_SINGLE_POINT_ZOOM,
  padding = DEFAULT_PADDING,
}: FitBoundsToPointsProps) {
  const map = useMap();

  useEffect(() => {
    if (points.length === 0) {
      return;
    }
    if (points.length === 1) {
      map.setView([points[0].lat, points[0].lng], singlePointZoom);
      return;
    }
    const bounds = L.latLngBounds(points.map((point): [number, number] => [point.lat, point.lng]));
    map.fitBounds(bounds, { padding: [padding, padding] });
  }, [points, map, singlePointZoom, padding]);

  return null;
}
