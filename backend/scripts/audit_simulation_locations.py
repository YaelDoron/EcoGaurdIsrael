"""Read-only audit of the canonical simulation locations' real land cover.

Diagnostic tool only (never imported by the app, never writes anything).
Queries the PRODUCTION vegetation path - CopernicusLandCoverClient +
VegetationMapper + FireSpreadInputService's dominant-class -> fuel mapping, at
the production VEGETATION_RADIUS_KM - for each canonical location's anchor and
a ring of points around it. The ring matters because simulated satellite
hotspots (and therefore the FireEvent origin whose vegetation spread actually
uses) land 0.2-2.0 km from the anchor, not on it.

Usage (from backend/):
    python -m scripts.audit_simulation_locations
    python -m scripts.audit_simulation_locations --only galilee
    python -m scripts.audit_simulation_locations --candidate "Har Meron W slope" 32.985 35.395
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter

from src.external.copernicus import CopernicusLandCoverClient
from src.mappers.vegetation_mapper import VegetationMapper
from src.services.fire_severity.fire_severity_input_config import VEGETATION_RADIUS_KM
from src.services.fire_spread.fire_spread_input_service import _map_dominant_land_cover
from src.simulation.generators.satellite_data_generator import SIMULATED_HOTSPOT_MAX_RADIUS_KM
from src.simulation.simulation_locations import SIMULATION_LOCATIONS

RING_RADII_KM = (1.0, SIMULATED_HOTSPOT_MAX_RADIUS_KM)
RING_BEARINGS_DEG = (0.0, 90.0, 180.0, 270.0)
_KM_PER_LAT_DEG = 111.32


def _offset(latitude: float, longitude: float, radius_km: float, bearing_deg: float) -> tuple[float, float]:
    bearing = math.radians(bearing_deg)
    d_lat = radius_km * math.cos(bearing) / _KM_PER_LAT_DEG
    d_lon = radius_km * math.sin(bearing) / (_KM_PER_LAT_DEG * math.cos(math.radians(latitude)))
    return round(latitude + d_lat, 6), round(longitude + d_lon, 6)


def sample(client: CopernicusLandCoverClient, mapper: VegetationMapper, latitude: float, longitude: float):
    """(dominant label, fuel name or None, distribution, pixel summary) for one point, via the production path.

    Paced, with one retry: a burst of statistics calls can be rate-limited (HTTP 4xx).
    """
    statistics = None
    for attempt in range(2):
        time.sleep(1.0 + 4.0 * attempt)
        try:
            statistics = client.get_land_cover_statistics(latitude, longitude, VEGETATION_RADIUS_KM)
            break
        except Exception as exc:  # noqa: BLE001 - diagnostic output only.
            if attempt == 1:
                return f"FAILED {type(exc).__name__}", None, (), ""
    pixels = (
        f"{statistics.sample_count} px, {statistics.valid_pixel_count} valid, {statistics.no_data_count} no-data"
        if statistics is not None and statistics.sample_count is not None
        else "px n/a"
    )
    vegetation = mapper.map_statistics(statistics)
    if vegetation is None:
        return "no usable data", None, (), pixels
    fuel = _map_dominant_land_cover(vegetation.dominant_land_cover)
    distribution = tuple(sorted(vegetation.land_cover_distribution, key=lambda item: -item[1]))
    return vegetation.dominant_land_cover, (fuel.name if fuel else None), distribution, pixels


def _format(distribution) -> str:
    return ", ".join(f"{name.replace(' cover', '')} {share:.0%}" for name, share in distribution)


def audit_point(client, mapper, label: str, latitude: float, longitude: float) -> None:
    dominant, fuel, distribution, pixels = sample(client, mapper, latitude, longitude)
    print(f"\n### {label}  ({latitude:.4f}, {longitude:.4f})")
    print(f"  anchor: dominant='{dominant}' -> fuel={fuel}  | {_format(distribution)}  [{pixels}]")
    fuels: Counter = Counter()
    for radius in RING_RADII_KM:
        for bearing in RING_BEARINGS_DEG:
            lat, lon = _offset(latitude, longitude, radius, bearing)
            ring_dominant, ring_fuel, ring_distribution, _ = sample(client, mapper, lat, lon)
            fuels[ring_fuel or f"unsupported:{ring_dominant}"] += 1
            print(f"  {radius:.0f} km @{bearing:>3.0f}: ({lat:.4f}, {lon:.4f}) '{ring_dominant}' -> {ring_fuel}  | {_format(ring_distribution)}")
    print(f"  ring fuel summary ({sum(fuels.values())} points): {dict(fuels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="audit a single canonical location key")
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "LAT", "LON"), default=[])
    parser.add_argument("--no-canonical", action="store_true", help="audit candidates only")
    args = parser.parse_args()

    client, mapper = CopernicusLandCoverClient(), VegetationMapper()
    print(f"Source: Copernicus Global Land Cover 100 m (2019), radius {VEGETATION_RADIUS_KM} km per point")
    if not args.no_canonical:
        for key, location in SIMULATION_LOCATIONS.items():
            if args.only and key != args.only:
                continue
            audit_point(client, mapper, f"{key} - {location.name}", location.latitude, location.longitude)
    for name, lat, lon in args.candidate:
        audit_point(client, mapper, f"candidate - {name}", float(lat), float(lon))
    return 0


if __name__ == "__main__":
    sys.exit(main())
