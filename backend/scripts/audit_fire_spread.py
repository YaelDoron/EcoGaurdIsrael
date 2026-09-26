"""Read-only audit of one FireEvent's persisted spread prediction.

Diagnostic tool only (never imported by the app, never writes anything):
reloads the exact persisted inputs the spread pipeline used for a FireEvent
(its severity assessment's vegetation snapshot and traced weather), rebuilds
the FireSpreadInput through the real FireSpreadInputService, re-runs the real
FireSpreadCalculator, and prints every emitted cell with its PROPAGATOR
factors (p_n, alpha_wh, e_m) against PROPAGATION_THRESHOLD.

Usage (from backend/):
    python -m scripts.audit_fire_spread --fire-event-id 726
    python -m scripts.audit_fire_spread --fire-event-id 726 --live-copernicus

`--live-copernicus` additionally re-queries Copernicus for the event point
and the canonical simulation location (one external call each).
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone

from src.calculators.fire_spread.fire_spread_calculator import (
    _MOORE_OFFSETS,
    FireSpreadCalculator,
    moisture_factor,
    nominal_spread_probability,
    wind_topography_factor,
)
from src.calculators.fire_spread.fire_spread_config import (
    CA_TIME_STEP_MINUTES,
    METHODOLOGY_VERSION,
    PROPAGATION_THRESHOLD,
)
from src.models.fire_spread_input import FireSpreadInput
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService
from src.utils.geo import haversine_distance_km

_BEARING_NAMES = {0.0: "N", 45.0: "NE", 90.0: "E", 135.0: "SE", 180.0: "S", 225.0: "SW", 270.0: "W", 315.0: "NW"}


def max_first_step_probability(input_data: FireSpreadInput) -> tuple[float, str]:
    """Best p_ij any origin neighbour can get in step 1 (same fuel class everywhere)."""
    best = max(
        (
            _transition(input_data, bearing),
            _BEARING_NAMES[bearing],
        )
        for _, bearing in _MOORE_OFFSETS
    )
    return best


def _transition(input_data: FireSpreadInput, bearing: float) -> float:
    p_n = nominal_spread_probability(input_data.fuel_class, input_data.fuel_class)
    alpha = wind_topography_factor(input_data.wind_speed_kmh, input_data.wind_direction_deg, bearing)
    e_m = moisture_factor(input_data.fuel_moisture_percent)
    return min(max((1.0 - (1.0 - p_n) ** alpha) * e_m, 0.0), 1.0)


def audit(fire_event_id: int, live_copernicus: bool) -> int:
    event = FireEventRepository().get_by_id(fire_event_id)
    if event is None:
        print(f"FireEvent {fire_event_id} not found.")
        return 1

    predictions = FireSpreadPredictionRepository().get_latest_for_event_and_horizons_as_of(
        fire_event_id, (30, 60), datetime.now(timezone.utc)
    )
    print(f"=== Fire Spread Audit: FireEvent {fire_event_id} ===")
    print(f"status={event.event.status.value}  origin=({event.event.latitude:.6f}, {event.event.longitude:.6f})")
    if not predictions:
        print("No persisted spread prediction.")
        return 1

    severity_repo = FireSeverityAssessmentRepository()
    calculator = FireSpreadCalculator()
    for horizon in sorted(predictions):
        stored = predictions[horizon]
        prediction = stored.prediction
        print(f"\n--- {horizon}-minute horizon (persisted prediction {stored.id}) ---")
        print(
            f"status={prediction.status.value}  methodology={prediction.methodology} v{prediction.methodology_version}"
            f"  predicted_at={prediction.predicted_at.isoformat()}  reason={prediction.insufficient_data_reason}"
        )
        if prediction.severity_assessment_id is None:
            continue
        severity = severity_repo.get_by_id(prediction.severity_assessment_id)
        a = severity.assessment
        print(
            f"severity: id={severity.assessment_id} level={a.level.value if a.level else None} score={a.score}"
            f"  (severity is NOT an input to the spread calculator; it only gates on status=valid)"
        )
        print(
            f"vegetation snapshot: dominant='{a.vegetation_dominant_land_cover}' fuel_score={a.vegetation_fuel_score}"
            f" source={a.vegetation_source} year={a.vegetation_dataset_year} radius_km={a.vegetation_radius_km}"
        )

        context = FireSpreadInputService().prepare_shared_context_for_event(
            event, prediction.predicted_at, resolved_severity=severity
        )
        result = FireSpreadInputService.build_input_for_horizon(context, horizon)
        if result.input_data is None:
            print(f"input not READY: {result.status.value} {result.insufficient_data_reason}")
            continue
        inp = result.input_data
        weather = WeatherRepository().get_observations_by_ids((result.weather_observation_id,))[0]
        obs = weather.observation
        print(
            f"weather obs {result.weather_observation_id}: T={obs.temperature}C RH={obs.relative_humidity}%"
            f" wind={obs.wind_speed} km/h from {obs.wind_direction} deg  station=({weather.station.latitude:.4f},"
            f" {weather.station.longitude:.4f}) dist={haversine_distance_km(event.event.latitude, event.event.longitude, weather.station.latitude, weather.station.longitude):.2f} km"
        )
        e_m = moisture_factor(inp.fuel_moisture_percent)
        p_n = nominal_spread_probability(inp.fuel_class, inp.fuel_class)
        print(
            f"calculator input: fuel_class={inp.fuel_class.name} p_n={p_n:.3f}"
            f" fuel_moisture(EMC)={inp.fuel_moisture_percent:.2f}% e_m={e_m:.4f}"
            f" wind_kmh={inp.wind_speed_kmh:.2f}"
        )
        best_p, best_dir = max_first_step_probability(inp)
        print(f"best possible step-1 p_ij = {best_p:.4f} ({best_dir})  vs  PROPAGATION_THRESHOLD = {PROPAGATION_THRESHOLD}")

        recomputed = calculator.calculate(inp)
        persisted = sorted((round(c.latitude, 6), round(c.longitude, 6), round(c.spread_probability, 6)) for c in prediction.cells)
        again = sorted((round(c.latitude, 6), round(c.longitude, 6), round(c.spread_probability, 6)) for c in recomputed.cells)
        print(f"persisted cells={len(persisted)} recomputed cells={len(again)} identical={persisted == again}")

        print(f"{'#':>2} {'dir':>3} {'lat':>10} {'lon':>10} {'dist_m':>7} {'p_n':>6} {'alpha_wh':>8} {'e_m':>6} {'p_ij':>7} {'thr':>4} propagated")
        for index, cell in enumerate(recomputed.cells, start=1):
            bearing = _bearing_from_origin(event.event.latitude, event.event.longitude, cell.latitude, cell.longitude)
            alpha = wind_topography_factor(inp.wind_speed_kmh, inp.wind_direction_deg, bearing)
            distance_m = haversine_distance_km(event.event.latitude, event.event.longitude, cell.latitude, cell.longitude) * 1000
            print(
                f"{index:>2} {_BEARING_NAMES.get(bearing, '?'):>3} {cell.latitude:>10.6f} {cell.longitude:>10.6f} {distance_m:>7.0f}"
                f" {p_n:>6.3f} {alpha:>8.4f} {e_m:>6.4f} {cell.spread_probability:>7.4f} {PROPAGATION_THRESHOLD:>4}"
                f" {'yes' if cell.spread_probability >= PROPAGATION_THRESHOLD else 'no (risk only)'}"
                f"  step={cell.reached_step} ({cell.reached_step * CA_TIME_STEP_MINUTES} min)"
            )

    if live_copernicus:
        _live_copernicus(event.event.latitude, event.event.longitude)
    print(f"\nmethodology version in code: {METHODOLOGY_VERSION}")
    return 0


def _bearing_from_origin(lat0: float, lon0: float, lat: float, lon: float) -> float:
    east = (lon - lon0) * math.cos(math.radians(lat0))
    north = lat - lat0
    bearing = math.degrees(math.atan2(east, north)) % 360.0
    return min(_BEARING_NAMES, key=lambda b: min(abs(b - bearing), 360 - abs(b - bearing)))


def _live_copernicus(event_lat: float, event_lon: float) -> None:
    from src.external.copernicus import CopernicusLandCoverClient
    from src.mappers.vegetation_mapper import VegetationMapper
    from src.services.fire_severity.fire_severity_input_service import VEGETATION_RADIUS_KM
    from src.simulation.simulation_locations import SIMULATION_LOCATIONS

    client = CopernicusLandCoverClient()
    mapper = VegetationMapper()
    canonical = SIMULATION_LOCATIONS["jerusalem_forest"]
    points = [("event origin", event_lat, event_lon), ("canonical jerusalem_forest", canonical.latitude, canonical.longitude)]
    print(f"\n--- live Copernicus (radius {VEGETATION_RADIUS_KM} km) ---")
    for label, lat, lon in points:
        try:
            stats = client.get_land_cover_statistics(lat, lon, VEGETATION_RADIUS_KM)
        except Exception as exc:  # noqa: BLE001 - diagnostic output only.
            print(f"{label} ({lat:.6f}, {lon:.6f}): retrieval FAILED {type(exc).__name__}")
            continue
        veg = mapper.map_statistics(stats)
        if veg is None:
            print(f"{label} ({lat:.6f}, {lon:.6f}): no usable data")
            continue
        distribution = ", ".join(f"{name}={share:.1%}" for name, share in sorted(veg.land_cover_distribution, key=lambda x: -x[1]))
        print(f"{label} ({lat:.6f}, {lon:.6f}): dominant='{veg.dominant_land_cover}' fuel_score={veg.fuel_score:.3f}")
        print(f"    {distribution}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fire-event-id", type=int, required=True)
    parser.add_argument("--live-copernicus", action="store_true")
    args = parser.parse_args()
    return audit(args.fire_event_id, args.live_copernicus)


if __name__ == "__main__":
    sys.exit(main())
