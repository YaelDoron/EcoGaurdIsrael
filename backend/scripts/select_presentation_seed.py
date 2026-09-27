"""Offline check/search of `presentation_demo` seeds (read-only: no database, no network).

For a candidate seed this replays the presentation timeline through the REAL
pieces the live pipeline uses - the seeded weather/satellite/news generators
(with each event's schedule seed key), the V5 feature extractor, the HGB V5
model runtime and the AI Hybrid V5 policy's decide_status - and reports each
fire's AI status after every satellite/news event, plus the maximum PROPAGATOR
transition probability each fire's generated weather allows.

It never changes a threshold, model or generator: it only looks for a seed
whose values tell the presentation story, for a daytime AND a night-time
start (the V5 features include the day/night flag of the observation time):

  * Judean Hills: SUSPECTED on its first pass, CONFIRMED on its second pass,
    and its (grassland) weather keeps spread above PROPAGATION_THRESHOLD
    with a margin;
  * Galilee: SUSPECTED on its first pass, CONFIRMED on its second pass, and
    risk-only spread (below the threshold with a margin for both the
    GENERIC_TREE and SHRUBS readings of its borderline land cover);
  * Jerusalem Forest (fire C): SUSPECTED on its first pass, CONFIRMED on its
    second pass, and every weather observation its Presentation weather
    profile generates lets its audited Copernicus fuel ('Shrub cover' ->
    SHRUBS) propagate (downwind p >= PROPAGATION_THRESHOLD).

Approximation (as in the earlier seed audits): each fire's accumulated
evidence is scored as one candidate (no separate history object); it
reproduced the live run's AI scores exactly.

Usage (from backend/):
    python -m scripts.select_presentation_seed            # check the default seed
    python -m scripts.select_presentation_seed 893 894    # check specific seeds
    python -m scripts.select_presentation_seed --search 1 2000
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from scripts.calibrate_fire_spread_threshold import fuel_moisture_percent
from src.calculators.fire_spread.fire_spread_calculator import _MOORE_OFFSETS, transition_probability
from src.config.settings import settings
from src.ml.fire_detection.fire_detection_ai_hybrid_policy_v5 import decide_status
from src.ml.fire_detection.fire_detection_dataset_v5 import FIRE_DETECTION_FEATURE_NAMES_V5
from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
from src.ml.fire_detection.fire_detection_model_runtime_v5 import FireDetectionModelV5Runtime
from src.ml.fire_detection.fire_detection_policy_config_v5 import MULTI_PIXEL_COUNT_FEATURES
from src.models.fire_detection_evidence import FireDetectionEvidence
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_spread_fuel_class import FireSpreadFuelClass
from src.models.fire_spread_prediction import PROPAGATION_THRESHOLD
from src.services.fire_detection.fire_detection_evidence_service import _SATELLITE_CONFIDENCE_MAP
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.presentation_scenario import (
    GALILEE,
    JERUSALEM_FOREST_FIRE,
    JUDEAN_HILLS,
    PRESENTATION_DEFAULT_SEED,
    build_presentation_demo_scenario,
)
from src.simulation.scenario_type import ScenarioType
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_event_executor import simulation_event_seed_key
from src.simulation.simulation_scenario import SimulationScenario

DAY_START = datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc)
NIGHT_START = datetime(2026, 9, 28, 21, 0, tzinfo=timezone.utc)
# Only the scripted opening has to tell the story; later updates may evolve.
STORY_WINDOW_SECONDS = 640
SPREAD_MARGIN = 0.02

_PIXEL_FEATURE_INDEXES = [FIRE_DETECTION_FEATURE_NAMES_V5.index(name) for name in MULTI_PIXEL_COUNT_FEATURES]


class _Predictor:
    def __init__(self) -> None:
        self._runtime = FireDetectionModelV5Runtime(
            model_path=settings.FIRE_DETECTION_AI_V5_MODEL_PATH,
            metadata_path=settings.FIRE_DETECTION_AI_V5_METADATA_PATH,
        )
        self._extractor = FireDetectionFeatureExtractorV5()

    def statuses(self, scenario: SimulationScenario, start: datetime) -> dict[str, list[tuple[int, str, float, str]]]:
        """incident id -> [(offset, event type, AI probability, status)] after each satellite/news event."""
        satellite = SatelliteDataGenerator(scenario.seed)
        news = NewsDataGenerator(scenario.seed)
        evidence: dict[str, list[FireDetectionEvidence]] = {}
        out: dict[str, list[tuple[int, str, float, str]]] = {}
        next_id = 1
        for event in scenario.events:
            incident = scenario.get_incident(event.incident_id)
            if incident.scenario_type is not ScenarioType.ACTIVE_FIRE or event.event_type is SimulationEventType.WEATHER:
                continue
            if event.offset_seconds > STORY_WINDOW_SECONDS:
                break
            timestamp = start + timedelta(seconds=event.offset_seconds)
            seed_key = simulation_event_seed_key(event)
            items = evidence.setdefault(event.incident_id, [])
            if event.event_type is SimulationEventType.SATELLITE:
                for hotspot in satellite.generate(incident.scenario_type, timestamp, incident.location, seed_key=seed_key).hotspots:
                    items.append(
                        FireDetectionEvidence(
                            evidence_id=next_id,
                            evidence_type=FireEvidenceType.SATELLITE,
                            latitude=hotspot.latitude,
                            longitude=hotspot.longitude,
                            observed_at=hotspot.detected_at,
                            satellite_confidence=_SATELLITE_CONFIDENCE_MAP.get(hotspot.confidence.lower()),
                            location_name=hotspot.location_name,
                            satellite_frp=hotspot.frp,
                            satellite_brightness=hotspot.brightness,
                            satellite_day_night=hotspot.day_night,
                            satellite_name=hotspot.satellite,
                            satellite_instrument=hotspot.instrument,
                        )
                    )
                    next_id += 1
            else:
                generated = news.generate(
                    incident.scenario_type,
                    timestamp,
                    incident.location,
                    report_index=event.source_event_index,
                    seed_key=seed_key,
                )
                for report in generated.reports:
                    items.append(
                        FireDetectionEvidence(
                            evidence_id=next_id,
                            evidence_type=FireEvidenceType.NEWS,
                            latitude=report.latitude,
                            longitude=report.longitude,
                            observed_at=report.published_at,
                        )
                    )
                    next_id += 1
            if not any(item.evidence_type is FireEvidenceType.SATELLITE for item in items):
                continue  # a news-only candidate is not scored by the AI path
            features = self._extractor.extract(tuple(items))
            probability = self._runtime.predict_probability(features)
            values = features.values if hasattr(features, "values") else tuple(features)
            pixels = sum(values[index] for index in _PIXEL_FEATURE_INDEXES)
            status = decide_status(probability, pixels)
            out.setdefault(event.incident_id, []).append(
                (event.offset_seconds, event.event_type.value, round(probability, 3), getattr(status, "value", str(status)))
            )
        return out


def spread_probabilities(scenario: SimulationScenario, incident_id: str, fuel: FireSpreadFuelClass) -> list[float]:
    """Max (over the 8 neighbour bearings) transition probability for each weather event in the story window."""
    generator = WeatherDataGenerator(scenario.seed)
    incident = scenario.get_incident(incident_id)
    values: list[float] = []
    for event in scenario.events:
        if event.incident_id != incident_id or event.event_type is not SimulationEventType.WEATHER:
            continue
        if event.offset_seconds > STORY_WINDOW_SECONDS:
            break
        generated = generator.generate(
            incident.scenario_type,
            DAY_START + timedelta(seconds=event.offset_seconds),
            incident.location,
            seed_key=simulation_event_seed_key(event),
            profile=incident.weather_profile,
        )
        for measurement in generated.measurements:
            observation = measurement.observation
            moisture = fuel_moisture_percent(observation.temperature, observation.relative_humidity)
            values.append(
                max(
                    transition_probability(
                        fuel, fuel, observation.wind_speed, observation.wind_direction, bearing, moisture
                    )
                    for _, bearing in _MOORE_OFFSETS
                )
            )
    return values


def satellite_pass_offsets(scenario: SimulationScenario, incident_id: str) -> list[int]:
    return [
        event.offset_seconds
        for event in scenario.events
        if event.incident_id == incident_id and event.event_type is SimulationEventType.SATELLITE
    ]


def _status_after(steps: list[tuple[int, str, float, str]], offset: int) -> str | None:
    matching = [step for step in steps if step[0] == offset]
    return matching[-1][3].upper() if matching else None


def evaluate(predictor: _Predictor, seed: int) -> tuple[bool, list[str]]:
    scenario = build_presentation_demo_scenario(seed)
    notes: list[str] = []
    runs = {label: predictor.statuses(scenario, start) for label, start in (("day", DAY_START), ("night", NIGHT_START))}
    ok = True
    jh_passes, gal_passes, jf_passes = (
        satellite_pass_offsets(scenario, incident_id) for incident_id in (JUDEAN_HILLS, GALILEE, JERUSALEM_FOREST_FIRE)
    )
    for label, statuses in runs.items():
        jh, gal, jf = (statuses.get(incident_id, []) for incident_id in (JUDEAN_HILLS, GALILEE, JERUSALEM_FOREST_FIRE))
        # Schedule-derived: each fire's first pass must give SUSPECTED and its
        # second pass CONFIRMED.
        checks = {
            f"Judean Hills SUSPECTED at T+{jh_passes[0]}": _status_after(jh, jh_passes[0]) == "SUSPECTED",
            f"Judean Hills CONFIRMED at T+{jh_passes[1]}": _status_after(jh, jh_passes[1]) == "CONFIRMED",
            f"Galilee SUSPECTED at T+{gal_passes[0]}": _status_after(gal, gal_passes[0]) == "SUSPECTED",
            f"Galilee CONFIRMED at T+{gal_passes[1]}": _status_after(gal, gal_passes[1]) == "CONFIRMED",
            f"Jerusalem Forest SUSPECTED at T+{jf_passes[0]}": _status_after(jf, jf_passes[0]) == "SUSPECTED",
            f"Jerusalem Forest CONFIRMED at T+{jf_passes[1]}": _status_after(jf, jf_passes[1]) == "CONFIRMED",
        }
        for name, passed in checks.items():
            ok &= passed
            if not passed:
                notes.append(f"[{label}] FAILED {name}")
    status_sequence = {
        label: {incident: [(t, status) for t, _, _, status in steps] for incident, steps in statuses.items()}
        for label, statuses in runs.items()
    }
    if status_sequence["day"] != status_sequence["night"]:
        ok = False
        notes.append("day and night status sequences differ")

    jh_spread = spread_probabilities(scenario, JUDEAN_HILLS, FireSpreadFuelClass.GRASSLAND)
    galilee_tree = spread_probabilities(scenario, GALILEE, FireSpreadFuelClass.GENERIC_TREE)
    galilee_shrubs = spread_probabilities(scenario, GALILEE, FireSpreadFuelClass.SHRUBS)
    if min(jh_spread) < PROPAGATION_THRESHOLD + SPREAD_MARGIN:
        ok = False
        notes.append(f"Judean Hills spread p min {min(jh_spread):.3f} too close to/below {PROPAGATION_THRESHOLD}")
    # Galilee resolves to GENERIC_TREE ("Tree cover"); it must stay risk-only with
    # a margin, and also stay below the threshold if its borderline land cover
    # were ever read as SHRUBS.
    if max(galilee_tree) >= PROPAGATION_THRESHOLD - SPREAD_MARGIN or max(galilee_shrubs) >= PROPAGATION_THRESHOLD:
        ok = False
        notes.append(
            f"Galilee spread p max {max(galilee_tree):.3f} (tree) / {max(galilee_shrubs):.3f} (shrubs) "
            f"too close to/above {PROPAGATION_THRESHOLD}"
        )
    # Fire C's audited Copernicus fuel is 'Shrub cover' -> SHRUBS: every weather
    # observation its Presentation profile generates must let it propagate.
    jf_spread = spread_probabilities(scenario, JERUSALEM_FOREST_FIRE, FireSpreadFuelClass.SHRUBS)
    if min(jf_spread) < PROPAGATION_THRESHOLD:
        ok = False
        notes.append(f"Jerusalem Forest spread p min {min(jf_spread):.3f} below {PROPAGATION_THRESHOLD}")
    notes.append(
        f"spread p: Judean Hills min {min(jh_spread):.3f}, Galilee max {max(galilee_tree):.3f} (GENERIC_TREE) / "
        f"{max(galilee_shrubs):.3f} (SHRUBS), Jerusalem Forest min {min(jf_spread):.3f} (SHRUBS) "
        f"(threshold {PROPAGATION_THRESHOLD})"
    )
    for incident_id, steps in runs["day"].items():
        notes.append(f"{incident_id}: " + "  ->  ".join(f"T+{t} {kind[:3]} p={p} {status}" for t, kind, p, status in steps))
    return ok, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", nargs="*", type=int)
    parser.add_argument("--search", nargs=2, type=int, metavar=("FIRST", "LAST"))
    args = parser.parse_args()
    predictor = _Predictor()
    if args.search:
        first, last = args.search
        hits = [seed for seed in range(first, last + 1) if evaluate(predictor, seed)[0]]
        print(f"seeds satisfying the presentation story in [{first}, {last}]: {hits}")
        return
    for seed in args.seeds or [PRESENTATION_DEFAULT_SEED]:
        ok, notes = evaluate(predictor, seed)
        print(f"seed {seed}: {'OK' if ok else 'NOT SUITABLE'}")
        for note in notes:
            print(f"   {note}")


if __name__ == "__main__":
    main()
