"""Seed reproducibility of generated simulation DATA (final demo-readiness pass).

Before: every generator mixed the ABSOLUTE event timestamp (run start +
offset) into its per-item RNG seed, so the same preset + seed produced
different weather values, hotspot properties and news text on every run.
Now the executor passes a schedule-derived `seed_key` (event type, source
index, offset), so generated content is a pure function of the seed and the
deterministic schedule; only the absolute timestamps follow the run's start.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.simulation import SIMULATION_LOCATIONS, ScenarioType
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.simulation_event import SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutor, simulation_event_seed_key
from src.simulation.simulation_scenario import build_operations_demo_scenario
from tests.simulation.test_simulation_event_executor import (
    CapturingGenerator,
    FakeNewsRepository,
    FakeSatelliteRepository,
    FakeWeatherRepository,
    make_event,
    make_scenario,
)

MORNING = datetime(2026, 9, 25, 7, 0, tzinfo=timezone.utc)
AFTERNOON = datetime(2026, 9, 25, 14, 37, 11, tzinfo=timezone.utc)  # same day/night window (06-18 UTC)
JERUSALEM = SIMULATION_LOCATIONS["jerusalem_forest"]


def _weather_values(generated):
    return [
        (m.station.external_station_id, m.observation.temperature, m.observation.relative_humidity,
         m.observation.wind_speed, m.observation.wind_direction, m.observation.wind_gust, m.observation.rainfall)
        for m in generated.measurements
    ]


def _hotspot_values(generated):
    return [(h.latitude, h.longitude, h.confidence, h.frp, h.brightness, h.day_night) for h in generated.hotspots]


def _news_values(generated):
    return [(r.title, r.summary, r.location_name, r.latitude, r.longitude) for r in generated.reports]


# --- Generators ---------------------------------------------------------------


def test_weather_values_depend_on_seed_key_not_wall_clock():
    generator = WeatherDataGenerator(seed=545)
    first = generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM, seed_key="weather:0@0")
    second = generator.generate(ScenarioType.ACTIVE_FIRE, AFTERNOON, JERUSALEM, seed_key="weather:0@0")
    assert _weather_values(first) == _weather_values(second)
    assert {m.observation.timestamp for m in second.measurements} == {AFTERNOON}  # timestamps still anchored


def test_satellite_properties_depend_on_seed_key_not_wall_clock():
    generator = SatelliteDataGenerator(seed=545)
    first = generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM, seed_key="satellite:0@40")
    second = generator.generate(ScenarioType.ACTIVE_FIRE, AFTERNOON, JERUSALEM, seed_key="satellite:0@40")
    assert _hotspot_values(first) == _hotspot_values(second)
    assert {h.detected_at for h in second.hotspots} == {AFTERNOON}


def test_news_content_depends_on_seed_key_not_wall_clock():
    generator = NewsDataGenerator(seed=545)
    first = generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM, report_index=1, seed_key="news:1@90")
    second = generator.generate(ScenarioType.ACTIVE_FIRE, AFTERNOON, JERUSALEM, report_index=1, seed_key="news:1@90")
    assert _news_values(first) == _news_values(second)
    assert second.reports[0].published_at == AFTERNOON


def test_different_seed_keys_still_vary_content():
    generator = WeatherDataGenerator(seed=545)
    a = generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM, seed_key="weather:0@0")
    b = generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM, seed_key="weather:1@130")
    assert _weather_values(a) != _weather_values(b)


def test_legacy_callers_without_seed_key_keep_timestamp_seeding():
    generator = WeatherDataGenerator(seed=545)
    assert _weather_values(generator.generate(ScenarioType.ACTIVE_FIRE, MORNING, JERUSALEM)) != _weather_values(
        generator.generate(ScenarioType.ACTIVE_FIRE, AFTERNOON, JERUSALEM)
    )


# --- Executor -------------------------------------------------------------------


def test_seed_key_is_the_event_schedule_identity():
    event = make_event(SimulationEventType.NEWS, source_event_index=1)
    assert simulation_event_seed_key(event) == "news:1@40"


@pytest.mark.parametrize(
    "event_type, kwarg_repo",
    [(SimulationEventType.WEATHER, "weather_generator"), (SimulationEventType.SATELLITE, "satellite_generator"),
     (SimulationEventType.NEWS, "news_generator")],
)
def test_executor_passes_a_timestamp_independent_seed_key(event_type, kwarg_repo):
    empty = {
        "weather_generator": type("G", (), {"measurements": ()})(),
        "satellite_generator": type("G", (), {"hotspots": ()})(),
        "news_generator": type("G", (), {"reports": ()})(),
    }[kwarg_repo]
    generator = CapturingGenerator(empty)
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(), satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(), **{kwarg_repo: generator},
    )
    event = make_event(event_type)
    executor.execute(scenario=make_scenario(), event=event, event_timestamp=MORNING)
    executor.execute(scenario=make_scenario(), event=event, event_timestamp=AFTERNOON)
    keys = [call["seed_key"] for call in generator.calls]
    assert keys == [simulation_event_seed_key(event)] * 2


def test_same_seed_reproduces_every_generated_value_of_a_full_operations_demo_schedule():
    scenario = build_operations_demo_scenario(seed=545)
    incidents = {incident.incident_id: incident for incident in scenario.incidents}

    def run(start: datetime):
        weather, satellite, news = WeatherDataGenerator(545), SatelliteDataGenerator(545), NewsDataGenerator(545)
        out = []
        for event in scenario.events:
            incident = incidents[event.incident_id]
            ts, key = start + timedelta(seconds=event.offset_seconds), simulation_event_seed_key(event)
            if event.event_type is SimulationEventType.WEATHER:
                out.append(_weather_values(weather.generate(incident.scenario_type, ts, incident.location, seed_key=key)))
            elif event.event_type is SimulationEventType.SATELLITE:
                out.append(_hotspot_values(satellite.generate(incident.scenario_type, ts, incident.location, seed_key=key)))
            elif event.event_type is SimulationEventType.NEWS:
                out.append(_news_values(news.generate(incident.scenario_type, ts, incident.location,
                                                      report_index=event.source_event_index, seed_key=key)))
        return out

    morning = run(MORNING)
    assert morning and all(morning)  # every scheduled event generated content
    assert morning == run(AFTERNOON)


# --- AI-relevant features ------------------------------------------------------------


def _v5_features_for_every_active_fire(seed: int, start: datetime):
    """Build FireDetectionEvidence exactly as FireDetectionEvidenceService._normalize_* does from the
    generated hotspots/reports, then run the real V5 feature extractor per active incident."""
    from src.ml.fire_detection.fire_detection_feature_extractor_v5 import FireDetectionFeatureExtractorV5
    from src.models.fire_detection_evidence import FireDetectionEvidence
    from src.models.fire_evidence_type import FireEvidenceType

    scenario = build_operations_demo_scenario(seed=seed)
    satellite, news = SatelliteDataGenerator(seed), NewsDataGenerator(seed)
    evidence_by_incident: dict[str, list] = {}
    next_id = 1
    for event in scenario.events:
        incident = next(i for i in scenario.incidents if i.incident_id == event.incident_id)
        ts, key = start + timedelta(seconds=event.offset_seconds), simulation_event_seed_key(event)
        items = evidence_by_incident.setdefault(event.incident_id, [])
        if event.event_type is SimulationEventType.SATELLITE:
            for h in satellite.generate(incident.scenario_type, ts, incident.location, seed_key=key).hotspots:
                items.append(FireDetectionEvidence(
                    evidence_id=next_id, evidence_type=FireEvidenceType.SATELLITE, latitude=h.latitude,
                    longitude=h.longitude, observed_at=h.detected_at, satellite_confidence="nominal",
                    location_name=h.location_name, satellite_frp=h.frp, satellite_brightness=h.brightness,
                    satellite_day_night=h.day_night, satellite_name=h.satellite, satellite_instrument=h.instrument))
                next_id += 1
        elif event.event_type is SimulationEventType.NEWS:
            for r in news.generate(incident.scenario_type, ts, incident.location,
                                   report_index=event.source_event_index, seed_key=key).reports:
                items.append(FireDetectionEvidence(
                    evidence_id=next_id, evidence_type=FireEvidenceType.NEWS, latitude=r.latitude,
                    longitude=r.longitude, observed_at=r.published_at))
                next_id += 1
    extractor = FireDetectionFeatureExtractorV5()
    return {iid: extractor.extract(tuple(items)) for iid, items in evidence_by_incident.items() if items}


@pytest.mark.parametrize("seed", [7, 545, 951])
def test_same_seed_gives_identical_v5_ai_feature_vectors_regardless_of_start_time(seed):
    morning = _v5_features_for_every_active_fire(seed, MORNING)
    assert morning  # at least one active fire with evidence
    assert morning == _v5_features_for_every_active_fire(seed, AFTERNOON)


def test_different_seeds_still_vary_incidents_values_and_ai_features():
    a, b = build_operations_demo_scenario(seed=951), build_operations_demo_scenario(seed=952)
    assert [e.offset_seconds for e in a.events] != [e.offset_seconds for e in b.events]  # schedule varies
    fa, fb = _v5_features_for_every_active_fire(951, MORNING), _v5_features_for_every_active_fire(952, MORNING)
    assert fa != fb
