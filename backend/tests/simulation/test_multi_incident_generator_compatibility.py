"""Compatibility tests for feeding multi-incident events into source generators."""

from datetime import datetime, timezone

from src.simulation import (
    SimulationEventType,
    build_carmel_golan_active_fire_scenario,
)
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator

TIMESTAMP = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)


def test_multi_incident_events_provide_generator_inputs_without_runtime_wiring():
    scenario = build_carmel_golan_active_fire_scenario(seed=123)
    weather_generator = WeatherDataGenerator(seed=scenario.seed)
    satellite_generator = SatelliteDataGenerator(seed=scenario.seed)
    news_generator = NewsDataGenerator(seed=scenario.seed)

    generated_by_event = {}
    for event in scenario.events:
        incident = scenario.get_incident(event.incident_id)
        if event.event_type is SimulationEventType.WEATHER:
            generated_by_event[event] = weather_generator.generate(
                scenario_type=incident.scenario_type,
                timestamp=TIMESTAMP,
                location=incident.location,
            )
        elif event.event_type is SimulationEventType.SATELLITE:
            generated_by_event[event] = satellite_generator.generate(
                scenario_type=incident.scenario_type,
                timestamp=TIMESTAMP,
                location=incident.location,
            )
        else:
            generated_by_event[event] = news_generator.generate(
                scenario_type=incident.scenario_type,
                timestamp=TIMESTAMP,
                location=incident.location,
                report_index=event.source_event_index,
            )

    carmel_weather = generated_by_event[scenario.events[0]]
    golan_weather = generated_by_event[scenario.events[1]]
    carmel_satellite = generated_by_event[scenario.events[2]]
    golan_satellite = generated_by_event[scenario.events[3]]
    carmel_initial_news = generated_by_event[scenario.events[4]]
    golan_initial_news = generated_by_event[scenario.events[5]]
    carmel_follow_up_news = generated_by_event[scenario.events[10]]

    assert carmel_weather.stations[0].name.startswith("SIM-CARMEL-")
    assert golan_weather.stations[0].name.startswith("SIM-GOLAN-")
    assert carmel_satellite.hotspots
    assert golan_satellite.hotspots
    assert carmel_satellite.hotspots[0].latitude != golan_satellite.hotspots[0].latitude
    assert "/carmel/report-1/" in carmel_initial_news.reports[0].source_url
    assert "/golan/report-1/" in golan_initial_news.reports[0].source_url
    assert "/carmel/report-2/" in carmel_follow_up_news.reports[0].source_url
