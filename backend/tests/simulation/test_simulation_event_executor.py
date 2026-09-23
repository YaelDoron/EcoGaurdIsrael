"""Tests for SimulationEventExecutor."""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.simulation import (
    CARMEL_LOCATION,
    GOLAN_LOCATION,
    ScenarioType,
    SimulatedIncident,
    SimulationEvent,
    SimulationEventExecutionResult,
    SimulationEventExecutor,
    SimulationEventType,
    SimulationScenario,
    build_carmel_golan_active_fire_scenario,
    simulation_event_timestamp,
)
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator

TIMESTAMP = datetime(2026, 9, 12, 14, 0, 40, tzinfo=timezone.utc)
STARTED_AT = datetime(2026, 9, 12, 14, 0, 0, tzinfo=timezone.utc)
CARMEL_INCIDENT_ID = "incident-carmel-01"
GOLAN_INCIDENT_ID = "incident-golan-01"


@dataclass(frozen=True)
class FakeSaveResult:
    is_duplicate: bool
    observation: object | None = None
    hotspot: object | None = None
    report: object | None = None


class FakeWeatherRepository:
    def __init__(self, duplicate_observation_indexes=None, failing_observation_indexes=None) -> None:
        self.stations_by_id = {}
        self.station_save_calls = []
        self.observation_save_calls = []
        self._duplicate_observation_indexes = set(duplicate_observation_indexes or ())
        self._failing_observation_indexes = set(failing_observation_indexes or ())

    def save_station(self, station):
        self.station_save_calls.append(station)
        self.stations_by_id[station.external_station_id] = station
        return station

    def save_observation(self, observation):
        index = len(self.observation_save_calls)
        self.observation_save_calls.append(observation)
        if index in self._failing_observation_indexes:
            raise RuntimeError(f"weather observation failure {index}")
        return FakeSaveResult(
            observation=observation,
            is_duplicate=index in self._duplicate_observation_indexes,
        )


class FakeSatelliteRepository:
    def __init__(self, duplicate_indexes=None, failing_indexes=None) -> None:
        self.save_calls = []
        self._duplicate_indexes = set(duplicate_indexes or ())
        self._failing_indexes = set(failing_indexes or ())

    def save_hotspot(self, hotspot):
        index = len(self.save_calls)
        self.save_calls.append(hotspot)
        if index in self._failing_indexes:
            raise RuntimeError(f"satellite failure {index}")
        return FakeSaveResult(hotspot=hotspot, is_duplicate=index in self._duplicate_indexes)


class FakeNewsRepository:
    def __init__(self, duplicate_indexes=None, failing_indexes=None) -> None:
        self.save_calls = []
        self._duplicate_indexes = set(duplicate_indexes or ())
        self._failing_indexes = set(failing_indexes or ())

    def save_report(self, report):
        index = len(self.save_calls)
        self.save_calls.append(report)
        if index in self._failing_indexes:
            raise RuntimeError(f"news failure {index}")
        return FakeSaveResult(report=report, is_duplicate=index in self._duplicate_indexes)


class CapturingGenerator:
    def __init__(self, generated) -> None:
        self.generated = generated
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.generated


class RaisingGenerator:
    def generate(self, **kwargs):
        raise RuntimeError("generator exploded")


class FakeTextProcessor:
    """A trivial, deterministic stand-in for TextProcessor: uppercases text
    so translated output is unmistakably distinct from the original."""

    def translate_report(self, title, summary, location_name):
        return (
            f"EN: {title}",
            f"EN: {summary}",
            None if location_name is None else f"EN: {location_name}",
        )

    def translate_location_name(self, location_name):
        return f"EN: {location_name}"


def make_scenario(scenario_type=ScenarioType.ACTIVE_FIRE, location=CARMEL_LOCATION) -> SimulationScenario:
    incident = SimulatedIncident(
        incident_id=CARMEL_INCIDENT_ID,
        scenario_type=scenario_type,
        location=location,
    )
    return SimulationScenario(
        duration_seconds=120,
        seed=123,
        incidents=(incident,),
        events=(),
    )


def make_event(
    event_type: SimulationEventType,
    incident_id: str = CARMEL_INCIDENT_ID,
    source_event_index: int = 0,
) -> SimulationEvent:
    return SimulationEvent(
        offset_seconds=40,
        event_type=event_type,
        incident_id=incident_id,
        source_event_index=source_event_index,
    )


def make_weather_measurement(station_id: int) -> SimpleNamespace:
    station = WeatherStation(
        external_station_id=station_id,
        name=f"SIM-TEST-{station_id}",
        latitude=32.7,
        longitude=35.0,
        active=True,
    )
    observation = WeatherObservation(
        station_external_id=station_id,
        timestamp=TIMESTAMP,
        temperature=31.0,
    )
    return SimpleNamespace(station=station, observation=observation)


def make_hotspot(latitude=32.7, longitude=35.0) -> SatelliteHotspot:
    return SatelliteHotspot(
        latitude=latitude,
        longitude=longitude,
        detected_at=TIMESTAMP,
        satellite="SIM-NOAA-20",
        instrument="VIIRS",
    )


def test_weather_event_generates_and_saves_three_observations_with_correct_inputs():
    generated = SimpleNamespace(measurements=tuple(make_weather_measurement(station_id) for station_id in (1, 2, 3)))
    generator = CapturingGenerator(generated)
    repository = FakeWeatherRepository()
    scenario = make_scenario(location=GOLAN_LOCATION)
    event = make_event(SimulationEventType.WEATHER)
    executor = SimulationEventExecutor(
        weather_generator=generator,
        weather_repository=repository,
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(scenario=scenario, event=event, event_timestamp=TIMESTAMP)

    assert isinstance(result, SimulationEventExecutionResult)
    assert generator.calls == [
        {
            "scenario_type": ScenarioType.ACTIVE_FIRE,
            "timestamp": TIMESTAMP,
            "location": GOLAN_LOCATION,
        }
    ]
    assert len(repository.station_save_calls) == 3
    assert len(repository.stations_by_id) == 3
    assert len(repository.observation_save_calls) == 3
    assert result.success is True
    assert result.generated_count == 3
    assert result.saved_count == 3
    assert result.duplicates_skipped == 0
    assert result.failed_count == 0
    assert result.details["stations_processed"] == 3


def test_weather_event_counts_duplicate_observations_without_duplicate_station_records():
    generated = SimpleNamespace(measurements=tuple(make_weather_measurement(station_id) for station_id in (1, 2, 3)))
    repository = FakeWeatherRepository(duplicate_observation_indexes={0, 2})
    executor = SimulationEventExecutor(
        weather_generator=CapturingGenerator(generated),
        weather_repository=repository,
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.WEATHER),
        event_timestamp=TIMESTAMP,
    )

    assert len(repository.station_save_calls) == 3
    assert len(repository.stations_by_id) == 3
    assert result.success is True
    assert result.generated_count == 3
    assert result.saved_count == 1
    assert result.duplicates_skipped == 2
    assert result.failed_count == 0


def test_weather_event_reports_partial_persistence_failure():
    generated = SimpleNamespace(measurements=tuple(make_weather_measurement(station_id) for station_id in (1, 2, 3)))
    repository = FakeWeatherRepository(failing_observation_indexes={1})
    executor = SimulationEventExecutor(
        weather_generator=CapturingGenerator(generated),
        weather_repository=repository,
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.WEATHER),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is False
    assert result.generated_count == 3
    assert result.saved_count == 2
    assert result.duplicates_skipped == 0
    assert result.failed_count == 1
    assert "weather observation failure 1" in result.error_message


def test_satellite_event_saves_active_fire_hotspot_with_correct_inputs():
    generated = SimpleNamespace(hotspots=(make_hotspot(),))
    generator = CapturingGenerator(generated)
    repository = FakeSatelliteRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=generator,
        satellite_repository=repository,
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(location=GOLAN_LOCATION),
        event=make_event(SimulationEventType.SATELLITE),
        event_timestamp=TIMESTAMP,
    )

    assert generator.calls[0]["scenario_type"] is ScenarioType.ACTIVE_FIRE
    assert generator.calls[0]["location"] == GOLAN_LOCATION
    assert generator.calls[0]["timestamp"] == TIMESTAMP
    assert repository.save_calls == [generated.hotspots[0]]
    assert result.success is True
    assert result.generated_count == 1
    assert result.saved_count == 1
    assert result.duplicates_skipped == 0


def test_satellite_event_translates_hotspot_location_name_when_a_text_processor_is_injected():
    generated = SimpleNamespace(hotspots=(make_hotspot(),))
    hotspot_with_location = replace(generated.hotspots[0], location_name="הרי יהודה")
    generator = CapturingGenerator(SimpleNamespace(hotspots=(hotspot_with_location,)))
    repository = FakeSatelliteRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=generator,
        satellite_repository=repository,
        news_repository=FakeNewsRepository(),
        text_processor=FakeTextProcessor(),
    )

    executor.execute(
        scenario=make_scenario(location=GOLAN_LOCATION),
        event=make_event(SimulationEventType.SATELLITE),
        event_timestamp=TIMESTAMP,
    )

    assert repository.save_calls[0].location_name == "EN: הרי יהודה"


def test_satellite_event_without_a_text_processor_persists_the_hotspot_unchanged():
    generated = SimpleNamespace(hotspots=(replace(make_hotspot(), location_name="הרי יהודה"),))
    repository = FakeSatelliteRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=CapturingGenerator(generated),
        satellite_repository=repository,
        news_repository=FakeNewsRepository(),
    )

    executor.execute(
        scenario=make_scenario(location=GOLAN_LOCATION),
        event=make_event(SimulationEventType.SATELLITE),
        event_timestamp=TIMESTAMP,
    )

    assert repository.save_calls[0].location_name == "הרי יהודה"


def test_satellite_event_with_no_fire_empty_output_succeeds():
    generator = SatelliteDataGenerator(seed=123)
    repository = FakeSatelliteRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=generator,
        satellite_repository=repository,
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(scenario_type=ScenarioType.HIGH_RISK_NO_FIRE),
        event=make_event(SimulationEventType.SATELLITE),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is True
    assert result.generated_count == 0
    assert result.saved_count == 0
    assert result.duplicates_skipped == 0
    assert result.failed_count == 0
    assert repository.save_calls == []


def test_satellite_event_counts_duplicate_hotspot():
    generated = SimpleNamespace(hotspots=(make_hotspot(),))
    repository = FakeSatelliteRepository(duplicate_indexes={0})
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=CapturingGenerator(generated),
        satellite_repository=repository,
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.SATELLITE),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is True
    assert result.generated_count == 1
    assert result.saved_count == 0
    assert result.duplicates_skipped == 1
    assert result.failed_count == 0


def test_news_event_passes_source_event_index_as_report_index_and_saves_report():
    generator = NewsDataGenerator(seed=123)
    repository = FakeNewsRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=generator,
        news_repository=repository,
    )
    event = make_event(SimulationEventType.NEWS, source_event_index=1)

    result = executor.execute(
        scenario=make_scenario(location=CARMEL_LOCATION),
        event=event,
        event_timestamp=TIMESTAMP,
    )

    assert result.success is True
    assert result.generated_count == 1
    assert result.saved_count == 1
    assert result.duplicates_skipped == 0
    assert result.details["report_index"] == 1
    assert "/carmel/report-2/" in repository.save_calls[0].source_url


def test_news_event_translates_title_summary_and_location_when_a_text_processor_is_injected():
    generator = NewsDataGenerator(seed=123)
    repository = FakeNewsRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=generator,
        news_repository=repository,
        text_processor=FakeTextProcessor(),
    )

    executor.execute(
        scenario=make_scenario(location=CARMEL_LOCATION),
        event=make_event(SimulationEventType.NEWS, source_event_index=1),
        event_timestamp=TIMESTAMP,
    )

    saved = repository.save_calls[0]
    assert saved.title.startswith("EN: ")
    assert saved.summary.startswith("EN: ")
    assert saved.location_name.startswith("EN: ")


def test_news_event_without_a_text_processor_persists_the_report_unchanged():
    generator = NewsDataGenerator(seed=123)
    repository = FakeNewsRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=generator,
        news_repository=repository,
    )

    executor.execute(
        scenario=make_scenario(location=CARMEL_LOCATION),
        event=make_event(SimulationEventType.NEWS, source_event_index=1),
        event_timestamp=TIMESTAMP,
    )

    saved = repository.save_calls[0]
    assert not saved.title.startswith("EN: ")


class FallbackTextProcessor:
    """A TextProcessor stand-in that always degrades to its fail-safe
    fallback (returns text unchanged), simulating an LLM failure/rate limit
    without needing a real network call - used to verify the executor logs
    a visible warning whenever Hebrew survives a translation attempt."""

    def translate_report(self, title, summary, location_name):
        return (title, summary, location_name)

    def translate_location_name(self, location_name):
        return location_name


def test_satellite_event_warns_when_translation_falls_back_and_hebrew_survives(caplog):
    generated = SimpleNamespace(hotspots=(replace(make_hotspot(), location_name="הרי יהודה"),))
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=CapturingGenerator(generated),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
        text_processor=FallbackTextProcessor(),
    )

    with caplog.at_level(logging.WARNING):
        executor.execute(
            scenario=make_scenario(location=GOLAN_LOCATION),
            event=make_event(SimulationEventType.SATELLITE),
            event_timestamp=TIMESTAMP,
        )

    assert any("untranslated Hebrew" in message for message in caplog.messages)


class SuccessfulTextProcessor:
    """A TextProcessor stand-in that returns pure, Hebrew-free English -
    unlike FakeTextProcessor (which just prefixes "EN: " onto the original
    text and so still contains Hebrew), this represents what a genuinely
    successful translation looks like."""

    def translate_report(self, title, summary, location_name):
        return ("Fire update", "Smoke seen nearby.", None if location_name is None else "Judean Hills")

    def translate_location_name(self, location_name):
        return "Judean Hills"


def test_satellite_event_does_not_warn_when_translation_succeeds(caplog):
    generated = SimpleNamespace(hotspots=(replace(make_hotspot(), location_name="הרי יהודה"),))
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=CapturingGenerator(generated),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
        text_processor=SuccessfulTextProcessor(),
    )

    with caplog.at_level(logging.WARNING):
        executor.execute(
            scenario=make_scenario(location=GOLAN_LOCATION),
            event=make_event(SimulationEventType.SATELLITE),
            event_timestamp=TIMESTAMP,
        )

    assert not any("untranslated Hebrew" in message for message in caplog.messages)


def test_satellite_event_warns_when_no_text_processor_is_configured_at_all(caplog):
    generated = SimpleNamespace(hotspots=(replace(make_hotspot(), location_name="הרי יהודה"),))
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_generator=CapturingGenerator(generated),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    with caplog.at_level(logging.WARNING):
        executor.execute(
            scenario=make_scenario(location=GOLAN_LOCATION),
            event=make_event(SimulationEventType.SATELLITE),
            event_timestamp=TIMESTAMP,
        )

    assert any("not configured" in message for message in caplog.messages)


def test_news_event_warns_when_translation_falls_back_and_hebrew_survives(caplog):
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=NewsDataGenerator(seed=123),
        news_repository=FakeNewsRepository(),
        text_processor=FallbackTextProcessor(),
    )

    with caplog.at_level(logging.WARNING):
        executor.execute(
            scenario=make_scenario(location=CARMEL_LOCATION),
            event=make_event(SimulationEventType.NEWS, source_event_index=1),
            event_timestamp=TIMESTAMP,
        )

    assert any("untranslated Hebrew" in message for message in caplog.messages)


def test_news_event_counts_duplicate_url():
    repository = FakeNewsRepository(duplicate_indexes={0})
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=NewsDataGenerator(seed=123),
        news_repository=repository,
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.NEWS),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is True
    assert result.generated_count == 1
    assert result.saved_count == 0
    assert result.duplicates_skipped == 1
    assert result.failed_count == 0


def test_carmel_golan_news_zero_indexes_are_independent_and_location_aware():
    scenario = build_carmel_golan_active_fire_scenario(seed=123)
    news_events = [event for event in scenario.events if event.event_type is SimulationEventType.NEWS]
    repository = FakeNewsRepository()
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=NewsDataGenerator(seed=scenario.seed),
        news_repository=repository,
    )

    carmel_result = executor.execute(scenario, news_events[0], TIMESTAMP)
    golan_result = executor.execute(scenario, news_events[1], TIMESTAMP)

    assert carmel_result.success is True
    assert golan_result.success is True
    assert news_events[0].source_event_index == 0
    assert news_events[1].source_event_index == 0
    assert "/carmel/report-1/" in repository.save_calls[0].source_url
    assert "/golan/report-1/" in repository.save_calls[1].source_url
    assert repository.save_calls[0].latitude != repository.save_calls[1].latitude


def test_weather_events_for_different_incidents_generate_distinct_station_networks():
    scenario = build_carmel_golan_active_fire_scenario(seed=123)
    weather_events = [event for event in scenario.events if event.event_type is SimulationEventType.WEATHER]
    repository = FakeWeatherRepository()
    executor = SimulationEventExecutor(
        weather_generator=WeatherDataGenerator(seed=scenario.seed),
        weather_repository=repository,
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    executor.execute(scenario, weather_events[0], TIMESTAMP)
    executor.execute(scenario, weather_events[1], TIMESTAMP)

    station_names = [station.name for station in repository.station_save_calls]
    assert any(name.startswith("SIM-CARMEL-") for name in station_names)
    assert any(name.startswith("SIM-GOLAN-") for name in station_names)
    assert len(repository.stations_by_id) == 6


def test_generator_exception_returns_failed_result():
    executor = SimulationEventExecutor(
        weather_generator=RaisingGenerator(),
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.WEATHER),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is False
    assert result.generated_count == 0
    assert result.saved_count == 0
    assert result.failed_count == 1
    assert result.error_message == "generator exploded"


def test_news_repository_exception_returns_failed_result():
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_generator=NewsDataGenerator(seed=123),
        news_repository=FakeNewsRepository(failing_indexes={0}),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.NEWS),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is False
    assert result.generated_count == 1
    assert result.saved_count == 0
    assert result.failed_count == 1
    assert "news failure 0" in result.error_message


def test_unknown_event_type_returns_failed_result_if_represented():
    event = object.__new__(SimulationEvent)
    object.__setattr__(event, "offset_seconds", 10)
    object.__setattr__(event, "event_type", "unknown")
    object.__setattr__(event, "incident_id", CARMEL_INCIDENT_ID)
    object.__setattr__(event, "source_event_index", 0)
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=event,
        event_timestamp=TIMESTAMP,
    )

    assert result.success is False
    assert result.failed_count == 1
    assert "Unsupported simulation event type" in result.error_message


def test_orphan_event_returns_failed_result():
    executor = SimulationEventExecutor(
        weather_repository=FakeWeatherRepository(),
        satellite_repository=FakeSatelliteRepository(),
        news_repository=FakeNewsRepository(),
    )

    result = executor.execute(
        scenario=make_scenario(),
        event=make_event(SimulationEventType.NEWS, incident_id=GOLAN_INCIDENT_ID),
        event_timestamp=TIMESTAMP,
    )

    assert result.success is False
    assert result.failed_count == 1
    assert "Unknown incident_id" in result.error_message


def test_simulation_event_timestamp_uses_event_offset_and_preserves_timezone():
    event = make_event(SimulationEventType.NEWS)

    assert simulation_event_timestamp(STARTED_AT, event) == TIMESTAMP
    assert simulation_event_timestamp(STARTED_AT, event).tzinfo is timezone.utc
