"""Execute simulation events by generating and persisting domain objects."""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.simulation.generators.news_data_generator import NewsDataGenerator
from src.simulation.generators.satellite_data_generator import SatelliteDataGenerator
from src.simulation.generators.weather_data_generator import WeatherDataGenerator
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_scenario import SimulationScenario

logger = logging.getLogger(__name__)

GeneratorFactory = Callable[[int], Any]


@dataclass(frozen=True)
class SimulationEventExecutionResult:
    """Generic result for one executed simulation event."""

    event: SimulationEvent
    success: bool
    generated_count: int
    saved_count: int
    duplicates_skipped: int
    failed_count: int
    error_message: str | None = None
    details: Mapping[str, int] = field(default_factory=dict)


class SimulationEventExecutor:
    """Generate and persist domain objects for simulation events."""

    def __init__(
        self,
        weather_repository: WeatherRepository | None = None,
        satellite_repository: SatelliteHotspotRepository | None = None,
        news_repository: NewsRepository | None = None,
        weather_generator: WeatherDataGenerator | None = None,
        satellite_generator: SatelliteDataGenerator | None = None,
        news_generator: NewsDataGenerator | None = None,
        weather_generator_factory: GeneratorFactory = WeatherDataGenerator,
        satellite_generator_factory: GeneratorFactory = SatelliteDataGenerator,
        news_generator_factory: GeneratorFactory = NewsDataGenerator,
    ) -> None:
        self._weather_repository = weather_repository or WeatherRepository()
        self._satellite_repository = satellite_repository or SatelliteHotspotRepository()
        self._news_repository = news_repository or NewsRepository()
        self._weather_generator = weather_generator
        self._satellite_generator = satellite_generator
        self._news_generator = news_generator
        self._weather_generator_factory = weather_generator_factory
        self._satellite_generator_factory = satellite_generator_factory
        self._news_generator_factory = news_generator_factory

    def execute(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        event_timestamp: datetime,
    ) -> SimulationEventExecutionResult:
        """Execute one simulation event at the caller-supplied timestamp."""
        if not isinstance(scenario, SimulationScenario):
            raise ValueError(f"scenario must be a SimulationScenario, got {scenario!r}")
        if not isinstance(event, SimulationEvent):
            raise ValueError(f"event must be a SimulationEvent, got {event!r}")

        logger.info(
            "Simulation event execution started: incident=%s event_type=%s offset=%s",
            event.incident_id,
            getattr(event.event_type, "value", event.event_type),
            event.offset_seconds,
        )

        if not isinstance(event_timestamp, datetime):
            return self._failed_result(event, f"event_timestamp must be a datetime, got {event_timestamp!r}")

        try:
            incident = scenario.get_incident(event.incident_id)
            if event.event_type is SimulationEventType.WEATHER:
                result = self._execute_weather_event(scenario, event, event_timestamp, incident)
            elif event.event_type is SimulationEventType.SATELLITE:
                result = self._execute_satellite_event(scenario, event, event_timestamp, incident)
            elif event.event_type is SimulationEventType.NEWS:
                result = self._execute_news_event(scenario, event, event_timestamp, incident)
            else:
                result = self._failed_result(event, f"Unsupported simulation event type: {event.event_type!r}")
        except Exception as exc:  # noqa: BLE001 - one event failure should become a structured result.
            logger.exception(
                "Simulation event execution failed: incident=%s event_type=%s",
                event.incident_id,
                getattr(event.event_type, "value", event.event_type),
            )
            return self._failed_result(event, str(exc))

        logger.info(
            "Simulation event execution finished: incident=%s event_type=%s generated=%s saved=%s "
            "duplicates=%s failed=%s success=%s",
            event.incident_id,
            getattr(event.event_type, "value", event.event_type),
            result.generated_count,
            result.saved_count,
            result.duplicates_skipped,
            result.failed_count,
            result.success,
        )
        return result

    def _execute_weather_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        event_timestamp: datetime,
        incident: Any,
    ) -> SimulationEventExecutionResult:
        generated = self._get_weather_generator(scenario.seed).generate(
            scenario_type=incident.scenario_type,
            timestamp=event_timestamp,
            location=incident.location,
        )
        measurements = tuple(generated.measurements)
        saved_count = 0
        duplicates_skipped = 0
        failed_count = 0
        error_messages: list[str] = []
        stations_processed = 0

        for measurement in measurements:
            try:
                self._weather_repository.save_station(measurement.station)
                stations_processed += 1
                save_result = self._weather_repository.save_observation(measurement.observation)
                if save_result.is_duplicate:
                    duplicates_skipped += 1
                else:
                    saved_count += 1
            except Exception as exc:  # noqa: BLE001 - continue other generated measurements.
                failed_count += 1
                error_messages.append(str(exc))
                logger.exception(
                    "Weather simulation measurement failed: incident=%s station=%s",
                    event.incident_id,
                    getattr(measurement.station, "external_station_id", None),
                )

        return SimulationEventExecutionResult(
            event=event,
            success=failed_count == 0,
            generated_count=len(measurements),
            saved_count=saved_count,
            duplicates_skipped=duplicates_skipped,
            failed_count=failed_count,
            error_message=_join_error_messages(error_messages),
            details={
                "stations_processed": stations_processed,
                "observations_generated": len(measurements),
            },
        )

    def _execute_satellite_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        event_timestamp: datetime,
        incident: Any,
    ) -> SimulationEventExecutionResult:
        generated = self._get_satellite_generator(scenario.seed).generate(
            scenario_type=incident.scenario_type,
            timestamp=event_timestamp,
            location=incident.location,
        )
        hotspots = tuple(generated.hotspots)
        saved_count = 0
        duplicates_skipped = 0
        failed_count = 0
        error_messages: list[str] = []

        for hotspot in hotspots:
            try:
                save_result = self._satellite_repository.save_hotspot(hotspot)
                if save_result.is_duplicate:
                    duplicates_skipped += 1
                else:
                    saved_count += 1
            except Exception as exc:  # noqa: BLE001 - continue other generated hotspots.
                failed_count += 1
                error_messages.append(str(exc))
                logger.exception("Satellite simulation hotspot failed: incident=%s", event.incident_id)

        return SimulationEventExecutionResult(
            event=event,
            success=failed_count == 0,
            generated_count=len(hotspots),
            saved_count=saved_count,
            duplicates_skipped=duplicates_skipped,
            failed_count=failed_count,
            error_message=_join_error_messages(error_messages),
            details={"hotspots_generated": len(hotspots)},
        )

    def _execute_news_event(
        self,
        scenario: SimulationScenario,
        event: SimulationEvent,
        event_timestamp: datetime,
        incident: Any,
    ) -> SimulationEventExecutionResult:
        generated = self._get_news_generator(scenario.seed).generate(
            scenario_type=incident.scenario_type,
            timestamp=event_timestamp,
            location=incident.location,
            report_index=event.source_event_index,
        )
        reports = tuple(generated.reports)
        saved_count = 0
        duplicates_skipped = 0
        failed_count = 0
        error_messages: list[str] = []

        for report in reports:
            try:
                save_result = self._news_repository.save_report(report)
                if save_result.is_duplicate:
                    duplicates_skipped += 1
                else:
                    saved_count += 1
            except Exception as exc:  # noqa: BLE001 - continue other generated reports.
                failed_count += 1
                error_messages.append(str(exc))
                logger.exception("News simulation report failed: incident=%s", event.incident_id)

        return SimulationEventExecutionResult(
            event=event,
            success=failed_count == 0,
            generated_count=len(reports),
            saved_count=saved_count,
            duplicates_skipped=duplicates_skipped,
            failed_count=failed_count,
            error_message=_join_error_messages(error_messages),
            details={
                "reports_generated": len(reports),
                "report_index": event.source_event_index,
            },
        )

    def _get_weather_generator(self, seed: int) -> Any:
        return self._weather_generator or self._weather_generator_factory(seed)

    def _get_satellite_generator(self, seed: int) -> Any:
        return self._satellite_generator or self._satellite_generator_factory(seed)

    def _get_news_generator(self, seed: int) -> Any:
        return self._news_generator or self._news_generator_factory(seed)

    @staticmethod
    def _failed_result(event: SimulationEvent, error_message: str) -> SimulationEventExecutionResult:
        return SimulationEventExecutionResult(
            event=event,
            success=False,
            generated_count=0,
            saved_count=0,
            duplicates_skipped=0,
            failed_count=1,
            error_message=error_message,
        )


def simulation_event_timestamp(scenario_started_at: datetime, event: SimulationEvent) -> datetime:
    """Calculate the timestamp for an event offset from a scenario start time."""
    if not isinstance(scenario_started_at, datetime):
        raise ValueError(f"scenario_started_at must be a datetime, got {scenario_started_at!r}")
    if not isinstance(event, SimulationEvent):
        raise ValueError(f"event must be a SimulationEvent, got {event!r}")
    return scenario_started_at + timedelta(seconds=event.offset_seconds)


def _join_error_messages(error_messages: list[str]) -> str | None:
    if not error_messages:
        return None
    return "; ".join(message or "Unknown error" for message in error_messages)
