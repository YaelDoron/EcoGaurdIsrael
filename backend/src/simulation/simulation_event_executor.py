"""Execute simulation events by generating and persisting domain objects."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from src.external.news.news_client import TextProcessor
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

# TextProcessor's translation methods never raise and never report whether
# they actually translated or fell back (see news_client.py's fail-safe
# contract) - this is the executor's own independent signal that a
# just-generated Hebrew name is about to be persisted untranslated, purely
# from observing the output text itself, so a developer watching simulation
# logs can see exactly which entity ended up with raw source text even if
# they missed the corresponding TRANSLATION FALLBACK TRIGGERED warning in
# news_client.py.
_HEBREW_CHARACTERS = re.compile(r"[֐-׿]")


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
        text_processor: TextProcessor | None = None,
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
        # `None` (the default) means "translate nothing" - simulated Hebrew
        # text/location names are persisted exactly as generated, unchanged
        # from before this was added. Every existing caller/test that does
        # not pass one keeps that exact prior behavior. A real TextProcessor
        # is only ever injected by production wiring (see
        # demo_simulation_runner.py), never constructed here - this class
        # must never require an LLM API key just to run a demo scenario.
        self._text_processor = text_processor

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
            elif event.event_type is SimulationEventType.RESOURCE_STATUS:
                result = self._execute_resource_status_event(event)
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

    def _translate_hotspots(self, hotspots: tuple[Any, ...]) -> tuple[Any, ...]:
        """Translate each hotspot's Hebrew location_name to English before it
        is ever persisted - a no-op (returns hotspots unchanged) when no
        text_processor was injected, or for a hotspot with no location_name
        at all (real FIRMS ingestion never sets one; nothing to translate)."""
        if self._text_processor is None:
            if hotspots:
                logger.warning(
                    "Simulation translation is not configured (no TextProcessor injected) - "
                    "%d satellite hotspot(s) will be persisted with their original, untranslated location_name.",
                    len(hotspots),
                )
            return hotspots
        translated = []
        for hotspot in hotspots:
            if hotspot.location_name is None:
                translated.append(hotspot)
                continue
            translated_name = self._text_processor.translate_location_name(hotspot.location_name)
            _warn_if_still_untranslated("satellite hotspot location_name", translated_name)
            translated.append(replace(hotspot, location_name=translated_name))
        return tuple(translated)

    def _translate_reports(self, reports: tuple[Any, ...]) -> tuple[Any, ...]:
        """Translate each report's title/summary/location_name to English
        before it is ever persisted - the same universal backend LLM
        translation NewsMonitoringAgent applies to real RSS ingestion,
        reused here for simulated news (a no-op when no text_processor was
        injected)."""
        if self._text_processor is None:
            if reports:
                logger.warning(
                    "Simulation translation is not configured (no TextProcessor injected) - "
                    "%d news report(s) will be persisted with their original, untranslated text.",
                    len(reports),
                )
            return reports
        translated = []
        for report in reports:
            title, summary, location_name = self._text_processor.translate_report(
                report.title, report.summary, report.location_name
            )
            _warn_if_still_untranslated("news report title", title)
            _warn_if_still_untranslated("news report location_name", location_name)
            translated.append(replace(report, title=title, summary=summary, location_name=location_name))
        return tuple(translated)

    @staticmethod
    def _execute_resource_status_event(event: SimulationEvent) -> SimulationEventExecutionResult:
        """Acknowledge a resource-status timeline event without direct persistence."""
        return SimulationEventExecutionResult(
            event=event,
            success=True,
            generated_count=1,
            saved_count=1,
            duplicates_skipped=0,
            failed_count=0,
            details={"resource_status_events": 1},
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
        hotspots = tuple(self._translate_hotspots(generated.hotspots))
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
        reports = tuple(self._translate_reports(generated.reports))
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


def _warn_if_still_untranslated(field_description: str, value: str | None) -> None:
    """Log a visible warning when a field that just went through
    TextProcessor still contains Hebrew script - TextProcessor's methods
    never raise and never report success/failure directly (by design, see
    news_client.py), so this is the executor's own signal, from the output
    text alone, that the LLM call for this specific field fell back to its
    original, untranslated value."""
    if value is not None and _HEBREW_CHARACTERS.search(value):
        logger.warning(
            "Simulation %s still contains untranslated Hebrew text after a translation attempt - "
            "the LLM translation likely failed or degraded; see the corresponding 'TRANSLATION FALLBACK "
            "TRIGGERED' warning from news_client.py for the exact reason: %r",
            field_description,
            value,
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
