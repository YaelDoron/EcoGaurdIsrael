"""A dedicated, deterministic multi-hour scenario for the final AI Hybrid V5 acceptance (Task 9C).

The standard demo presets are compressed into <= 900 simulated seconds, so every hotspot they produce falls inside ONE satellite
pass: they can never show the multi-pass history the V5 model reasons about. This scenario keeps the SAME production pipeline
(SimulationEventExecutor -> execute_simulation_event -> real detection / severity / spread / targets / routing / global planning)
and only replaces the EVIDENCE SOURCE of the satellite and news events with a fixed, plausible, hand-written observation plan
spread over several simulated hours. Weather still comes from the normal seeded weather generator.

Nothing here touches the model, its thresholds or the V5 generator: the plan is ordinary evidence (positions, FRP, brightness,
confidence class, news strength). Whether the approved model finds it increasingly convincing is decided by the model.

The plan describes ONE growing overnight wildfire near a location: each satellite pass sees more pixels, brighter and more
energetic, with the fire front advancing roughly a kilometre between passes - the opposite of a steady industrial heat source.
Consecutive passes are <= 6 h apart so they stay ONE FireEvent (ACTIVE_EVENT_MATCH_WINDOW_HOURS = 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random
from typing import Any

from src.models.fire_report import WildfireReport
from src.models.news_wildfire_signal_strength import NewsWildfireSignalStrength
from src.models.satellite_hotspot import SatelliteHotspot
from src.simulation.simulation_event import SimulationEvent, SimulationEventType
from src.simulation.simulation_event_executor import SimulationEventExecutionResult, SimulationEventExecutor
from src.simulation.simulation_location import SimulationLocation
from src.simulation.simulation_locations import CARMEL_LOCATION
from src.simulation.simulation_scenario import SimulationScenario, build_active_fire_scenario

ACCEPTANCE_SATELLITE = "NOAA-20"
ACCEPTANCE_INSTRUMENT = "VIIRS"
ACCEPTANCE_NEWS_FEED = "ai-acceptance-simulation"
_METERS_PER_DEGREE_LATITUDE = 111_195.0
_NEWS_DELAY_SECONDS = 10 * 60  # a report appears some minutes after the pass (same 60 min candidate)


@dataclass(frozen=True)
class AcceptanceHotspot:
    """One thermal pixel, positioned relative to the location (metres) with its measured values."""

    north_m: float
    east_m: float
    frp: float
    confidence: str  # "l" | "n" | "h"
    brightness: float


@dataclass(frozen=True)
class AcceptanceStep:
    """One simulated observation time: what the satellite pass and the news feed contribute."""

    label: str
    offset_seconds: int
    hotspots: tuple[AcceptanceHotspot, ...]
    news: tuple[NewsWildfireSignalStrength, ...] = ()

    @property
    def offset_hours(self) -> float:
        return self.offset_seconds / 3600.0


@dataclass(frozen=True)
class AIAcceptanceScenario:
    """Holder scenario (incident + weather source) + the evidence plan + its ordered SimulationEvents."""

    name: str
    location: SimulationLocation
    steps: tuple[AcceptanceStep, ...]
    holder_scenario: SimulationScenario

    @property
    def incident_id(self) -> str:
        return self.holder_scenario.incidents[0].incident_id

    def events_for_step(self, step_index: int) -> tuple[SimulationEvent, ...]:
        """WEATHER, then the SATELLITE pass, then (if the step has news) the NEWS report - like a normal timeline."""
        step = self.steps[step_index]
        events = [
            SimulationEvent(step.offset_seconds, SimulationEventType.WEATHER, self.incident_id, step_index),
            SimulationEvent(step.offset_seconds + 1, SimulationEventType.SATELLITE, self.incident_id, step_index),
        ]
        if step.news:
            events.append(
                SimulationEvent(step.offset_seconds + _NEWS_DELAY_SECONDS, SimulationEventType.NEWS, self.incident_id, step_index)
            )
        return tuple(events)


def _pixels(count: int, centre_north_m: float, centre_east_m: float, frp: float, confidence: tuple[str, ...],
            brightness: float, spread_m: float, seed: int) -> tuple[AcceptanceHotspot, ...]:
    """A fixed, seeded scatter of `count` pixels around a centre (deterministic; plain evidence, not model input)."""
    rng = random.Random(f"ai-acceptance-pixels:{seed}")
    return tuple(
        AcceptanceHotspot(
            north_m=round(centre_north_m + rng.uniform(-spread_m, spread_m), 1),
            east_m=round(centre_east_m + rng.uniform(-spread_m, spread_m), 1),
            frp=round(frp * rng.uniform(0.6, 1.4), 1),
            confidence=confidence[index % len(confidence)],
            brightness=round(brightness + rng.uniform(-4.0, 6.0), 1),
        )
        for index in range(count)
    )


def build_ai_progression_scenario(location: SimulationLocation = CARMEL_LOCATION) -> AIAcceptanceScenario:
    """An overnight fire seen by four NOAA-20 passes, 4 h apart, starting at 20:00 local time."""
    strength = NewsWildfireSignalStrength
    steps = (
        # T+0h  one faint pixel, no report yet                          -> a lone hotspot: ambiguous, worth watching
        AcceptanceStep("T+0h first faint pixel", 0, _pixels(1, 0, 0, 6, ("n",), 325, 0, 3)),
        # T+4h  the front has advanced ~0.9 km: three pixels, weak report -> more fire-like, still not enough
        AcceptanceStep("T+4h three pixels", 4 * 3600, _pixels(3, 700, 500, 10, ("n", "h"), 331, 300, 103), (strength.WEAK,)),
        # T+8h  five pixels, brighter, a moderate report                 -> strong multi-pass evidence
        AcceptanceStep("T+8h five pixels", 8 * 3600, _pixels(5, 1000, 800, 16, ("n", "h"), 334, 600, 203), (strength.MODERATE,)),
        # T+12h seven bright pixels, a strong report                     -> an established wildfire
        AcceptanceStep("T+12h seven bright pixels", 12 * 3600, _pixels(7, 1300, 1000, 26, ("h",), 339, 800, 303), (strength.STRONG,)),
    )
    holder = build_active_fire_scenario(location=location, seed=42)
    return AIAcceptanceScenario(name="ai_progression_overnight_fire", location=location, steps=steps, holder_scenario=holder)


class AIAcceptanceEvidenceExecutor(SimulationEventExecutor):
    """The normal executor, except SATELLITE / NEWS events persist the scenario's fixed observation plan instead of random data."""

    def __init__(self, scenario: AIAcceptanceScenario, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._plan = scenario

    def _execute_satellite_event(self, scenario, event, event_timestamp, incident) -> SimulationEventExecutionResult:
        step = self._plan.steps[event.source_event_index]
        latitude0, longitude0 = self._plan.location.latitude, self._plan.location.longitude
        longitude_scale = _METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude0))
        saved = duplicates = 0
        for pixel in step.hotspots:
            result = self._satellite_repository.save_hotspot(
                SatelliteHotspot(
                    latitude=round(latitude0 + pixel.north_m / _METERS_PER_DEGREE_LATITUDE, 6),
                    longitude=round(longitude0 + pixel.east_m / longitude_scale, 6),
                    detected_at=event_timestamp,  # every pixel of one overpass shares the pass time
                    confidence=pixel.confidence,
                    frp=pixel.frp,
                    brightness=pixel.brightness,
                    satellite=ACCEPTANCE_SATELLITE,
                    instrument=ACCEPTANCE_INSTRUMENT,
                    day_night=_day_night(event_timestamp),
                    location_name=self._plan.location.name,
                )
            )
            duplicates += 1 if result.is_duplicate else 0
            saved += 0 if result.is_duplicate else 1
        return SimulationEventExecutionResult(
            event=event, success=True, generated_count=len(step.hotspots), saved_count=saved, duplicates_skipped=duplicates,
            failed_count=0, details={"hotspots_generated": len(step.hotspots)},
        )

    def _execute_news_event(self, scenario, event, event_timestamp, incident) -> SimulationEventExecutionResult:
        step = self._plan.steps[event.source_event_index]
        saved = duplicates = 0
        for index, strength in enumerate(step.news):
            result = self._news_repository.save_report(
                WildfireReport(
                    source_url=f"https://ai-acceptance.example/{self._plan.name}/{event.source_event_index}/{index}",
                    source_feed=ACCEPTANCE_NEWS_FEED,
                    title="Wildfire reported",
                    summary="Smoke and flames reported near the location.",
                    location_name=None,
                    latitude=self._plan.location.latitude,
                    longitude=self._plan.location.longitude,
                    published_at=event_timestamp,
                    fetched_at=event_timestamp,
                    wildfire_signal_strength=strength,
                )
            )
            duplicates += 1 if result.is_duplicate else 0
            saved += 0 if result.is_duplicate else 1
        return SimulationEventExecutionResult(
            event=event, success=True, generated_count=len(step.news), saved_count=saved, duplicates_skipped=duplicates,
            failed_count=0, details={"reports_generated": len(step.news), "report_index": event.source_event_index},
        )


@dataclass(frozen=True)
class AIProgressionRecord:
    """One row of the acceptance report: what the AI saw and decided for one candidate at one simulated step."""

    label: str
    timestamp: datetime
    candidate_summary: str
    event_id: int | None
    probability: float
    policy_status: str
    event_status: str | None
    satellite_pass_count: int
    history_span_minutes: float | None
    current_satellite_pixel_count: int
    peak_confidence: float | None
    latest_assessment_probability: float | None
    response_eligible: bool


def summarize_step(step: AcceptanceStep) -> str:
    news = ", ".join(strength.value for strength in step.news) or "no news"
    classes = "/".join(sorted({pixel.confidence for pixel in step.hotspots}))
    return f"{len(step.hotspots)} pixel(s) [{classes}], news: {news}"


def record_progression(step, timestamp, detection_result, fire_event_repository, history_service) -> tuple[AIProgressionRecord, ...]:
    """Summarise one detection cycle (candidate summary, AI result, persisted event state) - read-only, no 25-feature dump."""
    from src.models.fire_event_response_eligibility import is_response_eligible
    from src.models.fire_detection_event_history import satellite_pass_time_span_minutes
    from src.models.fire_detection_satellite_pass import group_satellite_passes

    records = []
    for assessment in detection_result.candidate_assessments:
        event_status = peak = latest = span = None
        eligible = False
        if assessment.event_id is not None:
            stored = fire_event_repository.get_by_id(assessment.event_id)
            event_status, peak = stored.event.status.value, stored.event.detection_confidence
            eligible = is_response_eligible(stored.event.status)
            row = fire_event_repository.get_ml_assessment(assessment.event_id)
            latest = row.ml_probability if row is not None else None
            history = history_service.build_history(stored, timestamp + timedelta(minutes=5))
            span = satellite_pass_time_span_minutes(group_satellite_passes(history.satellite_evidence, history.satellite_pass_gap_minutes))
        records.append(
            AIProgressionRecord(
                label=step.label, timestamp=timestamp, candidate_summary=summarize_step(step), event_id=assessment.event_id,
                probability=assessment.ml_probability, policy_status=assessment.final_status.value, event_status=event_status,
                satellite_pass_count=assessment.ai_satellite_pass_count, history_span_minutes=span,
                current_satellite_pixel_count=assessment.ai_current_satellite_pixel_count, peak_confidence=peak,
                latest_assessment_probability=latest, response_eligible=eligible,
            )
        )
    return tuple(records)


def _day_night(moment: datetime) -> str:
    """Israel standard time (UTC+2): night from 19:00 to 05:59 - the same convention as the simulation generators."""
    local_hour = (moment.hour + 2) % 24
    return "N" if local_hour >= 19 or local_hour < 6 else "D"


__all__ = [
    "AIProgressionRecord", "record_progression", "summarize_step",
    "AIAcceptanceEvidenceExecutor", "AIAcceptanceScenario", "AcceptanceHotspot", "AcceptanceStep", "build_ai_progression_scenario",
]
