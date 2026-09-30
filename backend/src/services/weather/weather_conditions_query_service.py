"""Shared read-only "Weather Conditions" projection.

A Weather Conditions entry is the mean of the exact weather observations a
persisted, successfully calculated (VALID) Fire Danger assessment already
traced for its area. This is the projection the Operations Overview
Activity Feed has always shown; it lives here so the Operations Overview
and ChatbotAgent share one implementation instead of two.

Read-only: no FFWI recalculation, no fresh observation selection, no
external weather call - only repository reads of already-persisted rows,
and never a write. Wind direction and rainfall are deliberately absent: no
area-level aggregate of either exists anywhere in EcoGuard.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.weather_repository import WeatherRepository


@dataclass(frozen=True)
class AreaWeatherConditions:
    """One area's weather conditions, as averaged over the stations its
    Fire Danger assessment traced. Carries no internal database ids.

    `observed_at` is the latest contributing observation's own timestamp;
    `assessed_at` is the Fire Danger assessment's; `station_count` is the
    number of contributing observations (one per traced station).
    """

    area_name: str
    observed_at: datetime
    assessed_at: datetime
    station_count: int
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    wind_gust_kmh: float | None


@dataclass(frozen=True)
class AssessedWeatherConditions:
    """AreaWeatherConditions paired with the stored assessment it was
    projected from - for callers (Operations Overview) that need the
    assessment's own identity/timestamps alongside the conditions."""

    stored_assessment: StoredFireDangerAssessment
    conditions: AreaWeatherConditions


class WeatherConditionsQueryService:
    """Project persisted Fire Danger assessments' traced observations into Weather Conditions."""

    def __init__(
        self,
        *,
        fire_danger_assessment_repository: FireDangerAssessmentRepository | None = None,
        weather_repository: WeatherRepository | None = None,
    ) -> None:
        self._fire_danger_assessment_repository = (
            fire_danger_assessment_repository or FireDangerAssessmentRepository()
        )
        self._weather_repository = weather_repository or WeatherRepository()

    def summarize_assessments(
        self, stored_assessments: Iterable[StoredFireDangerAssessment]
    ) -> tuple[AssessedWeatherConditions, ...]:
        """Weather Conditions for each given assessment, in the given order.

        `stored_assessments` must carry their `observation_ids` (e.g. from
        get_recent_with_level_in or get_by_id). An assessment is skipped -
        never fabricated - when none of its traced observations can be
        loaded, or when no contributing observation has a temperature,
        humidity, or wind speed value. Each field is the mean of the
        contributing observations that have that value; `wind_gust_kmh` is
        None when none of them reports a gust.
        """
        stored_assessments = tuple(stored_assessments)
        if not stored_assessments:
            return ()

        all_observation_ids = tuple(
            {observation_id for stored in stored_assessments for observation_id in stored.observation_ids}
        )
        observations_by_id = {
            stored_observation.observation_id: stored_observation
            for stored_observation in self._weather_repository.get_observations_by_ids(all_observation_ids)
        }

        results = []
        for stored in stored_assessments:
            contributing = [
                observations_by_id[observation_id]
                for observation_id in stored.observation_ids
                if observation_id in observations_by_id
            ]
            if not contributing:
                # Defensive: a VALID assessment always has traced
                # observations, but never fabricate a signal without them.
                continue

            temperatures = [c.observation.temperature for c in contributing if c.observation.temperature is not None]
            humidities = [
                c.observation.relative_humidity for c in contributing if c.observation.relative_humidity is not None
            ]
            wind_speeds = [c.observation.wind_speed for c in contributing if c.observation.wind_speed is not None]
            gusts = [c.observation.wind_gust for c in contributing if c.observation.wind_gust is not None]
            if not temperatures or not humidities or not wind_speeds:
                continue

            results.append(
                AssessedWeatherConditions(
                    stored_assessment=stored,
                    conditions=AreaWeatherConditions(
                        area_name=stored.assessment.area_name,
                        observed_at=max(c.observation.timestamp for c in contributing),
                        assessed_at=stored.assessment.assessed_at,
                        station_count=len(contributing),
                        temperature_c=sum(temperatures) / len(temperatures),
                        relative_humidity_pct=sum(humidities) / len(humidities),
                        wind_speed_kmh=sum(wind_speeds) / len(wind_speeds),
                        wind_gust_kmh=(sum(gusts) / len(gusts)) if gusts else None,
                    ),
                )
            )
        return tuple(results)

    def get_latest_for_all_areas(self) -> tuple[AreaWeatherConditions, ...]:
        """Weather Conditions for every area's LATEST persisted assessment, sorted by area_name.

        An area whose latest assessment is not VALID (INSUFFICIENT_DATA,
        danger level None) is omitted - its weather is currently
        unavailable - rather than falling back to an older, staler
        assessment. get_latest_for_all_areas does not populate the
        observation trace, so each VALID latest assessment is re-read via
        get_by_id (one small read per area).
        """
        latest = self._fire_danger_assessment_repository.get_latest_for_all_areas()
        traced = []
        for stored in latest:
            if stored.assessment.level is None:
                continue
            stored_with_trace = self._fire_danger_assessment_repository.get_by_id(stored.assessment_id)
            if stored_with_trace is not None:
                traced.append(stored_with_trace)
        return tuple(
            sorted(
                (result.conditions for result in self.summarize_assessments(traced)),
                key=lambda conditions: conditions.area_name,
            )
        )
