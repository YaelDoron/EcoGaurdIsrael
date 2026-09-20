"""Read-only Fire Danger query service for the Operations dashboard (Task A4).

FireDangerQueryService assembles read models purely from data
FireDangerAssessmentAgent has already persisted (via
FireDangerAssessmentRepository) plus, for assessment detail, the weather
observations it traced (via WeatherRepository). It contains no FFWI
calculation, never runs FireDangerAssessmentAgent, never queries IMS, never
writes anything, and never infers FireEvent status/severity/spread - a
strict read/aggregate/map step, matching ActiveFireEventsService's own
read-only precedent (src/services/fire_event_read/active_fire_events_service.py).

Task A4 audit finding (see src/models/fire_danger_areas.py for the full
rationale): there is no independent, persistent assessment-area registry.
"Known area" is defined here as "has at least one persisted
FireDangerAssessment" - get_latest_for_area returns None only when area_id
has never been assessed at all (unknown area, -> 404 at the API layer); a
known area's snapshot always carries a populated `assessment` today (never
a true "known area, zero assessments" state, since that state cannot be
distinguished from "unknown area" without a registry this project does not
have).

No freshness/staleness judgement is made anywhere in this feature: the
project defines MAX_WEATHER_AGE_MINUTES for weather-*input* eligibility
during calculation only (src/services/fire_danger/fire_danger_input_config.py);
there is no canonical "assessment result is stale after N minutes" rule to
reuse, so none is invented here. The API layer (src/api/schemas/fire_danger.py)
optionally derives a neutral `age_seconds` (elapsed wall-clock time since
`assessed_at`) for display - a presentation computation, not a business
judgement, and not performed here since it depends on request time, not
persisted data.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.models.fire_danger_areas import (
    FireDangerAreaAssessmentSummary,
    FireDangerAreaSnapshot,
    FireDangerAreasResult,
)
from src.models.fire_danger_assessment_detail import (
    FireDangerAssessmentDetail,
    FireDangerAssessmentWeatherInputSummary,
)
from src.repositories.fire_danger_assessment_repository import (
    FireDangerAssessmentRepository,
    StoredFireDangerAssessment,
)
from src.repositories.weather_repository import WeatherRepository


class FireDangerQueryService:
    """Assemble Fire Danger read models from already-persisted assessment/weather data."""

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

    def get_latest_for_all_areas(self, *, as_of: datetime | None = None) -> FireDangerAreasResult:
        """Return every area that has ever been assessed, with its latest persisted assessment.

        Sorted deterministically by area_name then area_id (never by risk
        severity - see Task A4, Part 11) so a dashboard polling this
        endpoint repeatedly gets a stable ordering. Uses
        FireDangerAssessmentRepository.get_latest_for_all_areas, a single
        batched window-function query - no N+1 (one query per area).
        """
        snapshot_time = as_of if as_of is not None else datetime.now(timezone.utc)
        self._validate_aware_datetime("as_of", snapshot_time)

        stored = self._fire_danger_assessment_repository.get_latest_for_all_areas()
        areas = tuple(
            sorted(
                (self._to_area_snapshot(item) for item in stored),
                key=lambda area: (area.area_name, area.area_id),
            )
        )
        return FireDangerAreasResult(as_of=snapshot_time, areas=areas)

    def get_latest_for_area(self, area_id: str) -> FireDangerAreaSnapshot | None:
        """Return one area's latest persisted assessment.

        Returns None if area_id has never had a persisted assessment - the
        only "unknown area" signal this architecture can produce (see the
        module docstring).
        """
        stored = self._fire_danger_assessment_repository.get_latest_for_area(area_id)
        return self._to_area_snapshot(stored) if stored is not None else None

    def get_assessment_detail(self, assessment_id: int) -> FireDangerAssessmentDetail | None:
        """Return one persisted assessment with its preserved weather-input trace.

        Returns None if assessment_id does not exist. Never recalculates the
        score and never fetches current weather - `weather_inputs` are the
        exact WeatherObservation rows already traced at save time.
        """
        stored = self._fire_danger_assessment_repository.get_by_id(assessment_id)
        if stored is None:
            return None

        weather_inputs: tuple[FireDangerAssessmentWeatherInputSummary, ...] = ()
        if stored.observation_ids:
            stored_observations = self._weather_repository.get_observations_by_ids(stored.observation_ids)
            weather_inputs = tuple(
                FireDangerAssessmentWeatherInputSummary(
                    observation_id=item.observation_id,
                    station_external_id=item.station.external_station_id,
                    station_name=item.station.name,
                    observed_at=item.observation.timestamp,
                )
                for item in stored_observations
            )

        assessment = stored.assessment
        return FireDangerAssessmentDetail(
            assessment_id=stored.assessment_id,
            area_id=assessment.area_id,
            area_name=assessment.area_name,
            area_latitude=assessment.area_latitude,
            area_longitude=assessment.area_longitude,
            area_radius_km=assessment.area_radius_km,
            status=assessment.status,
            score=assessment.score,
            level=assessment.level,
            assessed_at=assessment.assessed_at,
            methodology=assessment.methodology,
            methodology_version=assessment.methodology_version,
            weather_inputs=weather_inputs,
        )

    @staticmethod
    def _to_area_snapshot(stored: StoredFireDangerAssessment) -> FireDangerAreaSnapshot:
        assessment = stored.assessment
        return FireDangerAreaSnapshot(
            area_id=assessment.area_id,
            area_name=assessment.area_name,
            area_latitude=assessment.area_latitude,
            area_longitude=assessment.area_longitude,
            area_radius_km=assessment.area_radius_km,
            assessment=FireDangerAreaAssessmentSummary(
                assessment_id=stored.assessment_id,
                status=assessment.status,
                score=assessment.score,
                level=assessment.level,
                assessed_at=assessment.assessed_at,
                methodology=assessment.methodology,
                methodology_version=assessment.methodology_version,
            ),
        )

    @staticmethod
    def _validate_aware_datetime(field_name: str, value: object) -> None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime, got {value!r}")
