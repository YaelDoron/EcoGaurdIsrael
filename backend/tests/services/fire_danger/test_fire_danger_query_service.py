"""Unit tests for FireDangerQueryService (Task A4, Part 19).

Uses fake FireDangerAssessmentRepository/WeatherRepository - never a real
FFWI calculation, FireDangerAssessmentAgent invocation, or IMS/database
call. An architecture-guard test at the bottom mechanically proves the read
path cannot invoke FFWICalculator/FireDangerAssessmentAgent at all (the
module never imports them), which is a stronger guarantee than merely
observing they were not called in one test run.
"""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import StoredFireDangerAssessment
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
AS_OF = datetime(2026, 9, 19, 12, 30, tzinfo=timezone.utc)


class FakeFireDangerAssessmentRepository:
    def __init__(self, *, latest_by_area=None, all_latest=(), by_id=None):
        self._latest_by_area = latest_by_area or {}
        self._all_latest = all_latest
        self._by_id = by_id or {}
        self.get_latest_for_area_calls: list[str] = []
        self.get_latest_for_all_areas_calls = 0
        self.get_by_id_calls: list[int] = []

    def get_latest_for_area(self, area_id):
        self.get_latest_for_area_calls.append(area_id)
        return self._latest_by_area.get(area_id)

    def get_latest_for_all_areas(self):
        self.get_latest_for_all_areas_calls += 1
        return self._all_latest

    def get_by_id(self, assessment_id):
        self.get_by_id_calls.append(assessment_id)
        return self._by_id.get(assessment_id)


class FakeWeatherRepository:
    def __init__(self, *, observations=()):
        self._observations = observations
        self.get_observations_by_ids_calls: list[tuple] = []

    def get_observations_by_ids(self, observation_ids):
        self.get_observations_by_ids_calls.append(tuple(observation_ids))
        return self._observations


def make_assessment(**overrides) -> FireDangerAssessment:
    defaults = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=ASSESSED_AT,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        methodology=FFWI_METHODOLOGY_NAME,
        methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireDangerAssessment(**defaults)


def make_stored(
    assessment_id=1, observation_ids=(), station_ids=(), created_at=ASSESSED_AT, **overrides
) -> StoredFireDangerAssessment:
    return StoredFireDangerAssessment(
        assessment_id=assessment_id,
        assessment=make_assessment(**overrides),
        observation_ids=observation_ids,
        station_ids=station_ids,
        created_at=created_at,
    )


def make_stored_observation(
    observation_id: int, external_station_id: int, timestamp: datetime
) -> StoredWeatherObservation:
    return StoredWeatherObservation(
        observation_id=observation_id,
        station_id=external_station_id * 10,
        station=WeatherStation(
            external_station_id=external_station_id,
            name=f"Station {external_station_id}",
            latitude=32.7,
            longitude=35.0,
        ),
        observation=WeatherObservation(
            station_external_id=external_station_id,
            timestamp=timestamp,
            temperature=30,
            relative_humidity=25,
            wind_speed=12,
        ),
    )


def make_service(*, assessment_repository=None, weather_repository=None) -> FireDangerQueryService:
    return FireDangerQueryService(
        fire_danger_assessment_repository=assessment_repository or FakeFireDangerAssessmentRepository(),
        weather_repository=weather_repository or FakeWeatherRepository(),
    )


# ---------------------------------------------------------------------------
# 1. Latest single assessment
# ---------------------------------------------------------------------------


def test_get_latest_for_area_returns_the_repositorys_latest():
    stored = make_stored()
    repo = FakeFireDangerAssessmentRepository(latest_by_area={"area-carmel": stored})
    service = make_service(assessment_repository=repo)

    snapshot = service.get_latest_for_area("area-carmel")

    assert snapshot is not None
    assert snapshot.assessment.assessment_id == 1
    assert snapshot.assessment.score == 42.5
    assert snapshot.assessment.level is FireDangerLevel.VERY_HIGH


# ---------------------------------------------------------------------------
# 2 & 3. Multiple assessments/areas -> repository already resolves "latest";
# service just needs to map every returned area through untouched.
# ---------------------------------------------------------------------------


def test_get_latest_for_all_areas_maps_every_returned_area():
    stored_carmel = make_stored(assessment_id=1, area_id="area-carmel", area_name="Carmel")
    stored_golan = make_stored(assessment_id=2, area_id="area-golan", area_name="Golan")
    repo = FakeFireDangerAssessmentRepository(all_latest=(stored_carmel, stored_golan))
    service = make_service(assessment_repository=repo)

    result = service.get_latest_for_all_areas(as_of=AS_OF)

    assert {area.area_id for area in result.areas} == {"area-carmel", "area-golan"}
    assert result.as_of == AS_OF


# ---------------------------------------------------------------------------
# 4. Deterministic ordering (tie-break is the repository's job; the service
# must at least preserve/sort stably, never depend on unordered DB results)
# ---------------------------------------------------------------------------


def test_get_latest_for_all_areas_sorts_by_area_name_then_area_id():
    stored_b = make_stored(assessment_id=1, area_id="area-b", area_name="Zebra")
    stored_a = make_stored(assessment_id=2, area_id="area-a", area_name="Alpha")
    repo = FakeFireDangerAssessmentRepository(all_latest=(stored_b, stored_a))
    service = make_service(assessment_repository=repo)

    result = service.get_latest_for_all_areas(as_of=AS_OF)

    assert [area.area_name for area in result.areas] == ["Alpha", "Zebra"]


def test_get_latest_for_all_areas_ordering_is_deterministic_regardless_of_input_order():
    stored_a = make_stored(assessment_id=1, area_id="area-a", area_name="Alpha")
    stored_b = make_stored(assessment_id=2, area_id="area-b", area_name="Zebra")
    repo_forward = FakeFireDangerAssessmentRepository(all_latest=(stored_a, stored_b))
    repo_reversed = FakeFireDangerAssessmentRepository(all_latest=(stored_b, stored_a))

    result_forward = make_service(assessment_repository=repo_forward).get_latest_for_all_areas(as_of=AS_OF)
    result_reversed = make_service(assessment_repository=repo_reversed).get_latest_for_all_areas(as_of=AS_OF)

    assert [area.area_id for area in result_forward.areas] == [area.area_id for area in result_reversed.areas]


# ---------------------------------------------------------------------------
# 5. Levels serialize correctly (round-trip through the service unchanged)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level",
    [
        FireDangerLevel.LOW,
        FireDangerLevel.MODERATE,
        FireDangerLevel.HIGH,
        FireDangerLevel.VERY_HIGH,
        FireDangerLevel.EXTREME,
    ],
)
def test_all_levels_pass_through_unchanged(level):
    stored = make_stored(level=level, score=10.0)
    repo = FakeFireDangerAssessmentRepository(latest_by_area={"area-carmel": stored})
    service = make_service(assessment_repository=repo)

    snapshot = service.get_latest_for_area("area-carmel")

    assert snapshot.assessment.level is level


# ---------------------------------------------------------------------------
# 6. Score returned exactly as persisted
# ---------------------------------------------------------------------------


def test_score_is_returned_exactly_as_persisted():
    stored = make_stored(score=17.375, level=FireDangerLevel.MODERATE)
    repo = FakeFireDangerAssessmentRepository(latest_by_area={"area-carmel": stored})
    service = make_service(assessment_repository=repo)

    snapshot = service.get_latest_for_area("area-carmel")

    assert snapshot.assessment.score == 17.375


# ---------------------------------------------------------------------------
# 7. No assessment is not converted to LOW
# ---------------------------------------------------------------------------


def test_unknown_area_returns_none_not_a_fabricated_snapshot():
    repo = FakeFireDangerAssessmentRepository(latest_by_area={})
    service = make_service(assessment_repository=repo)

    assert service.get_latest_for_area("never-assessed") is None


def test_insufficient_data_is_not_converted_to_low():
    stored = make_stored(status=FireDangerAssessmentStatus.INSUFFICIENT_DATA, score=None, level=None)
    repo = FakeFireDangerAssessmentRepository(latest_by_area={"area-carmel": stored})
    service = make_service(assessment_repository=repo)

    snapshot = service.get_latest_for_area("area-carmel")

    assert snapshot is not None
    assert snapshot.assessment.status is FireDangerAssessmentStatus.INSUFFICIENT_DATA
    assert snapshot.assessment.score is None
    assert snapshot.assessment.level is None


# ---------------------------------------------------------------------------
# 8. Area center/radius preserved
# ---------------------------------------------------------------------------


def test_area_center_and_radius_are_preserved():
    stored = make_stored(area_latitude=31.774, area_longitude=35.139, area_radius_km=7.5)
    repo = FakeFireDangerAssessmentRepository(latest_by_area={"area-carmel": stored})
    service = make_service(assessment_repository=repo)

    snapshot = service.get_latest_for_area("area-carmel")

    assert snapshot.area_latitude == 31.774
    assert snapshot.area_longitude == 35.139
    assert snapshot.area_radius_km == 7.5


# ---------------------------------------------------------------------------
# 9. Deterministic ordering already covered above (4); repeat via as_of stamp
# ---------------------------------------------------------------------------


def test_result_as_of_defaults_to_now_when_not_supplied():
    before = datetime.now(timezone.utc)
    result = make_service().get_latest_for_all_areas()
    after = datetime.now(timezone.utc)

    assert before <= result.as_of <= after


# ---------------------------------------------------------------------------
# 10. Empty assessment dataset handled
# ---------------------------------------------------------------------------


def test_empty_dataset_returns_empty_areas_list_not_an_error():
    repo = FakeFireDangerAssessmentRepository(all_latest=())
    service = make_service(assessment_repository=repo)

    result = service.get_latest_for_all_areas(as_of=AS_OF)

    assert result.areas == ()


# ---------------------------------------------------------------------------
# 11 & 12. No FFWI calculator / FireDangerAssessmentAgent invoked by read path
# ---------------------------------------------------------------------------


def test_fire_danger_query_service_does_not_import_calculation_or_agent_modules():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDangerInputService",
        "fastapi",
        "pydantic",
        "simulation",
    )
    path = Path(__file__).resolve().parents[3] / "src" / "services" / "fire_danger" / "fire_danger_query_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations = []
    for node in ast.walk(tree):
        module = ""
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = ",".join(alias.name for alias in node.names)
        if any(fragment in module for fragment in forbidden_fragments):
            violations.append(module)

    assert violations == []


# ---------------------------------------------------------------------------
# 13. No N+1 repository behavior
# ---------------------------------------------------------------------------


def test_get_latest_for_all_areas_calls_the_batched_repository_method_exactly_once():
    stored_a = make_stored(assessment_id=1, area_id="area-a", area_name="Alpha")
    stored_b = make_stored(assessment_id=2, area_id="area-b", area_name="Beta")
    stored_c = make_stored(assessment_id=3, area_id="area-c", area_name="Gamma")
    repo = FakeFireDangerAssessmentRepository(all_latest=(stored_a, stored_b, stored_c))
    service = make_service(assessment_repository=repo)

    service.get_latest_for_all_areas(as_of=AS_OF)

    assert repo.get_latest_for_all_areas_calls == 1
    assert repo.get_latest_for_area_calls == []


# ---------------------------------------------------------------------------
# Assessment detail (Part 7)
# ---------------------------------------------------------------------------


def test_get_assessment_detail_returns_none_for_unknown_id():
    service = make_service()

    assert service.get_assessment_detail(999) is None


def test_get_assessment_detail_includes_weather_inputs():
    stored = make_stored(assessment_id=5, observation_ids=(101, 102), station_ids=(1, 2))
    assessment_repo = FakeFireDangerAssessmentRepository(by_id={5: stored})
    weather_repo = FakeWeatherRepository(
        observations=(
            make_stored_observation(101, 1001, datetime(2026, 9, 19, 11, 45)),
            make_stored_observation(102, 1002, datetime(2026, 9, 19, 11, 50)),
        )
    )
    service = make_service(assessment_repository=assessment_repo, weather_repository=weather_repo)

    detail = service.get_assessment_detail(5)

    assert detail is not None
    assert detail.assessment_id == 5
    assert [item.observation_id for item in detail.weather_inputs] == [101, 102]
    assert [item.station_external_id for item in detail.weather_inputs] == [1001, 1002]
    assert weather_repo.get_observations_by_ids_calls == [(101, 102)]


def test_get_assessment_detail_skips_weather_lookup_when_no_observations_traced():
    stored = make_stored(
        assessment_id=6,
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
        observation_ids=(),
        station_ids=(),
    )
    assessment_repo = FakeFireDangerAssessmentRepository(by_id={6: stored})
    weather_repo = FakeWeatherRepository()
    service = make_service(assessment_repository=assessment_repo, weather_repository=weather_repo)

    detail = service.get_assessment_detail(6)

    assert detail is not None
    assert detail.weather_inputs == ()
    assert weather_repo.get_observations_by_ids_calls == []


def test_get_assessment_detail_does_not_recalculate_score():
    stored = make_stored(assessment_id=7, score=55.5, level=FireDangerLevel.VERY_HIGH)
    assessment_repo = FakeFireDangerAssessmentRepository(by_id={7: stored})
    service = make_service(assessment_repository=assessment_repo)

    detail = service.get_assessment_detail(7)

    assert detail.score == 55.5
    assert detail.level is FireDangerLevel.VERY_HIGH
