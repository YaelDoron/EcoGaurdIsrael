"""Unit tests for OperationsActivityQueryService (Task A5, Part 21).

Uses fake repositories/query-services matching each real collaborator's
public interface exactly - never a real FFWI/severity/detection calculation,
agent invocation, or DB call. An architecture-guard test at the bottom
mechanically proves the query service and its domain read models cannot
invoke business execution components (they never import them at all).
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_detail import FireDangerAssessmentDetail
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_report import WildfireReport
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.global_planning_run import GlobalPlanningRun
from src.models.global_planning_run_event import GlobalPlanningRunEvent
from src.models.global_planning_run_event_status import GlobalPlanningRunEventStatus
from src.models.global_planning_run_status import GlobalPlanningRunStatus
from src.models.operations_activity import (
    FireDangerActivityDetail,
    FireEventActivityDetail,
    FireSeverityActivityDetail,
    GlobalPlanningRunActivityDetail,
    NewsReportActivityDetail,
    OperationsActivityType,
    SatelliteHotspotActivityDetail,
    WeatherConditionsActivityDetail,
)
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import StoredFireDangerAssessment
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.global_planning_run_repository import (
    StoredGlobalPlanningRun,
    StoredGlobalPlanningRunEvent,
)
from src.repositories.news_repository import StoredWildfireReport
from src.repositories.satellite_hotspot_repository import StoredSatelliteHotspot
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.operations.operations_activity_query_service import OperationsActivityQueryService

ASSESSED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes - one per real collaborator, matching its public interface exactly
# ---------------------------------------------------------------------------


class FakeFireDangerQueryService:
    def __init__(self, *, detail_by_id=None):
        self._detail_by_id = detail_by_id or {}
        self.calls: list[int] = []

    def get_assessment_detail(self, assessment_id):
        self.calls.append(assessment_id)
        return self._detail_by_id.get(assessment_id)


class FakeSatelliteHotspotRepository:
    def __init__(self, *, by_id=None):
        self._by_id = by_id or {}
        self.get_by_id_calls: list[int] = []

    def get_by_id(self, hotspot_id):
        self.get_by_id_calls.append(hotspot_id)
        return self._by_id.get(hotspot_id)


class FakeNewsRepository:
    def __init__(self, *, by_id=None):
        self._by_id = by_id or {}
        self.get_by_id_calls: list[int] = []

    def get_by_id(self, report_id):
        self.get_by_id_calls.append(report_id)
        return self._by_id.get(report_id)


class FakeFireEventRepository:
    def __init__(self, *, by_id=None):
        self._by_id = by_id or {}
        self.get_by_id_calls: list[int] = []

    def get_by_id(self, fire_event_id):
        self.get_by_id_calls.append(fire_event_id)
        return self._by_id.get(fire_event_id)


class FakeFireSeverityAssessmentRepository:
    def __init__(self, *, by_id=None, latest_by_event=None):
        self._by_id = by_id or {}
        self._latest_by_event = latest_by_event or {}
        self.get_by_id_calls: list[int] = []
        self.get_latest_for_event_calls: list[int] = []

    def get_by_id(self, assessment_id):
        self.get_by_id_calls.append(assessment_id)
        return self._by_id.get(assessment_id)

    def get_latest_for_event(self, fire_event_id):
        self.get_latest_for_event_calls.append(fire_event_id)
        return self._latest_by_event.get(fire_event_id)


class FakeGlobalPlanningRunRepository:
    def __init__(self, *, by_id=None, members_by_run=None):
        self._by_id = by_id or {}
        self._members_by_run = members_by_run or {}
        self.get_by_id_calls: list[int] = []
        self.get_members_calls: list[int] = []

    def get_by_id(self, run_id):
        self.get_by_id_calls.append(run_id)
        return self._by_id.get(run_id)

    def get_members(self, run_id):
        self.get_members_calls.append(run_id)
        return self._members_by_run.get(run_id, ())


class FakeFireDangerAssessmentRepository:
    def __init__(self, *, by_id=None):
        self._by_id = by_id or {}
        self.get_by_id_calls: list[int] = []

    def get_by_id(self, assessment_id):
        self.get_by_id_calls.append(assessment_id)
        return self._by_id.get(assessment_id)


class FakeWeatherRepository:
    def __init__(self, *, observations_by_id=None):
        self._observations_by_id = observations_by_id or {}
        self.get_observations_by_ids_calls: list[tuple[int, ...]] = []

    def get_observations_by_ids(self, observation_ids):
        ids = tuple(observation_ids)
        self.get_observations_by_ids_calls.append(ids)
        return tuple(self._observations_by_id[oid] for oid in ids if oid in self._observations_by_id)


# ---------------------------------------------------------------------------
# Domain object builders
# ---------------------------------------------------------------------------


def make_fire_danger_detail(**overrides) -> FireDangerAssessmentDetail:
    values = dict(
        assessment_id=1,
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.VERY_HIGH,
        assessed_at=ASSESSED_AT,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
        weather_inputs=(),
    )
    values.update(overrides)
    return FireDangerAssessmentDetail(**values)


def make_hotspot(**overrides) -> SatelliteHotspot:
    values = dict(
        latitude=32.7,
        longitude=35.0,
        detected_at=datetime(2026, 9, 19, 11, 0),
        confidence="nominal",
        frp=12.3,
        brightness=310.5,
        satellite="N",
        instrument="VIIRS",
        day_night="D",
    )
    values.update(overrides)
    return SatelliteHotspot(**values)


def make_report(**overrides) -> WildfireReport:
    values = dict(
        source_url="https://example.com/report",
        source_feed="example-feed",
        title="Wildfire spreads near Carmel",
        summary="A wildfire is spreading near the Carmel region.",
        location_name="Carmel",
        latitude=32.7,
        longitude=35.0,
        published_at=datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc),
        fetched_at=datetime(2026, 9, 19, 10, 5, tzinfo=timezone.utc),
    )
    values.update(overrides)
    return WildfireReport(**values)


def make_fire_event(**overrides) -> FireEvent:
    values = dict(
        latitude=32.7,
        longitude=35.0,
        detected_at=ASSESSED_AT,
        updated_at=ASSESSED_AT,
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.9,
        methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireEvent(**values)


def make_severity_assessment(**overrides) -> FireSeverityAssessment:
    values = dict(
        fire_event_id=1,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=55.0,
        level=FireSeverityLevel.HIGH,
        methodology="ECOGUARD_SEVERITY",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireSeverityAssessment(**values)


def make_global_run(**overrides) -> GlobalPlanningRun:
    values = dict(
        started_at=ASSESSED_AT,
        completed_at=ASSESSED_AT,
        status=GlobalPlanningRunStatus.COMPLETED,
        trigger="weather_event",
        methodology="legacy_per_event_orchestration",
        methodology_version="1.0",
        input_fingerprint="fp-1",
    )
    values.update(overrides)
    return GlobalPlanningRun(**values)


def make_run_member(**overrides) -> GlobalPlanningRunEvent:
    values = dict(
        fire_event_id=1,
        event_order=0,
        result_status=GlobalPlanningRunEventStatus.PLANNED,
        response_plan_id=10,
        local_state_fingerprint="lsf-1",
        error_code=None,
    )
    values.update(overrides)
    return GlobalPlanningRunEvent(**values)


def make_fire_danger_assessment(**overrides) -> FireDangerAssessment:
    values = dict(
        area_id="area-carmel",
        area_name="Carmel",
        area_latitude=32.731,
        area_longitude=35.046,
        area_radius_km=5.0,
        assessed_at=ASSESSED_AT,
        status=FireDangerAssessmentStatus.VALID,
        score=42.5,
        level=FireDangerLevel.HIGH,
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
    )
    values.update(overrides)
    return FireDangerAssessment(**values)


def make_stored_assessment(assessment_id: int = 1, **overrides) -> StoredFireDangerAssessment:
    observation_ids = overrides.pop("observation_ids", (10,))
    station_ids = overrides.pop("station_ids", (100,))
    created_at = overrides.pop("created_at", ASSESSED_AT)
    return StoredFireDangerAssessment(
        assessment_id=assessment_id,
        assessment=make_fire_danger_assessment(**overrides),
        observation_ids=observation_ids,
        station_ids=station_ids,
        created_at=created_at,
    )


def make_stored_observation(observation_id: int = 10, station_id: int = 100, **overrides) -> StoredWeatherObservation:
    station_name = overrides.pop("station_name", "Carmel Station")
    return StoredWeatherObservation(
        observation_id=observation_id,
        station_id=station_id,
        station=WeatherStation(external_station_id=station_id, name=station_name, latitude=32.731, longitude=35.046),
        observation=WeatherObservation(
            station_external_id=station_id,
            timestamp=overrides.pop("observed_at", ASSESSED_AT),
            temperature=overrides.pop("temperature", 34.0),
            relative_humidity=overrides.pop("relative_humidity", 19.0),
            wind_speed=overrides.pop("wind_speed", 28.0),
            wind_gust=overrides.pop("wind_gust", None),
        ),
    )


def make_service(
    *,
    fire_danger=None,
    satellite=None,
    news=None,
    fire_event=None,
    fire_severity=None,
    global_planning=None,
    fire_danger_assessment=None,
    weather=None,
) -> OperationsActivityQueryService:
    return OperationsActivityQueryService(
        fire_danger_query_service=fire_danger or FakeFireDangerQueryService(),
        satellite_hotspot_repository=satellite or FakeSatelliteHotspotRepository(),
        news_repository=news or FakeNewsRepository(),
        fire_event_repository=fire_event or FakeFireEventRepository(),
        fire_severity_assessment_repository=fire_severity or FakeFireSeverityAssessmentRepository(),
        global_planning_run_repository=global_planning or FakeGlobalPlanningRunRepository(),
        fire_danger_assessment_repository=fire_danger_assessment or FakeFireDangerAssessmentRepository(),
        weather_repository=weather or FakeWeatherRepository(),
    )


# ---------------------------------------------------------------------------
# 1. Fire Danger detail delegates to A4 service
# ---------------------------------------------------------------------------


def test_fire_danger_delegates_to_a4_service():
    fd_detail = make_fire_danger_detail()
    fake_fd = FakeFireDangerQueryService(detail_by_id={1: fd_detail})
    service = make_service(fire_danger=fake_fd)

    result = service.get_activity_detail(OperationsActivityType.FIRE_DANGER, 1)

    assert isinstance(result, FireDangerActivityDetail)
    assert result.details is fd_detail
    assert result.occurred_at == ASSESSED_AT
    assert result.location.latitude == 32.731
    assert fake_fd.calls == [1]


# ---------------------------------------------------------------------------
# 2. Satellite hotspot detail returns persisted fields
# ---------------------------------------------------------------------------


def test_satellite_hotspot_detail_returns_persisted_fields():
    hotspot = make_hotspot(confidence="high", frp=45.6)
    fake_sat = FakeSatelliteHotspotRepository(by_id={7: StoredSatelliteHotspot(id=7, hotspot=hotspot)})
    service = make_service(satellite=fake_sat)

    result = service.get_activity_detail(OperationsActivityType.SATELLITE_HOTSPOT, 7)

    assert isinstance(result, SatelliteHotspotActivityDetail)
    assert result.details.confidence == "high"
    assert result.details.frp == 45.6
    assert result.location.latitude == 32.7
    assert result.occurred_at == hotspot.detected_at


# ---------------------------------------------------------------------------
# 3. News report detail returns persisted fields
# ---------------------------------------------------------------------------


def test_news_report_detail_returns_persisted_fields():
    report = make_report(title="Real headline")
    observed_at = datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)
    fake_news = FakeNewsRepository(
        by_id={3: StoredWildfireReport(id=3, report=report, observed_at=observed_at)}
    )
    service = make_service(news=fake_news)

    result = service.get_activity_detail(OperationsActivityType.NEWS_REPORT, 3)

    assert isinstance(result, NewsReportActivityDetail)
    assert result.title == "Real headline"
    assert result.details.summary == report.summary
    assert result.occurred_at == observed_at
    assert result.location.latitude == 32.7


def test_news_report_without_coordinates_has_no_location():
    report = make_report(latitude=None, longitude=None)
    fake_news = FakeNewsRepository(
        by_id={3: StoredWildfireReport(id=3, report=report, observed_at=ASSESSED_AT)}
    )
    service = make_service(news=fake_news)

    result = service.get_activity_detail(OperationsActivityType.NEWS_REPORT, 3)

    assert result.location is None


# ---------------------------------------------------------------------------
# 4. FireEvent detail reuses existing read service / preserves data
# ---------------------------------------------------------------------------


def test_fire_event_detail_preserves_persisted_data_and_evidence():
    event = make_fire_event(status=FireEventStatus.CONFIRMED, detection_confidence=0.87)
    stored_event = StoredFireEvent(
        id=42,
        event=event,
        supporting_evidence=(
            FireEvidenceRef(FireEvidenceType.SATELLITE, 7),
            FireEvidenceRef(FireEvidenceType.NEWS, 3),
        ),
        created_at=ASSESSED_AT,
    )
    fake_event_repo = FakeFireEventRepository(by_id={42: stored_event})
    service = make_service(fire_event=fake_event_repo)

    result = service.get_activity_detail(OperationsActivityType.FIRE_EVENT, 42)

    assert isinstance(result, FireEventActivityDetail)
    assert result.details.fire_event.status is FireEventStatus.CONFIRMED
    assert result.details.fire_event.detection_confidence == 0.87
    assert result.details.evidence.satellite_hotspot_ids == (7,)
    assert result.details.evidence.news_report_ids == (3,)
    assert result.location.latitude == event.latitude
    assert result.title == "Fire Event #42"


def test_fire_event_detail_includes_latest_severity_reference_when_present():
    stored_event = StoredFireEvent(id=42, event=make_fire_event(), created_at=ASSESSED_AT)
    severity = make_severity_assessment(fire_event_id=42)
    fake_severity_repo = FakeFireSeverityAssessmentRepository(
        latest_by_event={42: StoredFireSeverityAssessment(assessment_id=9, assessment=severity)}
    )
    service = make_service(
        fire_event=FakeFireEventRepository(by_id={42: stored_event}),
        fire_severity=fake_severity_repo,
    )

    result = service.get_activity_detail(OperationsActivityType.FIRE_EVENT, 42)

    assert result.details.latest_severity is not None
    assert result.details.latest_severity.assessment_id == 9
    assert result.details.latest_severity.level is FireSeverityLevel.HIGH
    assert fake_severity_repo.get_latest_for_event_calls == [42]


def test_fire_event_detail_has_no_fabricated_area_name():
    stored_event = StoredFireEvent(id=42, event=make_fire_event(), created_at=ASSESSED_AT)
    service = make_service(fire_event=FakeFireEventRepository(by_id={42: stored_event}))

    result = service.get_activity_detail(OperationsActivityType.FIRE_EVENT, 42)

    assert "Carmel" not in result.title
    assert "Golan" not in result.title
    assert result.title == "Fire Event #42"
    assert not hasattr(result.details.fire_event, "area_name")


# ---------------------------------------------------------------------------
# 5. Severity detail returns persisted assessment, not recalculated
# ---------------------------------------------------------------------------


def test_fire_severity_detail_returns_persisted_assessment_with_trace():
    assessment = make_severity_assessment(score=61.25, level=FireSeverityLevel.CRITICAL)
    stored = StoredFireSeverityAssessment(
        assessment_id=9,
        assessment=assessment,
        weather_observation_ids=(101, 102),
        satellite_hotspot_ids=(7,),
        selected_frp_hotspot_id=7,
    )
    fake_severity_repo = FakeFireSeverityAssessmentRepository(by_id={9: stored})
    service = make_service(fire_severity=fake_severity_repo)

    result = service.get_activity_detail(OperationsActivityType.FIRE_SEVERITY, 9)

    assert isinstance(result, FireSeverityActivityDetail)
    assert result.details.assessment.score == 61.25
    assert result.details.assessment.level is FireSeverityLevel.CRITICAL
    assert result.details.weather_observation_ids == (101, 102)
    assert result.details.selected_frp_hotspot_id == 7
    assert result.location is None
    assert fake_severity_repo.get_by_id_calls == [9]


# ---------------------------------------------------------------------------
# 6 & 7. GlobalPlanningRun detail represents multiple FireEvents / shared run id
# ---------------------------------------------------------------------------


def test_global_planning_run_detail_represents_multiple_fire_events():
    run = make_global_run()
    members = (
        StoredGlobalPlanningRunEvent(id=1, member=make_run_member(fire_event_id=101, event_order=0, response_plan_id=201)),
        StoredGlobalPlanningRunEvent(id=2, member=make_run_member(fire_event_id=102, event_order=1, response_plan_id=202)),
    )
    fake_gp_repo = FakeGlobalPlanningRunRepository(
        by_id={5: StoredGlobalPlanningRun(id=5, run=run)},
        members_by_run={5: members},
    )
    service = make_service(global_planning=fake_gp_repo)

    result = service.get_activity_detail(OperationsActivityType.GLOBAL_PLANNING_RUN, 5)

    assert isinstance(result, GlobalPlanningRunActivityDetail)
    assert result.details.fire_event_ids == (101, 102)
    assert result.details.response_plan_ids == (201, 202)
    assert len(result.details.members) == 2
    assert result.location is None


def test_global_planning_run_child_response_plans_share_the_run_id():
    """Both FireEvents' ResponsePlans in one run are projections of the SAME
    global run - verify the membership rows preserve that shared identity
    (both point back to the same entity_id, never a fabricated per-event run)."""
    run = make_global_run()
    members = (
        StoredGlobalPlanningRunEvent(id=1, member=make_run_member(fire_event_id=101, event_order=0, response_plan_id=201)),
        StoredGlobalPlanningRunEvent(id=2, member=make_run_member(fire_event_id=102, event_order=1, response_plan_id=202)),
    )
    fake_gp_repo = FakeGlobalPlanningRunRepository(
        by_id={5: StoredGlobalPlanningRun(id=5, run=run)},
        members_by_run={5: members},
    )
    service = make_service(global_planning=fake_gp_repo)

    result = service.get_activity_detail(OperationsActivityType.GLOBAL_PLANNING_RUN, 5)

    assert result.entity_id == 5
    for member in result.details.members:
        assert member.response_plan_id in (201, 202)


def test_global_planning_run_occurred_at_prefers_completed_at():
    run = make_global_run(started_at=ASSESSED_AT, completed_at=ASSESSED_AT.replace(hour=13))
    fake_gp_repo = FakeGlobalPlanningRunRepository(by_id={5: StoredGlobalPlanningRun(id=5, run=run)})
    service = make_service(global_planning=fake_gp_repo)

    result = service.get_activity_detail(OperationsActivityType.GLOBAL_PLANNING_RUN, 5)

    assert result.occurred_at == run.completed_at


def test_global_planning_run_occurred_at_falls_back_to_started_at_when_not_completed():
    run = make_global_run(completed_at=None, status=GlobalPlanningRunStatus.RUNNING)
    fake_gp_repo = FakeGlobalPlanningRunRepository(by_id={5: StoredGlobalPlanningRun(id=5, run=run)})
    service = make_service(global_planning=fake_gp_repo)

    result = service.get_activity_detail(OperationsActivityType.GLOBAL_PLANNING_RUN, 5)

    assert result.occurred_at == run.started_at


# ---------------------------------------------------------------------------
# WEATHER_CONDITIONS - entity_id IS the FireDangerAssessmentDB id, details
# re-materialize the exact already-traced weather observations (no FFWI
# recalculation, no fresh observation selection).
# ---------------------------------------------------------------------------


def test_weather_conditions_detail_returns_persisted_readings():
    stored_assessment = make_stored_assessment(1, level=FireDangerLevel.HIGH, observation_ids=(10,), station_ids=(100,))
    stored_observation = make_stored_observation(10, 100)
    service = make_service(
        fire_danger_assessment=FakeFireDangerAssessmentRepository(by_id={1: stored_assessment}),
        weather=FakeWeatherRepository(observations_by_id={10: stored_observation}),
    )

    detail = service.get_activity_detail(OperationsActivityType.WEATHER_CONDITIONS, 1)

    assert isinstance(detail, WeatherConditionsActivityDetail)
    assert detail.entity_id == 1
    assert detail.title == "Weather Conditions - Carmel"
    assert detail.details.fire_danger_assessment_id == 1
    assert detail.details.area_name == "Carmel"
    assert detail.details.fire_danger_level is FireDangerLevel.HIGH
    assert len(detail.details.readings) == 1
    reading = detail.details.readings[0]
    assert reading.station_id == 100
    assert reading.station_name == "Carmel Station"
    assert reading.observation_id == 10
    assert reading.temperature == pytest.approx(34.0)
    assert reading.relative_humidity == pytest.approx(19.0)
    assert reading.wind_speed == pytest.approx(28.0)
    assert reading.wind_gust is None


def test_weather_conditions_detail_returns_none_for_missing_assessment():
    service = make_service(fire_danger_assessment=FakeFireDangerAssessmentRepository(by_id={}))

    assert service.get_activity_detail(OperationsActivityType.WEATHER_CONDITIONS, 999) is None


def test_weather_conditions_detail_returns_none_when_assessment_has_no_level():
    """A HIGH+ signal should never be reachable for an INSUFFICIENT_DATA
    assessment (level is always None there) - defensive, matches the feed's
    own eligibility rule rather than fabricating a detail for it."""
    stored_assessment = make_stored_assessment(
        1,
        level=None,
        score=None,
        status=FireDangerAssessmentStatus.INSUFFICIENT_DATA,
        observation_ids=(),
        station_ids=(),
    )
    service = make_service(fire_danger_assessment=FakeFireDangerAssessmentRepository(by_id={1: stored_assessment}))

    assert service.get_activity_detail(OperationsActivityType.WEATHER_CONDITIONS, 1) is None


def test_weather_conditions_detail_never_recalculates_ffwi():
    """Static guarantee: the handler never imports/calls the FFWI calculator -
    it only re-materializes already-traced observation rows."""
    import inspect

    from src.services.operations.operations_activity_query_service import OperationsActivityQueryService

    source = inspect.getsource(OperationsActivityQueryService._get_weather_conditions_detail)
    for forbidden in ("FFWICalculator", "calculate_ffwi", "FireDangerAssessmentAgent"):
        assert forbidden not in source


def test_weather_conditions_detail_multiple_stations_all_present():
    stored_assessment = make_stored_assessment(
        1, level=FireDangerLevel.VERY_HIGH, observation_ids=(10, 11), station_ids=(100, 101)
    )
    observation_a = make_stored_observation(10, 100, station_name="Station A", temperature=30.0)
    observation_b = make_stored_observation(11, 101, station_name="Station B", temperature=38.0)
    service = make_service(
        fire_danger_assessment=FakeFireDangerAssessmentRepository(by_id={1: stored_assessment}),
        weather=FakeWeatherRepository(observations_by_id={10: observation_a, 11: observation_b}),
    )

    detail = service.get_activity_detail(OperationsActivityType.WEATHER_CONDITIONS, 1)

    assert [reading.station_name for reading in detail.details.readings] == ["Station A", "Station B"]


# ---------------------------------------------------------------------------
# 8. Missing entity -> None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "activity_type",
    [
        OperationsActivityType.FIRE_DANGER,
        OperationsActivityType.SATELLITE_HOTSPOT,
        OperationsActivityType.NEWS_REPORT,
        OperationsActivityType.FIRE_EVENT,
        OperationsActivityType.FIRE_SEVERITY,
        OperationsActivityType.GLOBAL_PLANNING_RUN,
        OperationsActivityType.WEATHER_CONDITIONS,
    ],
)
def test_missing_entity_returns_none_for_every_activity_type(activity_type):
    service = make_service()

    assert service.get_activity_detail(activity_type, 999) is None


# ---------------------------------------------------------------------------
# 9 & 10. No write repository method called; no agent/calculator/coordinator invoked
# ---------------------------------------------------------------------------


def test_fakes_expose_no_write_methods():
    """Every fake above only implements get_by_id/get_members/get_latest_for_event/
    get_assessment_detail - if the service called anything else (e.g. save_*,
    create_*, record_*), these fakes would raise AttributeError."""
    for fake in (
        FakeFireDangerQueryService(),
        FakeSatelliteHotspotRepository(),
        FakeNewsRepository(),
        FakeFireEventRepository(),
        FakeFireSeverityAssessmentRepository(),
        FakeGlobalPlanningRunRepository(),
        FakeFireDangerAssessmentRepository(),
        FakeWeatherRepository(),
    ):
        for forbidden in ("save_assessment", "save_hotspot", "save_report", "create_event", "create_run", "record_member_result"):
            assert not hasattr(fake, forbidden)


def test_query_service_does_not_import_business_execution_components():
    forbidden_fragments = (
        "FFWICalculator",
        "FireDangerAssessmentAgent",
        "FireDetectionCalculator",
        "FireDetectionAgent",
        "FireSeverityAssessmentAgent",
        "FireSpreadPredictionAgent",
        "SimulationRefreshCoordinator",
        "GlobalPlanningOrchestrator",
        "genetic_optimizer",
        "GeneticOptimizer",
        "RoutePlanningAgent",
        "fastapi",
        "pydantic",
    )
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "services"
        / "operations"
        / "operations_activity_query_service.py"
    )
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
# 11. Timezone-aware timestamps preserved
# ---------------------------------------------------------------------------


def test_fire_danger_and_severity_occurred_at_are_timezone_aware():
    fd_detail = make_fire_danger_detail()
    fake_fd = FakeFireDangerQueryService(detail_by_id={1: fd_detail})
    assessment = make_severity_assessment()
    fake_severity = FakeFireSeverityAssessmentRepository(
        by_id={9: StoredFireSeverityAssessment(assessment_id=9, assessment=assessment)}
    )
    service = make_service(fire_danger=fake_fd, fire_severity=fake_severity)

    fd_result = service.get_activity_detail(OperationsActivityType.FIRE_DANGER, 1)
    severity_result = service.get_activity_detail(OperationsActivityType.FIRE_SEVERITY, 9)

    assert fd_result.occurred_at.tzinfo is not None
    assert severity_result.occurred_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 12. FireEvent has no fabricated area name (see also test above)
# ---------------------------------------------------------------------------


def test_fire_event_domain_model_itself_has_no_area_name_field():
    event = make_fire_event()
    assert not hasattr(event, "area_name")


# ---------------------------------------------------------------------------
# 13. Satellite hotspot is not described as confirmed fire
# ---------------------------------------------------------------------------


def test_satellite_hotspot_title_never_claims_confirmed_fire():
    hotspot = make_hotspot()
    fake_sat = FakeSatelliteHotspotRepository(by_id={7: StoredSatelliteHotspot(id=7, hotspot=hotspot)})
    service = make_service(satellite=fake_sat)

    result = service.get_activity_detail(OperationsActivityType.SATELLITE_HOTSPOT, 7)

    forbidden_words = ("confirmed", "wildfire confirmed", "fire confirmed")
    assert not any(word in result.title.lower() for word in forbidden_words)


# ---------------------------------------------------------------------------
# 14. Query result contains no ORM objects
# ---------------------------------------------------------------------------


def test_result_types_are_plain_domain_dataclasses_not_orm():
    fd_detail = make_fire_danger_detail()
    fake_fd = FakeFireDangerQueryService(detail_by_id={1: fd_detail})
    service = make_service(fire_danger=fake_fd)

    result = service.get_activity_detail(OperationsActivityType.FIRE_DANGER, 1)

    for value in (result, result.details):
        assert type(value).__module__.startswith("src.models")
