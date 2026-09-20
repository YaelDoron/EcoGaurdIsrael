"""Unit tests for OperationsOverviewQueryService's Activity Feed (Task A6, Part 31).

Uses REAL repositories backed by SQLite in-memory (see
test_operations_overview_query_service.py's own docstring for why).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.calculators.fire_danger.ffwi_config import FFWI_METHODOLOGY_NAME, FFWI_METHODOLOGY_VERSION
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models.fire_danger_assessment import FireDangerAssessment
from src.models.fire_danger_assessment_status import FireDangerAssessmentStatus
from src.models.fire_danger_level import FireDangerLevel
from src.models.fire_event import FireEvent
from src.models.fire_event_status import FireEventStatus
from src.models.fire_evidence_ref import FireEvidenceRef
from src.models.fire_evidence_type import FireEvidenceType
from src.models.fire_report import WildfireReport
from src.models.fire_severity_assessment import FireSeverityAssessment
from src.models.fire_severity_assessment_status import FireSeverityAssessmentStatus
from src.models.fire_severity_level import FireSeverityLevel
from src.models.operations_activity import OperationsActivityType
from src.models.operations_overview import (
    FireDangerActivityPreview,
    FireEventActivityPreview,
    FireSeverityActivityPreview,
    GlobalPlanningRunActivityPreview,
    NewsReportActivityPreview,
    SatelliteHotspotActivityPreview,
)
from src.models.satellite_hotspot import SatelliteHotspot
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
from src.repositories.news_repository import NewsRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
from src.services.operations.operations_activity_query_service import OperationsActivityQueryService
from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService

AS_OF = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


class FakeSimulationRunManager:
    def get_current_snapshot(self):
        from src.services.simulation_control.simulation_run_manager import _idle_snapshot

        return _idle_snapshot()


def make_service(sqlite_session_factory) -> OperationsOverviewQueryService:
    return OperationsOverviewQueryService(
        fire_danger_query_service=FireDangerQueryService(
            fire_danger_assessment_repository=FireDangerAssessmentRepository(
                session_factory=sqlite_session_factory
            ),
            weather_repository=WeatherRepository(session_factory=sqlite_session_factory),
        ),
        active_fire_events_service=ActiveFireEventsService(
            fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
            fire_severity_assessment_repository=FireSeverityAssessmentRepository(
                session_factory=sqlite_session_factory
            ),
        ),
        fire_danger_assessment_repository=FireDangerAssessmentRepository(session_factory=sqlite_session_factory),
        satellite_hotspot_repository=SatelliteHotspotRepository(session_factory=sqlite_session_factory),
        news_repository=NewsRepository(session_factory=sqlite_session_factory),
        weather_repository=WeatherRepository(session_factory=sqlite_session_factory),
        fire_event_repository=FireEventRepository(session_factory=sqlite_session_factory),
        fire_severity_assessment_repository=FireSeverityAssessmentRepository(
            session_factory=sqlite_session_factory
        ),
        global_planning_run_repository=GlobalPlanningRunRepository(session_factory=sqlite_session_factory),
        simulation_run_manager=FakeSimulationRunManager(),
    )


def persist_weather_observation(weather_repository: WeatherRepository, station_offset: int) -> tuple[int, int]:
    station = WeatherStation(
        external_station_id=930000 + station_offset, name=f"S{station_offset}", latitude=32.7, longitude=35.0
    )
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=AS_OF - timedelta(minutes=5),
            temperature=30.0,
            relative_humidity=25.0,
            wind_speed=20.0,
        )
    )
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.7, longitude=35.0, radius_km=5.0,
        start_time=AS_OF - timedelta(minutes=30), end_time=AS_OF,
    )
    record = next(c for c in candidates if c.observation.station_external_id == station.external_station_id)
    return record.observation_id, record.station_id


def save_fire_danger(repository, weather_repository, **overrides) -> int:
    observation_id, station_id = persist_weather_observation(weather_repository, len(overrides) + 1)
    values = dict(
        area_id="area-carmel", area_name="Carmel", area_latitude=32.731, area_longitude=35.046,
        area_radius_km=5.0, assessed_at=AS_OF, status=FireDangerAssessmentStatus.VALID, score=42.5,
        level=FireDangerLevel.VERY_HIGH, methodology=FFWI_METHODOLOGY_NAME, methodology_version=FFWI_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    saved = repository.save_assessment(FireDangerAssessment(**values), (observation_id,), (station_id,))
    return saved.assessment_id


def save_hotspot(repository, **overrides) -> int:
    values = dict(latitude=32.7, longitude=35.0, detected_at=AS_OF, confidence="h", frp=50.0, satellite="N20")
    values.update(overrides)
    repository.save_hotspot(SatelliteHotspot(**values))
    return repository.get_recent(50)[0].id


def save_fire_event(fire_event_repository, satellite_repository, **overrides) -> int:
    hotspot_id = save_hotspot(satellite_repository)
    values = dict(
        latitude=32.7, longitude=35.0, detected_at=AS_OF, updated_at=AS_OF, status=FireEventStatus.CONFIRMED,
        detection_confidence=0.8, methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    saved = fire_event_repository.create_event(
        FireEvent(**values), (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),)
    )
    return saved.id


def save_severity(repository, weather_repository, satellite_repository, fire_event_id, **overrides) -> int:
    satellite_hotspot_id = satellite_repository.get_recent(1)[0].id
    observation_id, _ = persist_weather_observation(weather_repository, fire_event_id + 100)
    values = dict(
        fire_event_id=fire_event_id, assessed_at=AS_OF, status=FireSeverityAssessmentStatus.VALID,
        score=60.0, level=FireSeverityLevel.HIGH, methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
    )
    values.update(overrides)
    saved = repository.save_assessment(
        FireSeverityAssessment(**values), weather_observation_ids=(observation_id,),
        satellite_hotspot_ids=(satellite_hotspot_id,), selected_frp_hotspot_id=satellite_hotspot_id,
    )
    return saved.assessment_id


def save_global_run(repository, fire_event_ids, **overrides) -> int:
    values = dict(
        started_at=AS_OF, trigger="manual", methodology="legacy_per_event_orchestration",
        methodology_version="1.0", input_fingerprint=None, fire_event_ids=fire_event_ids,
    )
    values.update(overrides)
    saved = repository.create_run(**values)
    return saved.id


# ---------------------------------------------------------------------------
# 1. Empty feed
# ---------------------------------------------------------------------------


def test_empty_feed(sqlite_session_factory):
    service = make_service(sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    assert snapshot.activity_feed.items == ()
    assert snapshot.activity_feed.limit == 30


# ---------------------------------------------------------------------------
# 2-7. One item per source type
# ---------------------------------------------------------------------------


def test_one_fire_danger_item(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    # save_fire_danger's default level (VERY_HIGH) also produces its own
    # weather_conditions signal - filter to FIRE_DANGER specifically, same
    # convention already used by the fire_event/severity/global_planning
    # "one item" tests below (whose own setup also incidentally creates a
    # second type).
    assessment_id = save_fire_danger(fd_repo, weather_repo)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    fire_danger_items = [item for item in items if item.activity_type is OperationsActivityType.FIRE_DANGER]
    assert len(fire_danger_items) == 1
    item = fire_danger_items[0]
    assert item.entity_id == assessment_id
    assert item.activity_id == f"fire_danger:{assessment_id}"
    assert isinstance(item.preview, FireDangerActivityPreview)


def test_one_satellite_item(sqlite_session_factory):
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    hotspot_id = save_hotspot(satellite_repo)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    assert len(items) == 1
    assert items[0].activity_type is OperationsActivityType.SATELLITE_HOTSPOT
    assert items[0].entity_id == hotspot_id
    assert isinstance(items[0].preview, SatelliteHotspotActivityPreview)


def test_one_news_item(sqlite_session_factory):
    news_repo = NewsRepository(session_factory=sqlite_session_factory)
    news_repo.save_report(
        WildfireReport(
            source_url="https://example.com/one", source_feed="Feed", title="Headline",
            summary="Summary", location_name="Carmel", latitude=32.7, longitude=35.0,
            published_at=AS_OF, fetched_at=AS_OF,
        )
    )
    report_id = news_repo.get_recent(10)[0].id
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    assert len(items) == 1
    assert items[0].activity_type is OperationsActivityType.NEWS_REPORT
    assert items[0].entity_id == report_id
    assert items[0].title == "Headline"
    assert isinstance(items[0].preview, NewsReportActivityPreview)


def test_one_fire_event_item(sqlite_session_factory):
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    event_id = save_fire_event(fire_event_repo, satellite_repo)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    fire_event_items = [item for item in items if item.activity_type is OperationsActivityType.FIRE_EVENT]
    assert len(fire_event_items) == 1
    assert fire_event_items[0].entity_id == event_id
    assert fire_event_items[0].title == f"Fire Event #{event_id}"
    assert isinstance(fire_event_items[0].preview, FireEventActivityPreview)


def test_one_severity_item(sqlite_session_factory):
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    severity_repo = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    event_id = save_fire_event(fire_event_repo, satellite_repo)
    assessment_id = save_severity(severity_repo, weather_repo, satellite_repo, event_id)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    severity_items = [item for item in items if item.activity_type is OperationsActivityType.FIRE_SEVERITY]
    assert len(severity_items) == 1
    assert severity_items[0].entity_id == assessment_id
    assert isinstance(severity_items[0].preview, FireSeverityActivityPreview)
    assert severity_items[0].preview.fire_event_id == event_id


def test_one_global_planning_run_item(sqlite_session_factory):
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    gp_repo = GlobalPlanningRunRepository(session_factory=sqlite_session_factory)
    event_id = save_fire_event(fire_event_repo, satellite_repo)
    run_id = save_global_run(gp_repo, [event_id])
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    run_items = [item for item in items if item.activity_type is OperationsActivityType.GLOBAL_PLANNING_RUN]
    assert len(run_items) == 1
    assert run_items[0].entity_id == run_id
    assert isinstance(run_items[0].preview, GlobalPlanningRunActivityPreview)
    assert run_items[0].preview.fire_event_count == 1


def test_one_weather_conditions_item(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    assessment_id = save_fire_danger(fd_repo, weather_repo, level=FireDangerLevel.HIGH)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    weather_items = [item for item in items if item.activity_type is OperationsActivityType.WEATHER_CONDITIONS]
    assert len(weather_items) == 1
    assert weather_items[0].entity_id == assessment_id
    assert weather_items[0].activity_id == f"weather_conditions:{assessment_id}"
    assert weather_items[0].preview.area_name == "Carmel"
    assert weather_items[0].preview.temperature_c == 30.0


def test_weather_conditions_item_also_appears_for_moderate_level(sqlite_session_factory):
    """Part K: MODERATE now also gets a Weather Conditions signal - only
    INSUFFICIENT_DATA (no valid level at all) is excluded."""
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo, level=FireDangerLevel.MODERATE)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    weather_items = [item for item in items if item.activity_type is OperationsActivityType.WEATHER_CONDITIONS]
    assert len(weather_items) == 1
    assert weather_items[0].preview.fire_danger_level is FireDangerLevel.MODERATE


# ---------------------------------------------------------------------------
# Task 4: occurred_at (source time) vs available_at (real persistence time)
# ---------------------------------------------------------------------------


def test_fire_danger_occurred_at_is_assessed_at_available_at_is_real_creation_time(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    old_assessed_at = AS_OF - timedelta(days=1)
    save_fire_danger(fd_repo, weather_repo, assessed_at=old_assessed_at)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items
    fire_danger_items = [item for item in items if item.activity_type is OperationsActivityType.FIRE_DANGER]
    assert len(fire_danger_items) == 1
    item = fire_danger_items[0]

    assert item.occurred_at == old_assessed_at
    # available_at is the row's own real DB-insert time (server-assigned at
    # save time, i.e. roughly "now") - a full day newer than the domain
    # assessed_at this test deliberately backdated.
    assert item.available_at > old_assessed_at + timedelta(hours=1)
    assert item.available_at.tzinfo is not None


def test_weather_conditions_occurred_at_stays_source_time_available_at_uses_assessment_creation(
    sqlite_session_factory,
):
    """Part D/H: Weather's occurred_at remains the contributing observation's
    own timestamp (never overwritten); its available_at is the SAME
    assessment's real creation time - shared with the paired FIRE_DANGER item."""
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    old_assessed_at = AS_OF - timedelta(days=1)
    assessment_id = save_fire_danger(fd_repo, weather_repo, level=FireDangerLevel.HIGH, assessed_at=old_assessed_at)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items
    weather_item = next(item for item in items if item.activity_type is OperationsActivityType.WEATHER_CONDITIONS)
    fire_danger_item = next(item for item in items if item.activity_type is OperationsActivityType.FIRE_DANGER)
    assert weather_item.entity_id == assessment_id

    # Source time is the contributing observation's own timestamp (from
    # persist_weather_observation: AS_OF - 5 minutes), NOT the backdated
    # assessed_at and NOT any insertion time.
    assert weather_item.occurred_at == AS_OF - timedelta(minutes=5)
    # available_at is shared exactly with the paired FIRE_DANGER item -
    # both are projections of the one same persisted assessment.
    assert weather_item.available_at == fire_danger_item.available_at
    assert weather_item.available_at > old_assessed_at + timedelta(hours=1)


def test_weather_conditions_sorts_before_fire_danger_on_exact_available_at_tie(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo, level=FireDangerLevel.HIGH)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items
    types_in_order = [item.activity_type for item in items]

    weather_index = types_in_order.index(OperationsActivityType.WEATHER_CONDITIONS)
    fire_danger_index = types_in_order.index(OperationsActivityType.FIRE_DANGER)
    assert weather_index < fire_danger_index


def test_satellite_occurred_at_stays_detected_at_available_at_is_real_persisted_time(sqlite_session_factory):
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    old_detected_at = AS_OF - timedelta(days=1)
    hotspot_id = save_hotspot(satellite_repo, detected_at=old_detected_at)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items
    assert len(items) == 1
    item = items[0]
    assert item.entity_id == hotspot_id

    assert item.occurred_at == old_detected_at
    # available_at is the NEW, real DB-insert timestamp - never derived from
    # detected_at, and (since this row was backdated by a day) clearly newer.
    assert item.available_at != item.occurred_at
    assert item.available_at > old_detected_at + timedelta(hours=1)
    assert item.available_at.tzinfo is not None


def test_news_occurred_at_stays_published_at_available_at_is_fetched_at(sqlite_session_factory):
    news_repo = NewsRepository(session_factory=sqlite_session_factory)
    published_at = AS_OF - timedelta(hours=2)
    fetched_at = AS_OF - timedelta(minutes=1)
    news_repo.save_report(
        WildfireReport(
            source_url="https://example.com/available-at", source_feed="Feed", title="Headline",
            summary="s", location_name=None, latitude=None, longitude=None,
            published_at=published_at, fetched_at=fetched_at,
        )
    )
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items
    assert len(items) == 1
    item = items[0]

    assert item.occurred_at == published_at
    assert item.available_at == fetched_at
    assert item.available_at != item.occurred_at


def test_repeated_query_never_changes_available_at(sqlite_session_factory):
    """Reading the overview twice (simulating a browser refresh/poll) must
    never move any item's available_at - it is real persisted state, not
    recomputed per-request."""
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)
    service = make_service(sqlite_session_factory)

    first = {item.activity_id: item.available_at for item in service.get_overview(as_of=AS_OF).activity_feed.items}
    second = {item.activity_id: item.available_at for item in service.get_overview(as_of=AS_OF).activity_feed.items}

    assert first == second


# ---------------------------------------------------------------------------
# 8/9. All six interleaved -> correct global DESC order + deterministic ties
# ---------------------------------------------------------------------------


def test_all_six_types_interleaved_sort_by_available_at_desc(sqlite_session_factory):
    """Task 4, Part L: the feed sorts by `available_at` (real persistence
    order), not by each type's own domain/source `occurred_at`. Every
    `save_*` helper below is deliberately given DECREASING domain timestamps
    (fire_danger's t0 is the OLDEST source time, global_run's t0+50min the
    NEWEST) while being inserted in the OPPOSITE (increasing) real order -
    if the feed still sorted by occurred_at, fire_danger's assessed_at=t0
    would win; because it sorts by available_at (each row's own real
    DB-insert time, which strictly increases with insertion order here), the
    LAST-inserted row (global_planning_run) must come first instead.
    """
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repo = NewsRepository(session_factory=sqlite_session_factory)
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    severity_repo = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    gp_repo = GlobalPlanningRunRepository(session_factory=sqlite_session_factory)

    t0 = AS_OF - timedelta(minutes=50)
    # Source/domain times DESCEND with insertion order (oldest source time
    # inserted first) - the inverse of what a naive occurred_at-based sort
    # would need to put global_planning_run first.
    save_fire_danger(fd_repo, weather_repo, assessed_at=t0)
    save_hotspot(satellite_repo, detected_at=t0 - timedelta(minutes=10))
    news_repo.save_report(
        WildfireReport(
            source_url="https://example.com/interleave", source_feed="Feed", title="Headline",
            summary="s", location_name=None, latitude=None, longitude=None,
            published_at=t0 - timedelta(minutes=20), fetched_at=t0 - timedelta(minutes=20),
        )
    )
    event_id = save_fire_event(fire_event_repo, satellite_repo, detected_at=t0 - timedelta(minutes=30), updated_at=t0 - timedelta(minutes=30))
    save_severity(severity_repo, weather_repo, satellite_repo, event_id, assessed_at=t0 - timedelta(minutes=40))
    save_global_run(gp_repo, [event_id], started_at=t0 - timedelta(minutes=50))

    service = make_service(sqlite_session_factory)
    items = service.get_overview(as_of=AS_OF).activity_feed.items

    available_ats = [
        item.available_at if item.available_at.tzinfo else item.available_at.replace(tzinfo=timezone.utc)
        for item in items
    ]
    assert available_ats == sorted(available_ats, reverse=True)
    # global_planning_run was inserted LAST (highest real available_at)
    # despite having the OLDEST source/domain timestamp (started_at=t0-50min).
    assert [item.activity_type for item in items][0] is OperationsActivityType.GLOBAL_PLANNING_RUN


def test_timestamp_ties_are_deterministic(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo, area_id="area-a", assessed_at=AS_OF)
    save_fire_danger(fd_repo, weather_repo, area_id="area-b", assessed_at=AS_OF)
    service = make_service(sqlite_session_factory)

    first = [item.activity_id for item in service.get_overview(as_of=AS_OF).activity_feed.items]
    second = [item.activity_id for item in service.get_overview(as_of=AS_OF).activity_feed.items]

    assert first == second
    # tie-break within the FIRE_DANGER type: entity_id DESC. (Both
    # assessments' default VERY_HIGH level also each produce their own
    # weather_conditions item, tied on the same timestamp too, but that is a
    # separate type/tie-group - activity_type ASC keeps "fire_danger" before
    # "weather_conditions" regardless, so it does not interact with this
    # specific same-type tie-break being tested here.)
    fire_danger_ids = [activity_id for activity_id in first if activity_id.startswith("fire_danger:")]
    assert fire_danger_ids == sorted(fire_danger_ids, key=lambda activity_id: -int(activity_id.split(":")[1]))


# ---------------------------------------------------------------------------
# 10. Global limit enforced (not per-type)
# ---------------------------------------------------------------------------


def test_global_limit_enforced_across_all_types(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    for index in range(5):
        save_fire_danger(fd_repo, weather_repo, area_id=f"area-{index}", assessed_at=AS_OF - timedelta(minutes=index))
    for index in range(5):
        save_hotspot(satellite_repo, detected_at=AS_OF - timedelta(minutes=index))

    service = make_service(sqlite_session_factory)
    items = service.get_overview(as_of=AS_OF, activity_limit=4).activity_feed.items

    assert len(items) == 4


# ---------------------------------------------------------------------------
# 11. Bounded per-source reads (no full table scans)
# ---------------------------------------------------------------------------


def test_global_planning_candidates_use_one_batched_member_count_query_not_n_plus_1(
    sqlite_session_factory, monkeypatch
):
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    gp_repo = GlobalPlanningRunRepository(session_factory=sqlite_session_factory)
    for index in range(3):
        event_id = save_fire_event(fire_event_repo, satellite_repo, detected_at=AS_OF - timedelta(minutes=index), updated_at=AS_OF - timedelta(minutes=index))
        save_global_run(gp_repo, [event_id], started_at=AS_OF - timedelta(minutes=index))

    get_members_calls = []
    original_get_members = GlobalPlanningRunRepository.get_members

    def counted_get_members(self, run_id):
        get_members_calls.append(run_id)
        return original_get_members(self, run_id)

    monkeypatch.setattr(GlobalPlanningRunRepository, "get_members", counted_get_members)

    get_member_counts_calls = []
    original_get_member_counts = GlobalPlanningRunRepository.get_member_counts

    def counted_get_member_counts(self, run_ids):
        ids = tuple(run_ids)
        get_member_counts_calls.append(ids)
        return original_get_member_counts(self, ids)

    monkeypatch.setattr(GlobalPlanningRunRepository, "get_member_counts", counted_get_member_counts)

    service = make_service(sqlite_session_factory)
    items = service.get_overview(as_of=AS_OF).activity_feed.items

    run_items = [item for item in items if item.activity_type is OperationsActivityType.GLOBAL_PLANNING_RUN]
    assert len(run_items) == 3
    assert get_members_calls == []  # never the per-run method
    assert len(get_member_counts_calls) == 1  # exactly one batched call
    assert len(get_member_counts_calls[0]) == 3


def test_bounded_per_source_reads_not_full_table_scans(sqlite_session_factory, monkeypatch):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    for index in range(10):
        save_fire_danger(fd_repo, weather_repo, area_id=f"area-{index}", assessed_at=AS_OF - timedelta(minutes=index))

    calls = []
    original = FireDangerAssessmentRepository.get_recent

    def counted(self, limit):
        calls.append(limit)
        return original(self, limit)

    monkeypatch.setattr(FireDangerAssessmentRepository, "get_recent", counted)

    service = make_service(sqlite_session_factory)
    service.get_overview(as_of=AS_OF, activity_limit=3)

    assert calls == [3]


# ---------------------------------------------------------------------------
# 12. Feed does not call A5 detail service per item
# ---------------------------------------------------------------------------


def test_feed_does_not_call_a5_detail_service(sqlite_session_factory, monkeypatch):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)

    def boom(self, activity_type, entity_id):
        raise AssertionError("Activity Feed must not call the A5 detail service per item")

    monkeypatch.setattr(OperationsActivityQueryService, "get_activity_detail", boom)

    service = make_service(sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    # 2 items: the fire_danger assessment itself plus its own
    # weather_conditions signal (save_fire_danger's default level is
    # VERY_HIGH) - the assertion under test (no A5 call per item) is
    # unaffected by the exact count.
    assert len(snapshot.activity_feed.items) == 2


# ---------------------------------------------------------------------------
# 13/14. activity_id is a globally unambiguous composite; every item carries both fields
# ---------------------------------------------------------------------------


def test_activity_id_is_unambiguous_composite(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)  # assessment_id will be 1
    save_hotspot(satellite_repo)  # hotspot_id will be 1 too - same raw id, different table

    service = make_service(sqlite_session_factory)
    items = service.get_overview(as_of=AS_OF).activity_feed.items

    activity_ids = [item.activity_id for item in items]
    assert len(activity_ids) == len(set(activity_ids))
    for item in items:
        assert item.activity_id == f"{item.activity_type.value}:{item.entity_id}"


# ---------------------------------------------------------------------------
# 15. Every feed item resolves through A5's detail service
# ---------------------------------------------------------------------------


def test_every_feed_item_resolves_through_a5_detail_service(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    news_repo = NewsRepository(session_factory=sqlite_session_factory)
    fire_event_repo = FireEventRepository(session_factory=sqlite_session_factory)
    severity_repo = FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)
    gp_repo = GlobalPlanningRunRepository(session_factory=sqlite_session_factory)

    save_fire_danger(fd_repo, weather_repo)
    save_hotspot(satellite_repo)
    news_repo.save_report(
        WildfireReport(
            source_url="https://example.com/resolve", source_feed="Feed", title="Headline",
            summary="s", location_name=None, latitude=None, longitude=None,
            published_at=AS_OF, fetched_at=AS_OF,
        )
    )
    event_id = save_fire_event(fire_event_repo, satellite_repo)
    save_severity(severity_repo, weather_repo, satellite_repo, event_id)
    save_global_run(gp_repo, [event_id])

    overview_service = make_service(sqlite_session_factory)
    activity_service = OperationsActivityQueryService(
        fire_danger_query_service=FireDangerQueryService(
            fire_danger_assessment_repository=FireDangerAssessmentRepository(session_factory=sqlite_session_factory),
            weather_repository=weather_repo,
        ),
        satellite_hotspot_repository=satellite_repo,
        news_repository=news_repo,
        fire_event_repository=fire_event_repo,
        fire_severity_assessment_repository=severity_repo,
        global_planning_run_repository=gp_repo,
        fire_danger_assessment_repository=fd_repo,
        weather_repository=weather_repo,
    )

    items = overview_service.get_overview(as_of=AS_OF).activity_feed.items
    # 7: the original 6 sources plus save_fire_danger's default VERY_HIGH
    # level also producing its own weather_conditions signal.
    assert len(items) == 7
    for item in items:
        detail = activity_service.get_activity_detail(item.activity_type, item.entity_id)
        assert detail is not None
        assert detail.entity_id == item.entity_id


# ---------------------------------------------------------------------------
# 16. No "system_update" type exists
# ---------------------------------------------------------------------------


def test_no_system_update_activity_type_exists():
    assert not any(member.value == "system_update" for member in OperationsActivityType)
    assert not any("system" in member.value for member in OperationsActivityType)


# ---------------------------------------------------------------------------
# 17. No simulation progress tick appears as a feed item
# ---------------------------------------------------------------------------


def test_no_simulation_progress_tick_in_feed(sqlite_session_factory):
    service = make_service(sqlite_session_factory)

    snapshot = service.get_overview(as_of=AS_OF)

    for item in snapshot.activity_feed.items:
        assert "simulation" not in item.activity_type.value


# ---------------------------------------------------------------------------
# 18. Repeated unchanged query -> stable item order/content
# ---------------------------------------------------------------------------


def test_repeated_query_is_stable(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)
    service = make_service(sqlite_session_factory)

    first = service.get_overview(as_of=AS_OF)
    second = service.get_overview(as_of=AS_OF)

    assert first.activity_feed.items == second.activity_feed.items


# ---------------------------------------------------------------------------
# 19-25. Satellite hotspot location_name enrichment (read/presentation only)
# ---------------------------------------------------------------------------


def test_satellite_hotspot_gets_the_real_persisted_area_name_when_inside_a_known_area(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    save_fire_danger(
        fd_repo, weather_repo, area_id="area-carmel", area_name="Carmel Demo Area",
        area_latitude=32.731, area_longitude=35.046, area_radius_km=5.0,
    )
    # ~1.1km from the area center - comfortably inside its 5km radius.
    save_hotspot(satellite_repo, latitude=32.74, longitude=35.05)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    satellite_items = [item for item in items if item.activity_type is OperationsActivityType.SATELLITE_HOTSPOT]
    assert len(satellite_items) == 1
    assert satellite_items[0].preview.location_name == "Carmel Demo Area"


def test_satellite_hotspot_gets_no_location_name_when_outside_every_known_area(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    save_fire_danger(
        fd_repo, weather_repo, area_id="area-carmel", area_name="Carmel Demo Area",
        area_latitude=32.731, area_longitude=35.046, area_radius_km=5.0,
    )
    # Far outside the 5km Carmel circle - roughly 200km+ away.
    save_hotspot(satellite_repo, latitude=31.0, longitude=34.8)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    satellite_items = [item for item in items if item.activity_type is OperationsActivityType.SATELLITE_HOTSPOT]
    assert len(satellite_items) == 1
    assert satellite_items[0].preview.location_name is None


def test_satellite_hotspot_never_uses_the_merely_nearest_area_when_outside_its_radius(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    # A small-radius area whose circle does NOT reach the hotspot, even
    # though it is the geographically nearest known area.
    save_fire_danger(
        fd_repo, weather_repo, area_id="area-tiny", area_name="Tiny Area",
        area_latitude=31.0, area_longitude=34.8, area_radius_km=1.0,
    )
    save_hotspot(satellite_repo, latitude=31.0, longitude=34.8 + 0.5)  # ~47km away, outside the 1km radius
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    satellite_items = [item for item in items if item.activity_type is OperationsActivityType.SATELLITE_HOTSPOT]
    assert satellite_items[0].preview.location_name is None


def test_satellite_hotspot_resolves_deterministically_when_multiple_areas_contain_the_point(sqlite_session_factory):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    # Two overlapping large-radius areas both contain the hotspot - the
    # nearer center (area-near) must win, not the first/last saved.
    save_fire_danger(
        fd_repo, weather_repo, area_id="area-far", area_name="Far Area",
        area_latitude=32.9, area_longitude=35.2, area_radius_km=50.0,
    )
    save_fire_danger(
        fd_repo, weather_repo, area_id="area-near", area_name="Near Area",
        area_latitude=32.71, area_longitude=35.02, area_radius_km=50.0,
    )
    save_hotspot(satellite_repo, latitude=32.7, longitude=35.0)
    service = make_service(sqlite_session_factory)

    items = service.get_overview(as_of=AS_OF).activity_feed.items

    satellite_items = [item for item in items if item.activity_type is OperationsActivityType.SATELLITE_HOTSPOT]
    assert satellite_items[0].preview.location_name == "Near Area"


def test_satellite_location_enrichment_reuses_the_already_fetched_areas_no_extra_query(
    sqlite_session_factory, monkeypatch
):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)
    for index in range(5):
        save_hotspot(satellite_repo, detected_at=AS_OF - timedelta(minutes=index))

    calls = []
    original = FireDangerQueryService.get_latest_for_all_areas

    def counted(self, *, as_of):
        calls.append(as_of)
        return original(self, as_of=as_of)

    monkeypatch.setattr(FireDangerQueryService, "get_latest_for_all_areas", counted)

    service = make_service(sqlite_session_factory)
    service.get_overview(as_of=AS_OF)

    # Exactly one call regardless of how many satellite hotspot candidates
    # needed location resolution - the areas are fetched once and reused.
    assert len(calls) == 1


def test_satellite_location_enrichment_performs_no_database_writes(sqlite_session_factory, monkeypatch):
    fd_repo = FireDangerAssessmentRepository(session_factory=sqlite_session_factory)
    weather_repo = WeatherRepository(session_factory=sqlite_session_factory)
    satellite_repo = SatelliteHotspotRepository(session_factory=sqlite_session_factory)
    save_fire_danger(fd_repo, weather_repo)
    save_hotspot(satellite_repo)

    def boom(*args, **kwargs):
        raise AssertionError("Activity Feed assembly must never write to the database")

    monkeypatch.setattr(FireDangerAssessmentRepository, "save_assessment", boom)
    monkeypatch.setattr(SatelliteHotspotRepository, "save_hotspot", boom)

    service = make_service(sqlite_session_factory)
    snapshot = service.get_overview(as_of=AS_OF)

    # 3: fire_danger + its own weather_conditions signal (VERY_HIGH default) + satellite.
    assert len(snapshot.activity_feed.items) == 3
