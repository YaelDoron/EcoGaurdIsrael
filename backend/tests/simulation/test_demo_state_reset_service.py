"""Tests for DemoStateResetService (Task A1.6).

Seeds one full, realistically-linked chain across every runtime table this
service deletes (evidence -> FireEvent -> Severity -> Spread -> Targets ->
Routes -> ResponsePlan -> GlobalPlanningRun -> ResourceCommitment), plus the
static/cache data it must never touch (FireStation/FirefightingResource
definitions, WeatherStation, GraphNode/GraphEdge), to prove the FK-safe
delete order, the STATIC/CACHE preservation, the resource-status baseline
restore, idempotency, and transactional rollback - all against the
project's normal SQLite test DB, never Neon.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.database.models.fire_danger_assessment_db import FireDangerAssessmentDB
from src.database.models.fire_danger_assessment_weather_input_db import (
    FireDangerAssessmentWeatherInputDB,
)
from src.database.models.fire_event_db import FireEventDB
from src.database.models.fire_event_news_evidence_db import FireEventNewsEvidenceDB
from src.database.models.fire_event_satellite_evidence_db import FireEventSatelliteEvidenceDB
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.database.models.fire_spread_prediction_cell_db import FireSpreadPredictionCellDB
from src.database.models.fire_spread_prediction_db import FireSpreadPredictionDB
from src.database.models.fire_spread_prediction_weather_input_db import (
    FireSpreadPredictionWeatherInputDB,
)
from src.database.models.fire_station_db import FireStationDB
from src.database.models.firefighting_resource_db import FirefightingResourceDB
from src.database.models.global_planning_run_db import GlobalPlanningRunDB
from src.database.models.global_planning_run_event_db import GlobalPlanningRunEventDB
from src.database.models.graph_edge_db import GraphEdgeDB
from src.database.models.graph_node_db import GraphNodeDB
from src.database.models.plan_comparison_db import PlanComparisonDB
from src.database.models.resource_commitment_db import ResourceCommitmentDB
from src.database.models.response_action_db import ResponseActionDB
from src.database.models.response_plan_db import ResponsePlanDB
from src.database.models.response_plan_planning_state_db import ResponsePlanPlanningStateDB
from src.database.models.response_plan_uncovered_target_db import ResponsePlanUncoveredTargetDB
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
from src.database.models.route_planning_run_db import RoutePlanningRunDB
from src.database.models.route_result_db import RouteResultDB
from src.database.models.satellite_hotspot_db import SatelliteHotspotDB
from src.database.models.weather_observation_db import WeatherObservationDB
from src.database.models.weather_station_db import WeatherStationDB
from src.database.models.wildfire_report_db import WildfireReportDB
from src.models.resource_status import ResourceStatus
from src.simulation import demo_state_reset_service as reset_module
from src.simulation.demo_state_reset_service import (
    DemoStateResetDisabledError,
    DemoStateResetService,
)

T = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def enable_reset(monkeypatch) -> None:
    monkeypatch.setattr(reset_module, "settings", SimpleNamespace(ENABLE_DEMO_DATA_RESET=True))


def disable_reset(monkeypatch) -> None:
    monkeypatch.setattr(reset_module, "settings", SimpleNamespace(ENABLE_DEMO_DATA_RESET=False))


def seed_full_chain(sqlite_session_factory) -> dict[str, int | str]:
    """Seed one fully-linked runtime chain plus static/cache data. Returns key ids.

    The static/cache portion (FireStationDB/FirefightingResourceDB/
    GraphNodeDB/WeatherStationDB) is get-or-create, matching how the real
    simulation generators upsert this exact reference data on every run
    (see DemoStateResetService's own docstring) - this lets a test call
    this helper TWICE in the same database (seeding "Run A", resetting, then
    seeding "Run B") without a spurious unique/primary-key collision on data
    that a reset legitimately never deletes. The runtime chain below always
    inserts fresh rows - a second call after a reset must never collide with
    the first call's (deleted) runtime data.
    """
    session = sqlite_session_factory()

    # --- Static / cache data (must survive every reset; get-or-create so
    # this helper is safe to call more than once against the same DB) ---
    if session.get(FireStationDB, "STATION-1") is None:
        session.add(FireStationDB(id="STATION-1", name="Station 1", latitude=32.7, longitude=35.0))
    if session.get(FirefightingResourceDB, "TRUCK-1") is None:
        session.add(
            FirefightingResourceDB(id="TRUCK-1", station_id="STATION-1", status=ResourceStatus.UNAVAILABLE)
        )
    if session.get(FirefightingResourceDB, "TRUCK-2") is None:
        session.add(
            FirefightingResourceDB(id="TRUCK-2", station_id="STATION-1", status=ResourceStatus.AVAILABLE)
        )
    if session.get(GraphNodeDB, 1001) is None:
        session.add(GraphNodeDB(id=1001, latitude=32.7, longitude=35.0))
    if session.get(GraphNodeDB, 1002) is None:
        session.add(GraphNodeDB(id=1002, latitude=32.71, longitude=35.01))
    session.flush()
    if session.query(GraphEdgeDB).filter_by(source_node_id=1001, target_node_id=1002).first() is None:
        session.add(
            GraphEdgeDB(source_node_id=1001, target_node_id=1002, distance_meters=500.0, travel_time_seconds=60.0)
        )

    # --- Runtime chain (always fresh rows) ---
    weather_station = session.query(WeatherStationDB).filter_by(external_station_id=900001).one_or_none()
    if weather_station is None:
        weather_station = WeatherStationDB(
            external_station_id=900001, name="SIM-TEST-01", latitude=32.7, longitude=35.0
        )
        session.add(weather_station)
        session.flush()
    observation = WeatherObservationDB(
        station_id=weather_station.id,
        timestamp=T,
        temperature=35.0,
        relative_humidity=15.0,
        wind_speed=25.0,
        wind_direction=180.0,
        wind_gust=30.0,
        rainfall=0.0,
    )
    session.add(observation)
    session.flush()

    danger_assessment = FireDangerAssessmentDB(
        area_id="area-1",
        area_name="Area 1",
        area_latitude=32.7,
        area_longitude=35.0,
        area_radius_km=5.0,
        assessed_at=T,
        status="valid",
        score=45.0,
        danger_level="very_high",
        methodology="FOSBERG_FFWI",
        methodology_version="1.0",
    )
    session.add(danger_assessment)
    session.flush()
    session.add(
        FireDangerAssessmentWeatherInputDB(
            assessment_id=danger_assessment.id,
            weather_observation_id=observation.id,
            station_id=weather_station.id,
        )
    )

    hotspot = SatelliteHotspotDB(
        detection_key="hotspot-1",
        latitude=32.7,
        longitude=35.0,
        detected_at=T,
        confidence="h",
        frp=50.0,
        brightness=320.0,
        satellite="SIM-NOAA-20",
        instrument="VIIRS",
        day_night="D",
    )
    report = WildfireReportDB(
        source_url="https://example.test/report-1",
        source_feed="Test Feed",
        title="Fire near Area 1",
        summary="Smoke observed.",
        location_name="Area 1",
        latitude=32.7,
        longitude=35.0,
        published_at=T,
        fetched_at=T,
    )
    session.add_all([hotspot, report])
    session.flush()

    fire_event = FireEventDB(
        latitude=32.7,
        longitude=35.0,
        detected_at=T,
        updated_at=T,
        status="confirmed",
        detection_confidence=0.9,
        methodology="TEST_DETECTION",
        methodology_version="1.0",
    )
    session.add(fire_event)
    session.flush()
    session.add(FireEventSatelliteEvidenceDB(fire_event_id=fire_event.id, satellite_hotspot_id=hotspot.id))
    session.add(FireEventNewsEvidenceDB(fire_event_id=fire_event.id, wildfire_report_id=report.id))

    severity = FireSeverityAssessmentDB(
        fire_event_id=fire_event.id,
        assessed_at=T,
        status="valid",
        score=70.0,
        severity_level="high",
        methodology="TEST_SEVERITY",
        methodology_version="1.0",
    )
    session.add(severity)
    session.flush()
    session.add(
        FireSeverityAssessmentWeatherInputDB(assessment_id=severity.id, weather_observation_id=observation.id)
    )
    session.add(
        FireSeverityAssessmentSatelliteInputDB(
            assessment_id=severity.id, satellite_hotspot_id=hotspot.id, selected_for_frp=True
        )
    )

    spread = FireSpreadPredictionDB(
        fire_event_id=fire_event.id,
        severity_assessment_id=severity.id,
        predicted_at=T,
        horizon_minutes=30,
        status="valid",
        methodology="TEST_SPREAD",
        methodology_version="1.0",
    )
    session.add(spread)
    session.flush()
    session.add(
        FireSpreadPredictionCellDB(
            prediction_id=spread.id,
            latitude=32.71,
            longitude=35.01,
            spread_probability=0.5,
            spread_risk_score=50.0,
            reached_step=1,
            reached_minutes=10,
        )
    )
    session.add(FireSpreadPredictionWeatherInputDB(prediction_id=spread.id, weather_observation_id=observation.id))

    target_set = ResponseTargetSetDB(
        fire_event_id=fire_event.id,
        generated_at=T,
        methodology="TEST_TARGETS",
        methodology_version="1.0",
    )
    session.add(target_set)
    session.flush()
    target = ResponseTargetDB(
        response_target_set_id=target_set.id,
        fire_event_id=fire_event.id,
        target_order=0,
        target_type="active_fire",
        latitude=32.7,
        longitude=35.0,
        priority_score=10.0,
    )
    session.add(target)
    session.flush()

    route_run = RoutePlanningRunDB(
        fire_event_id=fire_event.id,
        response_target_set_id=target_set.id,
        planned_at=T,
        methodology="TEST_ROUTING",
        methodology_version="1.0",
        resource_ids=["TRUCK-1"],
    )
    session.add(route_run)
    session.flush()
    route_result = RouteResultDB(
        route_planning_run_id=route_run.id,
        resource_id="TRUCK-1",
        response_target_id=target.id,
        status="reachable",
        source_node_id=1001,
        target_node_id=1002,
        node_path=[1001, 1002],
        distance_meters=500.0,
        travel_time_seconds=60.0,
    )
    session.add(route_result)
    session.flush()

    global_run = GlobalPlanningRunDB(
        started_at=T,
        completed_at=T,
        status="completed",
        trigger="weather_update",
        methodology="global_genetic_resource_allocation",
        methodology_version="1.0",
    )
    session.add(global_run)
    session.flush()

    plan = ResponsePlanDB(
        fire_event_id=fire_event.id,
        response_target_set_id=target_set.id,
        route_planning_run_id=route_run.id,
        global_planning_run_id=global_run.id,
        generated_at=T,
        status="activated",
        methodology="GENETIC_RESOURCE_ALLOCATION",
        methodology_version="1.0",
        random_seed=42,
        plan_score=80.0,
        coverage_score=90.0,
        average_eta_seconds=300.0,
    )
    session.add(plan)
    session.flush()

    session.add(
        ResponseActionDB(
            response_plan_id=plan.id,
            action_order=0,
            resource_id="TRUCK-1",
            response_target_id=target.id,
            route_result_id=route_result.id,
        )
    )
    session.add(
        ResponsePlanUncoveredTargetDB(response_plan_id=plan.id, target_order=0, response_target_id=target.id)
    )
    session.add(
        ResponsePlanPlanningStateDB(response_plan_id=plan.id, planning_effective_state_fingerprint="fingerprint-1")
    )
    session.add(
        PlanComparisonDB(
            fire_event_id=fire_event.id,
            optimized_plan_id=plan.id,
            route_planning_run_id=route_run.id,
            response_target_set_id=target_set.id,
            optimized_score=80.0,
            baseline_score=60.0,
            optimized_coverage_score=90.0,
            baseline_coverage_score=70.0,
            optimized_average_eta_seconds=300.0,
            baseline_average_eta_seconds=400.0,
            score_difference=20.0,
            improvement_percentage=25.0,
        )
    )
    session.add(
        GlobalPlanningRunEventDB(
            global_planning_run_id=global_run.id,
            fire_event_id=fire_event.id,
            event_order=0,
            result_status="activated",
            response_plan_id=plan.id,
        )
    )
    session.add(
        ResourceCommitmentDB(
            resource_id="TRUCK-1",
            fire_event_id=fire_event.id,
            response_plan_id=plan.id,
            committed_at=T,
            dispatch_state="dispatched",
        )
    )
    session.commit()
    fire_event_id = fire_event.id
    session.close()
    return {"fire_event_id": fire_event_id}


def count_all(sqlite_session_factory, model) -> int:
    session = sqlite_session_factory()
    count = session.query(model).count()
    session.close()
    return count


# ---------------------------------------------------------------------------
# 1. Reset disabled
# ---------------------------------------------------------------------------


def test_reset_disabled_refuses_and_deletes_nothing(sqlite_session_factory, monkeypatch):
    disable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = DemoStateResetService(sqlite_session_factory)

    with pytest.raises(DemoStateResetDisabledError):
        service.reset_demo_state()

    assert count_all(sqlite_session_factory, FireEventDB) == 1
    assert count_all(sqlite_session_factory, GlobalPlanningRunDB) == 1


# ---------------------------------------------------------------------------
# 2. FK-safe cleanup succeeds across the full chain
# ---------------------------------------------------------------------------


def test_fk_safe_cleanup_succeeds_across_full_chain(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = DemoStateResetService(sqlite_session_factory)

    result = service.reset_demo_state()  # must not raise (no FK violations)

    assert result.total_deleted > 0


# ---------------------------------------------------------------------------
# 3. Runtime data removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        WeatherObservationDB,
        FireDangerAssessmentDB,
        FireDangerAssessmentWeatherInputDB,
        SatelliteHotspotDB,
        WildfireReportDB,
        FireEventDB,
        FireEventSatelliteEvidenceDB,
        FireEventNewsEvidenceDB,
        FireSeverityAssessmentDB,
        FireSeverityAssessmentWeatherInputDB,
        FireSeverityAssessmentSatelliteInputDB,
        FireSpreadPredictionDB,
        FireSpreadPredictionCellDB,
        FireSpreadPredictionWeatherInputDB,
        ResponseTargetSetDB,
        ResponseTargetDB,
        RoutePlanningRunDB,
        RouteResultDB,
        ResponsePlanDB,
        ResponseActionDB,
        ResponsePlanUncoveredTargetDB,
        ResponsePlanPlanningStateDB,
        PlanComparisonDB,
        GlobalPlanningRunDB,
        GlobalPlanningRunEventDB,
        ResourceCommitmentDB,
    ],
)
def test_runtime_tables_are_empty_after_reset(sqlite_session_factory, monkeypatch, model):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    assert count_all(sqlite_session_factory, model) == 0


# ---------------------------------------------------------------------------
# 4. Static data preserved
# ---------------------------------------------------------------------------


def test_static_fire_station_and_resource_definitions_preserved(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    assert count_all(sqlite_session_factory, FireStationDB) == 1
    assert count_all(sqlite_session_factory, FirefightingResourceDB) == 2


def test_weather_station_definitions_preserved(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    assert count_all(sqlite_session_factory, WeatherStationDB) == 1


# ---------------------------------------------------------------------------
# 5. Road-network cache preserved
# ---------------------------------------------------------------------------


def test_road_network_cache_preserved(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    assert count_all(sqlite_session_factory, GraphNodeDB) == 2
    assert count_all(sqlite_session_factory, GraphEdgeDB) == 1


# ---------------------------------------------------------------------------
# 6. Resource operational state restored to baseline
# ---------------------------------------------------------------------------


def test_resource_status_restored_to_available_baseline(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)

    result = DemoStateResetService(sqlite_session_factory).reset_demo_state()

    session = sqlite_session_factory()
    statuses = {
        resource.id: resource.status
        for resource in session.query(FirefightingResourceDB).all()
    }
    session.close()

    assert statuses == {
        "TRUCK-1": ResourceStatus.AVAILABLE,
        "TRUCK-2": ResourceStatus.AVAILABLE,
    }
    assert result.resources_restored == 2


# ---------------------------------------------------------------------------
# 7. Idempotency
# ---------------------------------------------------------------------------


def test_reset_is_idempotent(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = DemoStateResetService(sqlite_session_factory)

    first = service.reset_demo_state()
    second = service.reset_demo_state()

    assert first.total_deleted > 0
    assert second.total_deleted == 0
    assert second.resources_restored == 2  # still restores status, even with nothing to delete


# ---------------------------------------------------------------------------
# 8. Transaction safety: failure mid-reset rolls back everything
# ---------------------------------------------------------------------------


def test_failure_mid_reset_rolls_back_completely(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = DemoStateResetService(sqlite_session_factory)

    def _boom(session):
        raise RuntimeError("simulated failure during resource baseline restore")

    monkeypatch.setattr(service, "_restore_resource_baseline", _boom)

    with pytest.raises(RuntimeError, match="simulated failure"):
        service.reset_demo_state()

    # Deletes happened earlier in the SAME uncommitted transaction as the
    # injected failure; rollback must undo them too, not just skip the commit.
    assert count_all(sqlite_session_factory, FireEventDB) == 1
    assert count_all(sqlite_session_factory, GlobalPlanningRunDB) == 1
    assert count_all(sqlite_session_factory, ResourceCommitmentDB) == 1

    session = sqlite_session_factory()
    resource = session.get(FirefightingResourceDB, "TRUCK-1")
    session.close()
    assert resource.status is ResourceStatus.UNAVAILABLE  # unchanged - restore never committed


# ---------------------------------------------------------------------------
# 9. Completeness guard (Task 5, Part 3): every mapped table is explicitly
# classified - this is what makes "audit all current runtime persistence
# tables" durable instead of a one-time manual check that silently rots as
# new tables/features are added.
# ---------------------------------------------------------------------------

_PRESERVED_TABLE_NAMES = frozenset(
    {
        FireStationDB.__tablename__,
        FirefightingResourceDB.__tablename__,
        WeatherStationDB.__tablename__,
        GraphNodeDB.__tablename__,
        GraphEdgeDB.__tablename__,
    }
)


def test_every_mapped_table_is_explicitly_classified_as_deleted_or_preserved():
    """A brand-new table that nobody added to _DELETE_ORDER_MODELS or the
    preserved set must fail this test immediately, rather than silently
    surviving a "clean" reset and leaking stale data into a new run."""
    import src.database.models  # noqa: F401 - ensure every model is registered on Base.metadata
    from src.database.base import Base

    deleted_table_names = {model.__tablename__ for model in reset_module._DELETE_ORDER_MODELS}
    classified = deleted_table_names | _PRESERVED_TABLE_NAMES
    all_table_names = set(Base.metadata.tables.keys())

    unclassified = all_table_names - classified
    assert unclassified == set(), (
        f"New table(s) {sorted(unclassified)} exist but are not classified in "
        "DemoStateResetService: add each one to either _DELETE_ORDER_MODELS "
        "(runtime demo data) or _PRESERVED_TABLE_NAMES (static/reference/cache)."
    )
    stale_classification = classified - all_table_names
    assert stale_classification == set(), (
        f"Classified table(s) {sorted(stale_classification)} no longer exist - "
        "remove the stale entry."
    )


# ---------------------------------------------------------------------------
# 10. GET-style reads never delete/reset anything
# ---------------------------------------------------------------------------


def make_overview_service(sqlite_session_factory):
    """Wire a full OperationsOverviewQueryService to the SAME sqlite
    session_factory as the reset service under test - mirrors the pattern
    already established in test_operations_overview_activity_feed.py, but
    kept local here since this file's concern is reset/isolation, not A6."""
    from src.repositories.fire_danger_assessment_repository import FireDangerAssessmentRepository
    from src.repositories.fire_event_repository import FireEventRepository
    from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
    from src.repositories.global_planning_run_repository import GlobalPlanningRunRepository
    from src.repositories.news_repository import NewsRepository
    from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
    from src.repositories.weather_repository import WeatherRepository
    from src.services.fire_danger.fire_danger_query_service import FireDangerQueryService
    from src.services.fire_event_read.active_fire_events_service import ActiveFireEventsService
    from src.services.operations.operations_overview_query_service import OperationsOverviewQueryService
    from src.services.simulation_control.simulation_run_manager import _idle_snapshot

    class FakeSimulationRunManager:
        def get_current_snapshot(self):
            return _idle_snapshot()

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


def test_getting_the_overview_repeatedly_never_deletes_or_resets_anything(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = make_overview_service(sqlite_session_factory)

    for _ in range(5):
        service.get_overview(as_of=T)

    assert count_all(sqlite_session_factory, FireEventDB) == 1
    assert count_all(sqlite_session_factory, GlobalPlanningRunDB) == 1
    assert count_all(sqlite_session_factory, SatelliteHotspotDB) == 1
    assert count_all(sqlite_session_factory, WildfireReportDB) == 1


# ---------------------------------------------------------------------------
# 11. End-to-end: Operations Overview is fully empty immediately after reset
# ---------------------------------------------------------------------------


def test_operations_overview_is_fully_empty_immediately_after_reset(sqlite_session_factory, monkeypatch):
    enable_reset(monkeypatch)
    seed_full_chain(sqlite_session_factory)
    service = make_overview_service(sqlite_session_factory)

    before = service.get_overview(as_of=T)
    assert before.activity_feed.items != ()
    assert before.active_fires != ()
    assert before.fire_danger_areas != ()
    # The old GlobalPlanningRun is visible pre-reset - this is what a stale
    # "View Response Plan" action would otherwise keep pointing to.
    assert any(item.activity_type.value == "global_planning_run" for item in before.activity_feed.items)

    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    after = service.get_overview(as_of=T)
    assert after.activity_feed.items == ()
    assert after.active_fires == ()
    assert after.fire_danger_areas == ()


# ---------------------------------------------------------------------------
# 12. Run isolation: Run A -> reset -> Run B never cross-contaminate
# ---------------------------------------------------------------------------


def test_reset_then_new_run_produces_only_current_run_data_no_cross_run_contamination(
    sqlite_session_factory, monkeypatch
):
    enable_reset(monkeypatch)
    run_a_ids = seed_full_chain(sqlite_session_factory)
    DemoStateResetService(sqlite_session_factory).reset_demo_state()

    run_b_ids = seed_full_chain(sqlite_session_factory)

    # Note: SQLite may legitimately RECYCLE a low integer rowid once a table
    # is fully empty (Postgres/Neon SERIAL sequences never do this - ids
    # only ever increase in production), so Run B's FireEvent id is not
    # asserted to differ from Run A's here. What actually matters for
    # isolation - and what the assertions below prove - is that exactly one
    # FireEvent exists after Run B and nothing about it traces back to
    # Run A's (fully deleted) evidence/severity/plan data.
    del run_a_ids

    session = sqlite_session_factory()
    fire_events = session.query(FireEventDB).all()
    hotspot_ids = {row.id for row in session.query(SatelliteHotspotDB).all()}
    report_ids = {row.id for row in session.query(WildfireReportDB).all()}
    evidence_satellite_ids = {row.satellite_hotspot_id for row in session.query(FireEventSatelliteEvidenceDB).all()}
    evidence_news_ids = {row.wildfire_report_id for row in session.query(FireEventNewsEvidenceDB).all()}
    severity_event_ids = {row.fire_event_id for row in session.query(FireSeverityAssessmentDB).all()}
    plan_ids = {row.id for row in session.query(ResponsePlanDB).all()}
    global_run_ids = {row.id for row in session.query(GlobalPlanningRunDB).all()}
    commitment_plan_ids = {row.response_plan_id for row in session.query(ResourceCommitmentDB).all()}
    session.close()

    # Exactly Run B's own FireEvent survives - Run A's was fully deleted.
    assert [event.id for event in fire_events] == [run_b_ids["fire_event_id"]]
    # Every evidence link references ONLY currently-persisted (Run B) rows -
    # never a satellite hotspot/news report id that belonged to Run A and no
    # longer exists.
    assert evidence_satellite_ids <= hotspot_ids
    assert evidence_news_ids <= report_ids
    # Severity attaches only to Run B's FireEvent.
    assert severity_event_ids == {run_b_ids["fire_event_id"]}
    # Exactly one GlobalPlanningRun/ResponsePlan/ResourceCommitment chain
    # exists (Run B's) - Run A's old ids are gone, never reused/referenced.
    assert len(plan_ids) == 1
    assert len(global_run_ids) == 1
    assert commitment_plan_ids <= plan_ids
