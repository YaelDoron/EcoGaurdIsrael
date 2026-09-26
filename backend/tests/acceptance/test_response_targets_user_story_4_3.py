"""Acceptance tests for User Story 4.3: prioritized wildfire response targets."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agents.analysis import ResponseTargetGenerationAgent, ResponseTargetGenerationStatus
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.calculators.response_target import ResponseTargetCalculator
from src.calculators.response_target.response_target_config import (
    ACTIVE_FIRE_BASE_PRIORITY,
    MIN_PREDICTED_TARGET_RISK_SCORE,
    PREDICTION_HORIZON_FACTORS,
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
    TARGET_DEDUP_DISTANCE_METERS,
)
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadPrediction,
    FireSpreadPredictionCell,
    FireSpreadPredictionStatus,
    ResponseTargetType,
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.response_target_repository import ResponseTargetRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.response_target import ResponseTargetInputService

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
CARMEL_LATITUDE = 32.731
CARMEL_LONGITUDE = 35.046
GOLAN_LATITUDE = 33.127
GOLAN_LONGITUDE = 35.781


@pytest.fixture
def stack(sqlite_session_factory):
    fire_event_repository = FireEventRepository(sqlite_session_factory)
    severity_repository = FireSeverityAssessmentRepository(sqlite_session_factory)
    spread_repository = FireSpreadPredictionRepository(sqlite_session_factory)
    target_repository = ResponseTargetRepository(sqlite_session_factory)
    satellite_repository = SatelliteHotspotRepository(sqlite_session_factory)
    weather_repository = WeatherRepository(sqlite_session_factory)
    agent = ResponseTargetGenerationAgent(
        input_service=ResponseTargetInputService(
            fire_event_repository=fire_event_repository,
            fire_severity_assessment_repository=severity_repository,
            fire_spread_prediction_repository=spread_repository,
        ),
        calculator=ResponseTargetCalculator(),
        repository=target_repository,
    )
    return {
        "agent": agent,
        "fire_events": fire_event_repository,
        "severity": severity_repository,
        "spread": spread_repository,
        "targets": target_repository,
        "satellite": satellite_repository,
        "weather": weather_repository,
    }


def test_at1_active_fire_location_becomes_active_fire_target(stack):
    fire_event_id = persist_fire_event(stack, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)
    stored = stack["targets"].get_by_id(result.target_set_id)

    assert result.status is ResponseTargetGenerationStatus.GENERATED
    assert result.target_count == 1
    active_targets = [target for target in result.targets if target.target_type is ResponseTargetType.ACTIVE_FIRE]
    assert len(active_targets) == 1
    active = active_targets[0]
    assert active.fire_event_id == fire_event_id
    assert active.latitude == CARMEL_LATITUDE
    assert active.longitude == CARMEL_LONGITUDE
    assert active.prediction_horizon_minutes is None
    assert active.spread_prediction_id is None
    assert active.spread_prediction_cell_id is None
    assert stored.target_set.fire_event_id == fire_event_id
    assert stored.target_set.targets == result.targets


def test_at2_high_spread_prediction_becomes_predicted_risk_target_with_traceability(stack):
    fire_event_id = persist_fire_event(stack)
    stored_prediction = persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(make_cell(latitude=32.760, longitude=35.080, risk_score=MIN_PREDICTED_TARGET_RISK_SCORE + 10.0),),
    )
    source_cell = stack["spread"].get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF).cells[0]

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    predicted_targets = [target for target in result.targets if target.target_type is ResponseTargetType.PREDICTED_RISK]
    assert len(predicted_targets) == 1
    predicted = predicted_targets[0]
    assert predicted.fire_event_id == fire_event_id
    assert predicted.latitude == source_cell.cell.latitude
    assert predicted.longitude == source_cell.cell.longitude
    assert predicted.prediction_horizon_minutes == 30
    assert predicted.spread_prediction_id == stored_prediction.id
    assert predicted.spread_prediction_cell_id == source_cell.cell_id


def test_at3_active_fire_and_predicted_risk_targets_are_distinguishable(stack):
    fire_event_id = persist_fire_event(stack)
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(make_cell(latitude=32.760, longitude=35.080, risk_score=82.0),),
    )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    active = next(target for target in result.targets if target.target_type is ResponseTargetType.ACTIVE_FIRE)
    predicted = next(target for target in result.targets if target.target_type is ResponseTargetType.PREDICTED_RISK)
    assert active.target_type is not predicted.target_type
    assert active.prediction_horizon_minutes is None
    assert active.spread_prediction_id is None
    assert active.spread_prediction_cell_id is None
    assert predicted.prediction_horizon_minutes == 30
    assert predicted.spread_prediction_id is not None
    assert predicted.spread_prediction_cell_id is not None


def test_at4_higher_priority_targets_appear_first_and_target_order_is_persisted(stack):
    fire_event_id = persist_fire_event(stack)
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(make_cell(latitude=32.760, longitude=35.080, risk_score=80.0),),
    )
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=60,
        cells=(make_cell(latitude=32.775, longitude=35.095, risk_score=80.0),),
    )
    persist_severity(stack, fire_event_id, score=70.0, assessed_at=AS_OF - timedelta(minutes=1))

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)
    stored = stack["targets"].get_by_id(result.target_set_id)

    assert [target.priority_score for target in result.targets] == [
        ACTIVE_FIRE_BASE_PRIORITY + 70.0,
        80.0 * PREDICTION_HORIZON_FACTORS[30],
        80.0 * PREDICTION_HORIZON_FACTORS[60],
    ]
    assert [target.priority_score for target in result.targets] == sorted(
        (target.priority_score for target in result.targets),
        reverse=True,
    )
    assert [stored_target.target_order for stored_target in stored.targets] == [0, 1, 2]
    assert tuple(stored_target.target for stored_target in stored.targets) == result.targets


def test_at5_below_threshold_predictions_are_excluded_and_threshold_is_inclusive(stack):
    fire_event_id = persist_fire_event(stack)
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(
            make_cell(latitude=32.760, longitude=35.080, risk_score=MIN_PREDICTED_TARGET_RISK_SCORE - 0.01),
            make_cell(latitude=32.775, longitude=35.095, risk_score=MIN_PREDICTED_TARGET_RISK_SCORE),
        ),
    )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    predicted_targets = [target for target in result.targets if target.target_type is ResponseTargetType.PREDICTED_RISK]
    assert len(predicted_targets) == 1
    assert predicted_targets[0].latitude == 32.775
    assert predicted_targets[0].priority_score == MIN_PREDICTED_TARGET_RISK_SCORE


def test_at6_duplicate_predicted_locations_are_deduplicated_with_best_candidate_retained(stack):
    fire_event_id = persist_fire_event(stack)
    lower_priority_prediction = persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=60,
        cells=(make_cell(latitude=32.76000, longitude=35.08000, risk_score=95.0),),
    )
    higher_priority_prediction = persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(make_cell(latitude=32.76020, longitude=35.08020, risk_score=90.0),),
    )
    higher_priority_cell = stack["spread"].get_latest_for_event_and_horizon_as_of(fire_event_id, 30, AS_OF).cells[0]

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    predicted_targets = [target for target in result.targets if target.target_type is ResponseTargetType.PREDICTED_RISK]
    assert len(predicted_targets) == 1
    retained = predicted_targets[0]
    assert retained.spread_prediction_id == higher_priority_prediction.id
    assert retained.spread_prediction_id != lower_priority_prediction.id
    assert retained.spread_prediction_cell_id == higher_priority_cell.cell_id
    assert retained.priority_score == 90.0


def test_at7_fire_event_isolation_prevents_cross_event_predictions_and_targets(stack):
    carmel_event_id = persist_fire_event(stack, latitude=CARMEL_LATITUDE, longitude=CARMEL_LONGITUDE)
    golan_event_id = persist_fire_event(stack, latitude=GOLAN_LATITUDE, longitude=GOLAN_LONGITUDE)
    persist_spread_prediction(
        stack,
        carmel_event_id,
        horizon_minutes=30,
        latitude=CARMEL_LATITUDE,
        longitude=CARMEL_LONGITUDE,
        cells=(make_cell(latitude=32.760, longitude=35.080, risk_score=80.0),),
    )
    persist_spread_prediction(
        stack,
        golan_event_id,
        horizon_minutes=30,
        latitude=GOLAN_LATITUDE,
        longitude=GOLAN_LONGITUDE,
        cells=(make_cell(latitude=33.150, longitude=35.810, risk_score=82.0),),
    )

    carmel_result = stack["agent"].generate(fire_event_id=carmel_event_id, as_of=AS_OF)
    golan_result = stack["agent"].generate(fire_event_id=golan_event_id, as_of=AS_OF)

    assert {target.fire_event_id for target in carmel_result.targets} == {carmel_event_id}
    assert {target.fire_event_id for target in golan_result.targets} == {golan_event_id}
    assert {target.spread_prediction_id for target in carmel_result.targets if target.spread_prediction_id} != {
        target.spread_prediction_id for target in golan_result.targets if target.spread_prediction_id
    }
    assert all(target.latitude < 33.0 for target in carmel_result.targets)
    assert all(target.latitude > 33.0 for target in golan_result.targets)


@pytest.mark.parametrize("spread_state", ["none", "insufficient_data"])
def test_at8_active_fire_without_valid_spread_still_generates_active_fire_target(stack, spread_state):
    fire_event_id = persist_fire_event(stack)
    if spread_state == "insufficient_data":
        persist_spread_prediction(
            stack,
            fire_event_id,
            horizon_minutes=30,
            status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            cells=(),
            weather_observation_id=None,
        )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.status is ResponseTargetGenerationStatus.GENERATED
    assert result.target_count == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE


def test_at9_identical_effective_input_produces_same_logical_ordered_targets(stack):
    fire_event_id = persist_fire_event(stack)
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(
            make_cell(latitude=32.775, longitude=35.095, risk_score=70.0, reached_step=2),
            make_cell(latitude=32.760, longitude=35.080, risk_score=85.0, reached_step=1),
        ),
    )

    first = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)
    second = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    # Performance pass: identical effective input now REUSES the existing
    # target set (append-only history stays at 1 row) instead of minting a
    # duplicate - "same logical ordered targets" now means the same row,
    # not merely equal content in two different rows.
    assert first.target_set_id == second.target_set_id
    assert logical_targets(first.targets) == logical_targets(second.targets)
    assert len(stack["targets"].get_history_for_event(fire_event_id)) == 1


def test_active_fire_overlap_predicted_location_is_removed_but_active_fire_remains(stack):
    fire_event_id = persist_fire_event(stack)
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        cells=(make_cell(latitude=CARMEL_LATITUDE + 0.0001, longitude=CARMEL_LONGITUDE + 0.0001, risk_score=99.0),),
    )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.target_count == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert result.targets[0].latitude == CARMEL_LATITUDE
    assert result.targets[0].longitude == CARMEL_LONGITUDE


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_fire_events_do_not_create_response_target_snapshots(stack, status):
    fire_event_id = persist_fire_event(stack)
    stored_event = stack["fire_events"].get_by_id(fire_event_id)
    stack["fire_events"].update_event(
        fire_event_id,
        FireEvent(
            latitude=stored_event.event.latitude,
            longitude=stored_event.event.longitude,
            detected_at=stored_event.event.detected_at,
            updated_at=AS_OF,
            status=status,
            detection_confidence=stored_event.event.detection_confidence,
            methodology=stored_event.event.methodology,
            methodology_version=stored_event.event.methodology_version,
        ),
    )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.status is ResponseTargetGenerationStatus.INACTIVE_EVENT
    assert result.target_set_id is None
    assert result.targets == ()
    assert stack["targets"].get_history_for_event(fire_event_id) == ()


def test_latest_state_semantics_do_not_reuse_stale_valid_spread_or_severity(stack):
    fire_event_id = persist_fire_event(stack)
    persist_severity(stack, fire_event_id, score=90.0, assessed_at=AS_OF - timedelta(minutes=20))
    persist_severity(
        stack,
        fire_event_id,
        status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
        assessed_at=AS_OF - timedelta(minutes=10),
        weather_observation_ids=(),
        satellite_hotspot_ids=(),
        selected_frp_hotspot_id=None,
    )
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        predicted_at=AS_OF - timedelta(minutes=20),
        cells=(make_cell(latitude=32.760, longitude=35.080, risk_score=90.0),),
    )
    persist_spread_prediction(
        stack,
        fire_event_id,
        horizon_minutes=30,
        predicted_at=AS_OF - timedelta(minutes=10),
        status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
        cells=(),
        weather_observation_id=None,
    )

    result = stack["agent"].generate(fire_event_id=fire_event_id, as_of=AS_OF)

    assert result.target_count == 1
    assert result.targets[0].target_type is ResponseTargetType.ACTIVE_FIRE
    assert result.targets[0].priority_score == ACTIVE_FIRE_BASE_PRIORITY


def test_ac_static_architecture_guardrail_keeps_response_targets_independent_of_routing_and_simulation():
    forbidden_import_fragments = (
        "simulation",
        "scripts",
        "routing",
        "road_network",
        "RoadNetworkRepository",
        "GraphNode",
        "GraphEdge",
        "Dijkstra",
        "ResponsePlan",
    )
    production_files = [
        (Path(__file__).resolve().parents[3] / "backend/src/models/response_target.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/models/response_target_set.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/models/predicted_risk_target_candidate.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/models/response_target_input.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/calculators/response_target/response_target_calculator.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/services/response_target/response_target_input_service.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/repositories/response_target_repository.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/agents/analysis/response_target_generation_agent.py"),
        (Path(__file__).resolve().parents[3] / "backend/src/agents/analysis/response_target_generation_result.py"),
    ]

    violations = []
    for path in production_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = ",".join(alias.name for alias in node.names)
            if module and any(fragment in module for fragment in forbidden_import_fragments):
                violations.append((str(path), module))

    assert violations == []


def test_ac_methodology_constants_are_the_documented_v1_values():
    assert RESPONSE_TARGET_METHODOLOGY_NAME == "ECOGUARD_RESPONSE_TARGET_PRIORITY"
    assert RESPONSE_TARGET_METHODOLOGY_VERSION == "1.0"
    assert ACTIVE_FIRE_BASE_PRIORITY == 100.0
    assert MIN_PREDICTED_TARGET_RISK_SCORE == 60.0
    assert PREDICTION_HORIZON_FACTORS == {30: 1.0, 60: 0.85}
    assert TARGET_DEDUP_DISTANCE_METERS == 100.0


def persist_fire_event(
    stack,
    *,
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
) -> int:
    satellite_id = persist_hotspot(stack, latitude=latitude, longitude=longitude, detected_at=AS_OF - timedelta(minutes=40))
    return stack["fire_events"].create_event(
        FireEvent(
            latitude=latitude,
            longitude=longitude,
            detected_at=AS_OF - timedelta(minutes=40),
            updated_at=AS_OF - timedelta(minutes=5),
            status=FireEventStatus.CONFIRMED,
            detection_confidence=0.9,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, satellite_id),),
    ).id


def persist_hotspot(stack, *, latitude: float, longitude: float, detected_at: datetime) -> int:
    stack["satellite"].save_hotspot(
        SatelliteHotspot(
            latitude=latitude,
            longitude=longitude,
            detected_at=detected_at,
            confidence="h",
            frp=70.0,
            satellite="N20",
        )
    )
    return max(record.id for record in stack["satellite"].get_recent_hotspots(as_of=AS_OF, lookback_minutes=360))


def persist_weather(stack, *, station_offset: int, latitude: float, longitude: float) -> int:
    station = WeatherStation(
        external_station_id=970000 + station_offset,
        name=f"Response Target Acceptance Station {station_offset}",
        latitude=latitude,
        longitude=longitude,
    )
    stack["weather"].save_station(station)
    stack["weather"].save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=AS_OF - timedelta(minutes=5),
            temperature=28.0,
            relative_humidity=35.0,
            wind_speed=6.0,
            wind_direction=270.0,
        )
    )
    candidates = stack["weather"].get_recent_observations_for_area_candidates(
        latitude=latitude,
        longitude=longitude,
        radius_km=5.0,
        start_time=AS_OF - timedelta(minutes=30),
        end_time=AS_OF,
    )
    return next(
        record.observation_id
        for record in candidates
        if record.observation.station_external_id == station.external_station_id
    )


def persist_severity(
    stack,
    fire_event_id: int,
    *,
    score: float | None = 50.0,
    status: FireSeverityAssessmentStatus = FireSeverityAssessmentStatus.VALID,
    level: FireSeverityLevel | None = FireSeverityLevel.HIGH,
    assessed_at: datetime = AS_OF - timedelta(minutes=15),
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    weather_observation_ids: tuple[int, ...] | None = None,
    satellite_hotspot_ids: tuple[int, ...] | None = None,
    selected_frp_hotspot_id: int | None | str = "auto",
) -> int:
    if weather_observation_ids is None:
        weather_observation_ids = (persist_weather(stack, station_offset=fire_event_id, latitude=latitude, longitude=longitude),)
    if satellite_hotspot_ids is None:
        satellite_hotspot_ids = (
            persist_hotspot(stack, latitude=latitude, longitude=longitude, detected_at=assessed_at - timedelta(minutes=5)),
        )
    if selected_frp_hotspot_id == "auto":
        selected_frp_hotspot_id = satellite_hotspot_ids[0] if satellite_hotspot_ids else None
    return stack["severity"].save_assessment(
        FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at,
            status=status,
            score=score,
            level=level,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API" if status is FireSeverityAssessmentStatus.VALID else None,
            vegetation_dataset_year=2019 if status is FireSeverityAssessmentStatus.VALID else None,
            vegetation_radius_km=1.0 if status is FireSeverityAssessmentStatus.VALID else None,
            vegetation_dominant_land_cover="Shrub cover" if status is FireSeverityAssessmentStatus.VALID else None,
            vegetation_fuel_score=0.8 if status is FireSeverityAssessmentStatus.VALID else None,
        ),
        weather_observation_ids=weather_observation_ids,
        satellite_hotspot_ids=satellite_hotspot_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
    ).assessment_id


def persist_spread_prediction(
    stack,
    fire_event_id: int,
    *,
    horizon_minutes: int,
    cells: tuple[FireSpreadPredictionCell, ...],
    latitude: float = CARMEL_LATITUDE,
    longitude: float = CARMEL_LONGITUDE,
    predicted_at: datetime = AS_OF - timedelta(minutes=5),
    status: FireSpreadPredictionStatus = FireSpreadPredictionStatus.VALID,
    weather_observation_id: int | None | str = "auto",
):
    assessment_id = persist_severity(
        stack,
        fire_event_id,
        latitude=latitude,
        longitude=longitude,
        assessed_at=predicted_at - timedelta(minutes=5),
    )
    if weather_observation_id == "auto":
        weather_observation_id = persist_weather(
            stack,
            station_offset=1000 + fire_event_id + horizon_minutes,
            latitude=latitude,
            longitude=longitude,
        )
    return stack["spread"].save_prediction(
        FireSpreadPrediction(
            fire_event_id=fire_event_id,
            severity_assessment_id=assessment_id,
            predicted_at=predicted_at,
            horizon_minutes=horizon_minutes,
            status=status,
            methodology=METHODOLOGY_NAME,
            methodology_version=METHODOLOGY_VERSION,
            cells=cells,
        ),
        weather_observation_id=weather_observation_id,
    )


def make_cell(
    *,
    latitude: float,
    longitude: float,
    risk_score: float,
    reached_step: int = 1,
) -> FireSpreadPredictionCell:
    return FireSpreadPredictionCell(
        latitude=latitude,
        longitude=longitude,
        spread_probability=risk_score / 100.0,
        spread_risk_score=risk_score,
        reached_step=reached_step,
        reached_minutes=reached_step * 5,
    )


def logical_targets(targets):
    return tuple(
        (
            target.target_type,
            target.latitude,
            target.longitude,
            target.priority_score,
            target.prediction_horizon_minutes,
            target.spread_prediction_id,
            target.spread_prediction_cell_id,
        )
        for target in targets
    )
