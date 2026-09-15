"""Unit tests for ResponseTargetRepository using SQLite in-memory."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
from src.calculators.response_target.response_target_config import (
    RESPONSE_TARGET_METHODOLOGY_NAME,
    RESPONSE_TARGET_METHODOLOGY_VERSION,
)
from src.database.models.response_target_db import ResponseTargetDB
from src.database.models.response_target_set_db import ResponseTargetSetDB
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
    ResponseTarget,
    ResponseTargetSet,
    ResponseTargetType,
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import ResponseTargetRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.response_target_repository import (
    ResponseTargetRepository,
    StoredResponseTargetSet,
)
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository

GENERATED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_TIME = GENERATED_AT - timedelta(minutes=40)


@pytest.fixture
def repository(sqlite_session_factory) -> ResponseTargetRepository:
    return ResponseTargetRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def fire_event_repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def severity_repository(sqlite_session_factory) -> FireSeverityAssessmentRepository:
    return FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def spread_repository(sqlite_session_factory) -> FireSpreadPredictionRepository:
    return FireSpreadPredictionRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def weather_repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def event_id(fire_event_repository, satellite_repository) -> int:
    return persist_event(fire_event_repository, satellite_repository)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        updated_at=GENERATED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def persist_event(fire_event_repository, satellite_repository, **overrides) -> int:
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=32.731,
            longitude=35.046,
            detected_at=EVENT_TIME,
            confidence="h",
            frp=72.0,
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=GENERATED_AT, lookback_minutes=360)[0].id
    stored = fire_event_repository.create_event(
        make_event(**overrides),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return stored.id


def persist_weather(weather_repository: WeatherRepository, station_offset: int = 0) -> int:
    station = WeatherStation(
        external_station_id=930000 + station_offset,
        name=f"Station {station_offset}",
        latitude=32.731,
        longitude=35.046,
    )
    weather_repository.save_station(station)
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=station.external_station_id,
            timestamp=GENERATED_AT - timedelta(minutes=5),
            temperature=28.0,
            relative_humidity=35.0,
            wind_speed=6.0,
            wind_direction=270.0,
        )
    )
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.731,
        longitude=35.046,
        radius_km=5.0,
        start_time=GENERATED_AT - timedelta(minutes=30),
        end_time=GENERATED_AT,
    )
    return next(
        record.observation_id
        for record in candidates
        if record.observation.station_external_id == station.external_station_id
    )


def persist_assessment(severity_repository, weather_repository, satellite_repository, fire_event_id: int) -> int:
    weather_id = persist_weather(weather_repository, station_offset=fire_event_id)
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=32.731,
            longitude=35.046,
            detected_at=EVENT_TIME,
            confidence="h",
            frp=55.0,
            satellite="N21",
        )
    )
    hotspot_id = max(
        record.id for record in satellite_repository.get_recent_hotspots(as_of=GENERATED_AT, lookback_minutes=360)
    )
    stored = severity_repository.save_assessment(
        FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=GENERATED_AT - timedelta(minutes=10),
            status=FireSeverityAssessmentStatus.VALID,
            score=55.0,
            level=FireSeverityLevel.HIGH,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
            vegetation_dataset_year=2019,
            vegetation_radius_km=1.0,
            vegetation_dominant_land_cover="Shrub cover",
            vegetation_fuel_score=0.8,
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    return stored.assessment_id


def make_cell(**overrides) -> FireSpreadPredictionCell:
    defaults = dict(
        latitude=32.74,
        longitude=35.06,
        spread_probability=0.8,
        spread_risk_score=80.0,
        reached_step=1,
        reached_minutes=5,
    )
    defaults.update(overrides)
    return FireSpreadPredictionCell(**defaults)


def persist_prediction(
    spread_repository,
    severity_repository,
    weather_repository,
    satellite_repository,
    fire_event_id: int,
    **overrides,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, fire_event_id)
    observation_id = persist_weather(weather_repository, station_offset=100 + fire_event_id)
    cells = overrides.pop("cells", (make_cell(),))
    prediction = FireSpreadPrediction(
        fire_event_id=fire_event_id,
        severity_assessment_id=assessment_id,
        predicted_at=overrides.pop("predicted_at", GENERATED_AT - timedelta(minutes=5)),
        horizon_minutes=overrides.pop("horizon_minutes", 30),
        status=overrides.pop("status", FireSpreadPredictionStatus.VALID),
        methodology=METHODOLOGY_NAME,
        methodology_version=METHODOLOGY_VERSION,
        cells=cells,
    )
    return spread_repository.save_prediction(prediction, weather_observation_id=observation_id)


def make_active(fire_event_id: int, **overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.ACTIVE_FIRE,
        latitude=32.731,
        longitude=35.046,
        priority_score=150.0,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def make_predicted(fire_event_id: int, prediction_id: int, cell_id: int, **overrides) -> ResponseTarget:
    defaults = dict(
        fire_event_id=fire_event_id,
        target_type=ResponseTargetType.PREDICTED_RISK,
        latitude=32.74,
        longitude=35.06,
        priority_score=80.0,
        prediction_horizon_minutes=30,
        spread_prediction_id=prediction_id,
        spread_prediction_cell_id=cell_id,
    )
    defaults.update(overrides)
    return ResponseTarget(**defaults)


def make_target_set(fire_event_id: int, targets: tuple[ResponseTarget, ...], **overrides) -> ResponseTargetSet:
    defaults = dict(
        fire_event_id=fire_event_id,
        generated_at=GENERATED_AT,
        methodology=RESPONSE_TARGET_METHODOLOGY_NAME,
        methodology_version=RESPONSE_TARGET_METHODOLOGY_VERSION,
        targets=targets,
    )
    defaults.update(overrides)
    return ResponseTargetSet(**defaults)


def get_response_target_set_rows(sqlite_session_factory):
    session = sqlite_session_factory()
    rows = session.execute(select(ResponseTargetSetDB)).scalars().all()
    session.close()
    return rows


def get_response_target_rows(sqlite_session_factory):
    session = sqlite_session_factory()
    rows = session.execute(select(ResponseTargetDB).order_by(ResponseTargetDB.target_order.asc())).scalars().all()
    session.close()
    return rows


def test_save_set_creates_header_and_active_fire_target(repository, event_id, sqlite_session_factory):
    target_set = make_target_set(event_id, (make_active(event_id),))

    stored = repository.save_target_set(target_set)

    assert isinstance(stored, StoredResponseTargetSet)
    assert stored.id > 0
    assert len(get_response_target_set_rows(sqlite_session_factory)) == 1
    rows = get_response_target_rows(sqlite_session_factory)
    assert len(rows) == 1
    assert rows[0].target_type == "active_fire"


def test_save_predicted_target_with_exact_prediction_references(
    repository,
    event_id,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    prediction = persist_prediction(spread_repository, severity_repository, weather_repository, satellite_repository, event_id)
    cell_id = spread_repository.get_latest_for_event_and_horizon_as_of(event_id, 30, GENERATED_AT).cells[0].cell_id

    stored = repository.save_target_set(
        make_target_set(event_id, (make_active(event_id), make_predicted(event_id, prediction.id, cell_id)))
    )

    predicted = stored.targets[1].target
    assert predicted.spread_prediction_id == prediction.id
    assert predicted.spread_prediction_cell_id == cell_id
    assert predicted.prediction_horizon_minutes == 30


def test_multiple_targets_save_zero_based_order(
    repository,
    event_id,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    prediction = persist_prediction(
        spread_repository,
        severity_repository,
        weather_repository,
        satellite_repository,
        event_id,
        cells=(make_cell(), make_cell(latitude=32.75, longitude=35.07, reached_step=2, reached_minutes=10)),
    )
    stored_prediction = spread_repository.get_latest_for_event_and_horizon_as_of(event_id, 30, GENERATED_AT)
    targets = (
        make_active(event_id),
        make_predicted(event_id, prediction.id, stored_prediction.cells[0].cell_id),
        make_predicted(event_id, prediction.id, stored_prediction.cells[1].cell_id, latitude=32.75, longitude=35.07),
    )

    stored = repository.save_target_set(make_target_set(event_id, targets))

    assert [target.target_order for target in stored.targets] == [0, 1, 2]
    assert [row.target_order for row in get_response_target_rows(repository._session_factory)] == [0, 1, 2]


def test_round_trip_preserves_fields(
    repository,
    event_id,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    prediction = persist_prediction(spread_repository, severity_repository, weather_repository, satellite_repository, event_id)
    cell_id = spread_repository.get_latest_for_event_and_horizon_as_of(event_id, 30, GENERATED_AT).cells[0].cell_id
    target_set = make_target_set(
        event_id,
        (
            make_active(event_id, latitude=32.7311, longitude=35.0461, priority_score=177.7),
            make_predicted(event_id, prediction.id, cell_id, latitude=32.744, longitude=35.066, priority_score=68.5),
        ),
    )

    saved = repository.save_target_set(target_set)
    found = repository.get_by_id(saved.id)

    assert found.target_set == target_set
    assert found.target_set.methodology == RESPONSE_TARGET_METHODOLOGY_NAME
    assert found.target_set.methodology_version == RESPONSE_TARGET_METHODOLOGY_VERSION
    assert found.target_set.generated_at == GENERATED_AT
    assert found.target_set.fire_event_id == event_id
    assert found.targets[1].target.spread_prediction_id == prediction.id
    assert found.targets[1].target.spread_prediction_cell_id == cell_id


def test_get_by_id_returns_none_for_unknown_id(repository):
    assert repository.get_by_id(999999) is None


def test_missing_spread_prediction_reference_fails(repository, event_id):
    target_set = make_target_set(event_id, (make_active(event_id), make_predicted(event_id, 999999, 999999)))

    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set(target_set)


def test_missing_spread_prediction_cell_reference_fails(
    repository,
    event_id,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    prediction = persist_prediction(spread_repository, severity_repository, weather_repository, satellite_repository, event_id)

    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set(make_target_set(event_id, (make_active(event_id), make_predicted(event_id, prediction.id, 999999))))


def test_prediction_fire_event_mismatch_fails(
    repository,
    event_id,
    fire_event_repository,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    other_prediction = persist_prediction(
        spread_repository,
        severity_repository,
        weather_repository,
        satellite_repository,
        other_event_id,
    )
    other_cell_id = spread_repository.get_latest_for_event_and_horizon_as_of(other_event_id, 30, GENERATED_AT).cells[0].cell_id

    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set(
            make_target_set(event_id, (make_active(event_id), make_predicted(event_id, other_prediction.id, other_cell_id)))
        )


def test_prediction_cell_mismatch_fails(
    repository,
    event_id,
    severity_repository,
    spread_repository,
    weather_repository,
    satellite_repository,
):
    first_prediction = persist_prediction(spread_repository, severity_repository, weather_repository, satellite_repository, event_id)
    second_prediction = persist_prediction(
        spread_repository,
        severity_repository,
        weather_repository,
        satellite_repository,
        event_id,
        predicted_at=GENERATED_AT - timedelta(minutes=1),
        cells=(make_cell(latitude=32.75, longitude=35.07),),
    )
    second_cell_id = spread_repository.get_by_id(second_prediction.id).prediction.cells[0]
    second_cell_id = spread_repository.get_latest_for_event_and_horizon_as_of(event_id, 30, GENERATED_AT).cells[0].cell_id
    # The latest lookup returns second's cell because it is newer; pair it with first's prediction id.
    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set(
            make_target_set(event_id, (make_active(event_id), make_predicted(event_id, first_prediction.id, second_cell_id)))
        )


def test_malformed_active_or_predicted_targets_blocked_by_domain(event_id):
    with pytest.raises(ValueError):
        ResponseTarget(
            fire_event_id=event_id,
            target_type=ResponseTargetType.ACTIVE_FIRE,
            latitude=32.731,
            longitude=35.046,
            priority_score=100.0,
            spread_prediction_id=1,
        )
    with pytest.raises(ValueError):
        ResponseTarget(
            fire_event_id=event_id,
            target_type=ResponseTargetType.PREDICTED_RISK,
            latitude=32.74,
            longitude=35.06,
            priority_score=80.0,
        )


def test_failure_rolls_back_entire_set(repository, event_id, sqlite_session_factory):
    target_set = make_target_set(event_id, (make_active(event_id), make_predicted(event_id, 999999, 999999)))

    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set(target_set)

    assert get_response_target_set_rows(sqlite_session_factory) == []
    assert get_response_target_rows(sqlite_session_factory) == []


def test_append_only_saves_do_not_collapse(repository, event_id):
    first = repository.save_target_set(make_target_set(event_id, (make_active(event_id),)))
    second = repository.save_target_set(
        make_target_set(
            event_id,
            (make_active(event_id, priority_score=180.0),),
            generated_at=GENERATED_AT + timedelta(minutes=5),
        )
    )

    assert first.id != second.id
    assert repository.get_by_id(first.id) is not None
    assert repository.get_by_id(second.id) is not None


def test_latest_for_event_as_of_returns_latest_before_boundary(repository, event_id):
    older = repository.save_target_set(
        make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT - timedelta(minutes=10))
    )
    newer = repository.save_target_set(
        make_target_set(event_id, (make_active(event_id, priority_score=175.0),), generated_at=GENERATED_AT)
    )
    repository.save_target_set(
        make_target_set(event_id, (make_active(event_id, priority_score=200.0),), generated_at=GENERATED_AT + timedelta(minutes=10))
    )

    latest = repository.get_latest_for_event_as_of(event_id, GENERATED_AT)

    assert latest.id == newer.id
    assert latest.id != older.id


def test_latest_for_event_as_of_exact_boundary_included(repository, event_id):
    saved = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))

    latest = repository.get_latest_for_event_as_of(event_id, GENERATED_AT)

    assert latest.id == saved.id


def test_latest_for_event_as_of_excludes_other_fire_events(repository, event_id, fire_event_repository, satellite_repository):
    other_event_id = persist_event(fire_event_repository, satellite_repository)
    expected = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))
    repository.save_target_set(
        make_target_set(other_event_id, (make_active(other_event_id),), generated_at=GENERATED_AT + timedelta(minutes=1))
    )

    latest = repository.get_latest_for_event_as_of(event_id, GENERATED_AT + timedelta(minutes=10))

    assert latest.id == expected.id


def test_latest_for_event_as_of_tie_break_uses_newest_id(repository, event_id):
    first = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))
    second = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))

    latest = repository.get_latest_for_event_as_of(event_id, GENERATED_AT)

    assert latest.id == second.id
    assert second.id > first.id


def test_latest_for_event_as_of_returns_none_when_no_earlier_set(repository, event_id):
    repository.save_target_set(
        make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT + timedelta(minutes=1))
    )

    assert repository.get_latest_for_event_as_of(event_id, GENERATED_AT) is None


def test_history_order_is_deterministic(repository, event_id):
    first = repository.save_target_set(
        make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT - timedelta(minutes=10))
    )
    second = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))
    third = repository.save_target_set(make_target_set(event_id, (make_active(event_id),), generated_at=GENERATED_AT))

    history = repository.get_history_for_event(event_id)

    assert [item.id for item in history] == [third.id, second.id, first.id]


def test_invalid_repository_arguments_rejected(repository):
    with pytest.raises(ResponseTargetRepositoryError):
        repository.save_target_set("not-a-target-set")
    with pytest.raises(ResponseTargetRepositoryError):
        repository.get_by_id(0)
    with pytest.raises(ResponseTargetRepositoryError):
        repository.get_latest_for_event_as_of(0, GENERATED_AT)
    with pytest.raises(ResponseTargetRepositoryError):
        repository.get_latest_for_event_as_of(1, datetime(2026, 9, 14, 12, 0))
    with pytest.raises(ResponseTargetRepositoryError):
        repository.get_history_for_event(True)
