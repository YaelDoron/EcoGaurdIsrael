"""Unit tests for FireSeverityAssessmentRepository using SQLite in-memory."""
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
from src.database.models.fire_severity_assessment_db import FireSeverityAssessmentDB
from src.database.models.fire_severity_assessment_satellite_input_db import (
    FireSeverityAssessmentSatelliteInputDB,
)
from src.database.models.fire_severity_assessment_weather_input_db import (
    FireSeverityAssessmentWeatherInputDB,
)
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import FireSeverityAssessmentRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import (
    FireSeverityAssessmentRepository,
    StoredFireSeverityAssessment,
)
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository

ASSESSED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_TIME = ASSESSED_AT - timedelta(minutes=20)


@pytest.fixture
def repository(sqlite_session_factory) -> FireSeverityAssessmentRepository:
    return FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def fire_event_repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def weather_repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        updated_at=ASSESSED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def make_assessment(fire_event_id: int, **overrides) -> FireSeverityAssessment:
    defaults = dict(
        fire_event_id=fire_event_id,
        assessed_at=ASSESSED_AT,
        status=FireSeverityAssessmentStatus.VALID,
        score=82.5,
        level=FireSeverityLevel.CRITICAL,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        vegetation_dataset_year=2019,
        vegetation_radius_km=1.0,
        vegetation_dominant_land_cover="Tree cover",
        vegetation_fuel_score=0.9,
    )
    defaults.update(overrides)
    return FireSeverityAssessment(**defaults)


def make_hotspot(**overrides) -> SatelliteHotspot:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        confidence="h",
        frp=72.0,
        satellite="N20",
    )
    defaults.update(overrides)
    return SatelliteHotspot(**defaults)


def persist_weather(weather_repository: WeatherRepository, station_offset: int = 0, **overrides) -> int:
    station = WeatherStation(
        external_station_id=910000 + station_offset,
        name=f"Station {station_offset}",
        latitude=32.731,
        longitude=35.046,
    )
    weather_repository.save_station(station)
    observation = WeatherObservation(
        station_external_id=station.external_station_id,
        timestamp=ASSESSED_AT - timedelta(minutes=5 + station_offset),
        temperature=30.0,
        relative_humidity=25.0,
        wind_speed=20.0,
        **overrides,
    )
    weather_repository.save_observation(observation)
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.731,
        longitude=35.046,
        radius_km=5.0,
        start_time=ASSESSED_AT - timedelta(minutes=30),
        end_time=ASSESSED_AT,
    )
    return next(
        record.observation_id
        for record in candidates
        if record.observation.station_external_id == station.external_station_id
    )


def persist_hotspot(satellite_repository: SatelliteHotspotRepository, offset_minutes: int = 0, **overrides) -> int:
    hotspot = make_hotspot(detected_at=EVENT_TIME + timedelta(minutes=offset_minutes), **overrides)
    satellite_repository.save_hotspot(hotspot)
    return satellite_repository.get_recent_hotspots(
        as_of=ASSESSED_AT,
        lookback_minutes=360,
    )[0].id


def persist_fire_event(
    fire_event_repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
) -> tuple[int, int]:
    hotspot_id = persist_hotspot(satellite_repository)
    saved = fire_event_repository.create_event(
        make_event(),
        (FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return saved.id, hotspot_id


def test_save_valid_assessment_persists_fields_and_traceability(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, first_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    second_hotspot_id = persist_hotspot(satellite_repository, offset_minutes=1, frp=55)
    first_weather_id = persist_weather(weather_repository, station_offset=1)
    second_weather_id = persist_weather(weather_repository, station_offset=2)

    saved = repository.save_assessment(
        make_assessment(fire_event_id),
        weather_observation_ids=(second_weather_id, first_weather_id),
        satellite_hotspot_ids=(second_hotspot_id, first_hotspot_id),
        selected_frp_hotspot_id=first_hotspot_id,
    )

    assert isinstance(saved, StoredFireSeverityAssessment)
    assert saved.assessment_id > 0
    assert saved.assessment.fire_event_id == fire_event_id
    assert saved.assessment.score == pytest.approx(82.5)
    assert saved.assessment.level is FireSeverityLevel.CRITICAL
    assert saved.assessment.methodology == FIRE_SEVERITY_METHODOLOGY_NAME
    assert saved.assessment.methodology_version == FIRE_SEVERITY_METHODOLOGY_VERSION
    assert saved.assessment.vegetation_fuel_score == pytest.approx(0.9)
    assert saved.weather_observation_ids == (first_weather_id, second_weather_id)
    assert saved.satellite_hotspot_ids == (first_hotspot_id, second_hotspot_id)
    assert saved.selected_frp_hotspot_id == first_hotspot_id
    assert repository.get_selected_frp_hotspot_id(saved.assessment_id) == first_hotspot_id


def test_valid_assessment_without_vegetation_persists(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)

    saved = repository.save_assessment(
        make_assessment(
            fire_event_id,
            vegetation_source=None,
            vegetation_dataset_year=None,
            vegetation_radius_km=None,
            vegetation_dominant_land_cover=None,
            vegetation_fuel_score=None,
        ),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    assert saved.assessment.vegetation_source is None


def test_selected_frp_hotspot_marked_exactly_once(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
    sqlite_session_factory,
):
    fire_event_id, first_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    second_hotspot_id = persist_hotspot(satellite_repository, offset_minutes=1, frp=40)
    weather_id = persist_weather(weather_repository)

    saved = repository.save_assessment(
        make_assessment(fire_event_id),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(first_hotspot_id, second_hotspot_id),
        selected_frp_hotspot_id=second_hotspot_id,
    )

    session = sqlite_session_factory()
    rows = session.execute(
        select(FireSeverityAssessmentSatelliteInputDB).where(
            FireSeverityAssessmentSatelliteInputDB.assessment_id == saved.assessment_id
        )
    ).scalars().all()
    session.close()
    assert sum(1 for row in rows if row.selected_for_frp) == 1
    assert repository.get_selected_frp_hotspot_id(saved.assessment_id) == second_hotspot_id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weather_observation_ids": (), "satellite_hotspot_ids": (1,), "selected_frp_hotspot_id": 1},
        {"weather_observation_ids": (1,), "satellite_hotspot_ids": (), "selected_frp_hotspot_id": None},
        {"weather_observation_ids": (1,), "satellite_hotspot_ids": (1,), "selected_frp_hotspot_id": None},
        {"weather_observation_ids": (1,), "satellite_hotspot_ids": (1,), "selected_frp_hotspot_id": 2},
    ],
)
def test_valid_assessment_trace_requirements_rejected(repository, kwargs):
    with pytest.raises(FireSeverityAssessmentRepositoryError):
        repository.save_assessment(make_assessment(1), **kwargs)


def test_invalid_fire_event_fk_rolls_back(repository, weather_repository, satellite_repository, sqlite_session_factory):
    weather_id = persist_weather(weather_repository)
    hotspot_id = persist_hotspot(satellite_repository)

    with pytest.raises(FireSeverityAssessmentRepositoryError):
        repository.save_assessment(
            make_assessment(999),
            weather_observation_ids=(weather_id,),
            satellite_hotspot_ids=(hotspot_id,),
            selected_frp_hotspot_id=hotspot_id,
        )

    session = sqlite_session_factory()
    count = len(session.execute(select(FireSeverityAssessmentDB)).scalars().all())
    session.close()
    assert count == 0


def test_invalid_weather_fk_rolls_back(repository, fire_event_repository, satellite_repository, sqlite_session_factory):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)

    with pytest.raises(FireSeverityAssessmentRepositoryError):
        repository.save_assessment(
            make_assessment(fire_event_id),
            weather_observation_ids=(999,),
            satellite_hotspot_ids=(hotspot_id,),
            selected_frp_hotspot_id=hotspot_id,
        )

    session = sqlite_session_factory()
    count = len(session.execute(select(FireSeverityAssessmentDB)).scalars().all())
    session.close()
    assert count == 0


def test_invalid_satellite_fk_rolls_back(repository, fire_event_repository, weather_repository, sqlite_session_factory):
    fire_event_id, _ = persist_fire_event(fire_event_repository, SatelliteHotspotRepository(sqlite_session_factory))
    weather_id = persist_weather(weather_repository)

    with pytest.raises(FireSeverityAssessmentRepositoryError):
        repository.save_assessment(
            make_assessment(fire_event_id),
            weather_observation_ids=(weather_id,),
            satellite_hotspot_ids=(999,),
            selected_frp_hotspot_id=999,
        )

    session = sqlite_session_factory()
    count = len(session.execute(select(FireSeverityAssessmentDB)).scalars().all())
    session.close()
    assert count == 0


@pytest.mark.parametrize(
    ("status", "weather_ids", "satellite_ids", "selected_id"),
    [
        (FireSeverityAssessmentStatus.INSUFFICIENT_DATA, (), (), None),
        (FireSeverityAssessmentStatus.INSUFFICIENT_DATA, (1,), (), None),
        (FireSeverityAssessmentStatus.INSUFFICIENT_DATA, (), (1,), None),
        (FireSeverityAssessmentStatus.INACTIVE_EVENT, (), (), None),
    ],
)
def test_partial_statuses_persist(repository, fire_event_repository, weather_repository, satellite_repository, status, weather_ids, satellite_ids, selected_id):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    resolved_weather_ids = tuple(weather_id if value == 1 else value for value in weather_ids)
    resolved_satellite_ids = tuple(hotspot_id if value == 1 else value for value in satellite_ids)

    saved = repository.save_assessment(
        make_assessment(fire_event_id, status=status, score=None, level=None),
        weather_observation_ids=resolved_weather_ids,
        satellite_hotspot_ids=resolved_satellite_ids,
        selected_frp_hotspot_id=selected_id,
    )

    assert saved.assessment.status is status
    assert saved.assessment.score is None
    assert saved.assessment.level is None


def test_duplicate_trace_ids_are_deduplicated_deterministically(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)

    saved = repository.save_assessment(
        make_assessment(fire_event_id),
        weather_observation_ids=(weather_id, weather_id),
        satellite_hotspot_ids=(hotspot_id, hotspot_id),
        selected_frp_hotspot_id=hotspot_id,
    )

    assert saved.weather_observation_ids == (weather_id,)
    assert saved.satellite_hotspot_ids == (hotspot_id,)
    assert repository.get_weather_input_ids(saved.assessment_id) == (weather_id,)
    assert repository.get_satellite_input_ids(saved.assessment_id) == (hotspot_id,)


def test_multiple_assessments_for_same_fire_event_preserve_history_and_latest(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    older = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT - timedelta(hours=1), score=50, level=FireSeverityLevel.HIGH),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    newer = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT, score=80, level=FireSeverityLevel.CRITICAL),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    latest = repository.get_latest_for_event(fire_event_id)

    assert repository.get_by_id(older.assessment_id).assessment.score == pytest.approx(50)
    assert repository.get_by_id(newer.assessment_id).assessment.score == pytest.approx(80)
    assert latest.assessment_id == newer.assessment_id


def test_latest_tie_break_uses_newest_id(repository, fire_event_repository, weather_repository, satellite_repository):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    first = repository.save_assessment(
        make_assessment(fire_event_id),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    second = repository.save_assessment(
        make_assessment(fire_event_id),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    assert repository.get_latest_for_event(fire_event_id).assessment_id == second.assessment_id
    assert second.assessment_id > first.assessment_id


def test_get_latest_for_unknown_event_returns_none(repository):
    assert repository.get_latest_for_event(999) is None


def test_get_latest_for_event_as_of_ignores_future_assessment(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    current = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT - timedelta(minutes=5), score=70, level=FireSeverityLevel.HIGH),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT + timedelta(minutes=5), score=90, level=FireSeverityLevel.CRITICAL),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    latest = repository.get_latest_for_event_as_of(fire_event_id, ASSESSED_AT)

    assert latest.assessment_id == current.assessment_id
    assert latest.assessment.score == pytest.approx(70)


def test_get_latest_for_event_as_of_exact_timestamp_boundary_included(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    expected = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    latest = repository.get_latest_for_event_as_of(fire_event_id, ASSESSED_AT)

    assert latest.assessment_id == expected.assessment_id


def test_get_latest_for_event_as_of_tie_break_uses_newest_id(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    fire_event_id, hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    first = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    second = repository.save_assessment(
        make_assessment(fire_event_id, assessed_at=ASSESSED_AT),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )

    latest = repository.get_latest_for_event_as_of(fire_event_id, ASSESSED_AT)

    assert latest.assessment_id == second.assessment_id
    assert second.assessment_id > first.assessment_id


def test_get_latest_for_event_as_of_filters_by_fire_event(
    repository,
    fire_event_repository,
    weather_repository,
    satellite_repository,
):
    first_event_id, first_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    second_event_id, second_hotspot_id = persist_fire_event(fire_event_repository, satellite_repository)
    weather_id = persist_weather(weather_repository)
    expected = repository.save_assessment(
        make_assessment(first_event_id, assessed_at=ASSESSED_AT, score=65, level=FireSeverityLevel.HIGH),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(first_hotspot_id,),
        selected_frp_hotspot_id=first_hotspot_id,
    )
    repository.save_assessment(
        make_assessment(second_event_id, assessed_at=ASSESSED_AT + timedelta(minutes=1), score=90, level=FireSeverityLevel.CRITICAL),
        weather_observation_ids=(weather_id,),
        satellite_hotspot_ids=(second_hotspot_id,),
        selected_frp_hotspot_id=second_hotspot_id,
    )

    latest = repository.get_latest_for_event_as_of(first_event_id, ASSESSED_AT + timedelta(minutes=10))

    assert latest.assessment_id == expected.assessment_id


def test_get_latest_for_event_as_of_returns_none_when_no_past_assessment(repository, fire_event_repository, satellite_repository):
    fire_event_id, _ = persist_fire_event(fire_event_repository, satellite_repository)

    assert repository.get_latest_for_event_as_of(fire_event_id, ASSESSED_AT) is None
