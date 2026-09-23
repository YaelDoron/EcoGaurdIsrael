"""Unit tests for FireSpreadPredictionRepository using SQLite in-memory.

Importing FireSpreadPredictionRepository (and its ORM modules) here is
sufficient to register FireSpreadPredictionDB/CellDB/WeatherInputDB on
Base.metadata before the shared `sqlite_engine` fixture runs
`Base.metadata.create_all()` -- conftest.py does not need to be touched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread.fire_spread_config import METHODOLOGY_NAME, METHODOLOGY_VERSION
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
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.exceptions import FireSpreadPredictionRepositoryError
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import (
    FireSpreadPredictionRepository,
    StoredFireSpreadPredictionWithCells,
    StoredFireSpreadPrediction,
)
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository

PREDICTED_AT = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_TIME = PREDICTED_AT - timedelta(minutes=40)


@pytest.fixture
def repository(sqlite_session_factory) -> FireSpreadPredictionRepository:
    return FireSpreadPredictionRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def fire_event_repository(sqlite_session_factory) -> FireEventRepository:
    return FireEventRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def severity_repository(sqlite_session_factory) -> FireSeverityAssessmentRepository:
    return FireSeverityAssessmentRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def weather_repository(sqlite_session_factory) -> WeatherRepository:
    return WeatherRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def satellite_repository(sqlite_session_factory) -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository(session_factory=sqlite_session_factory)


@pytest.fixture
def event_id(fire_event_repository, satellite_repository) -> int:
    """A persisted, valid FireEvent (with real supporting satellite evidence)."""
    return _persist_event(fire_event_repository, satellite_repository)


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=32.731,
        longitude=35.046,
        detected_at=EVENT_TIME,
        updated_at=PREDICTED_AT - timedelta(minutes=5),
        status=FireEventStatus.CONFIRMED,
        detection_confidence=0.85,
        methodology=FIRE_DETECTION_METHODOLOGY_NAME,
        methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
    )
    defaults.update(overrides)
    return FireEvent(**defaults)


def _persist_event(
    fire_event_repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
    **overrides,
) -> int:
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
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=PREDICTED_AT, lookback_minutes=360)[0].id

    stored = fire_event_repository.create_event(
        make_event(**overrides),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return stored.id


def persist_assessment(
    severity_repository: FireSeverityAssessmentRepository,
    weather_repository: WeatherRepository,
    satellite_repository: SatelliteHotspotRepository,
    fire_event_id: int,
    **overrides,
) -> int:
    """Persist a FireSeverityAssessment, with real weather/satellite traceability
    when it is VALID (the repository requires at least one of each for VALID)."""
    defaults = dict(
        fire_event_id=fire_event_id,
        assessed_at=PREDICTED_AT - timedelta(minutes=10),
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
    )
    defaults.update(overrides)
    assessment = FireSeverityAssessment(**defaults)

    weather_ids: tuple[int, ...] = ()
    satellite_ids: tuple[int, ...] = ()
    selected_frp_hotspot_id = None
    if assessment.status is FireSeverityAssessmentStatus.VALID:
        weather_ids = (persist_weather(weather_repository, station_offset=1),)
        satellite_repository.save_hotspot(
            SatelliteHotspot(
                latitude=32.731,
                longitude=35.046,
                detected_at=EVENT_TIME,
                confidence="h",
                frp=50.0,
                satellite="N21",
            )
        )
        hotspot_id = max(
            (record.id for record in satellite_repository.get_recent_hotspots(as_of=PREDICTED_AT, lookback_minutes=360)),
        )
        satellite_ids = (hotspot_id,)
        selected_frp_hotspot_id = hotspot_id

    stored = severity_repository.save_assessment(
        assessment=assessment,
        weather_observation_ids=weather_ids,
        satellite_hotspot_ids=satellite_ids,
        selected_frp_hotspot_id=selected_frp_hotspot_id,
    )
    return stored.assessment_id


def persist_weather(weather_repository: WeatherRepository, station_offset: int = 0) -> int:
    station = WeatherStation(
        external_station_id=920000 + station_offset,
        name=f"Station {station_offset}",
        latitude=32.731,
        longitude=35.046,
    )
    weather_repository.save_station(station)
    observation = WeatherObservation(
        station_external_id=station.external_station_id,
        timestamp=PREDICTED_AT - timedelta(minutes=5 + station_offset),
        temperature=28.0,
        relative_humidity=35.0,
        wind_speed=6.0,
        wind_direction=270.0,
    )
    weather_repository.save_observation(observation)
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=32.731,
        longitude=35.046,
        radius_km=5.0,
        start_time=PREDICTED_AT - timedelta(minutes=30),
        end_time=PREDICTED_AT,
    )
    return next(
        record.observation_id
        for record in candidates
        if record.observation.station_external_id == station.external_station_id
    )


def make_cell(**overrides) -> FireSpreadPredictionCell:
    defaults = dict(
        latitude=32.735,
        longitude=35.05,
        spread_probability=0.6,
        spread_risk_score=60.0,
        reached_step=1,
        reached_minutes=5,
    )
    defaults.update(overrides)
    return FireSpreadPredictionCell(**defaults)


def make_prediction(fire_event_id: int, severity_assessment_id, **overrides) -> FireSpreadPrediction:
    defaults = dict(
        fire_event_id=fire_event_id,
        severity_assessment_id=severity_assessment_id,
        predicted_at=PREDICTED_AT,
        horizon_minutes=30,
        status=FireSpreadPredictionStatus.VALID,
        methodology=METHODOLOGY_NAME,
        methodology_version=METHODOLOGY_VERSION,
        cells=(make_cell(),),
    )
    defaults.update(overrides)
    return FireSpreadPrediction(**defaults)


# ---------------------------------------------------------------------------
# Save VALID
# ---------------------------------------------------------------------------


def test_save_valid_prediction_with_cells(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id),
        weather_observation_id=observation_id,
    )

    assert isinstance(stored, StoredFireSpreadPrediction)
    assert stored.id > 0
    assert stored.prediction.status is FireSpreadPredictionStatus.VALID
    assert len(stored.prediction.cells) == 1
    assert stored.weather_observation_id == observation_id
    assert stored.prediction.effective_state_fingerprint is None


def test_save_valid_prediction_round_trips_effective_state_fingerprint(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    fingerprint = "a" * 64

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, effective_state_fingerprint=fingerprint),
        weather_observation_id=observation_id,
    )
    found = repository.get_by_id(stored.id)
    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert stored.prediction.effective_state_fingerprint == fingerprint
    assert found.prediction.effective_state_fingerprint == fingerprint
    assert latest.prediction.effective_state_fingerprint == fingerprint


@pytest.mark.parametrize("fingerprint", ["A" * 64, "g" * 64, "a" * 63, "", 123])
def test_invalid_effective_state_fingerprint_rejected(fingerprint, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)

    with pytest.raises(ValueError):
        make_prediction(event_id, assessment_id, effective_state_fingerprint=fingerprint)


def test_save_valid_no_spread_prediction_with_zero_cells(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, cells=()),
        weather_observation_id=observation_id,
    )

    assert stored.prediction.status is FireSpreadPredictionStatus.VALID
    assert stored.prediction.cells == ()


def test_save_valid_persists_all_cell_fields_correctly(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    cells = (
        make_cell(latitude=32.73, longitude=35.04, spread_probability=0.55, spread_risk_score=55.0, reached_step=1, reached_minutes=5),
        make_cell(latitude=32.74, longitude=35.05, spread_probability=0.7, spread_risk_score=70.0, reached_step=2, reached_minutes=10),
    )

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, cells=cells),
        weather_observation_id=observation_id,
    )

    reconstructed = stored.prediction.cells
    assert len(reconstructed) == 2
    assert reconstructed[0].reached_step == 1
    assert reconstructed[1].reached_step == 2
    assert reconstructed[0].spread_probability == 0.55
    assert reconstructed[1].spread_probability == 0.7


def test_save_valid_preserves_fire_event_and_severity_references(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id),
        weather_observation_id=observation_id,
    )

    assert stored.prediction.fire_event_id == event_id
    assert stored.prediction.severity_assessment_id == assessment_id


def test_save_valid_without_weather_observation_id_rejected(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)

    with pytest.raises(FireSpreadPredictionRepositoryError):
        repository.save_prediction(
            prediction=make_prediction(event_id, assessment_id),
            weather_observation_id=None,
        )


# ---------------------------------------------------------------------------
# Save non-VALID
# ---------------------------------------------------------------------------


def test_save_insufficient_data_with_zero_cells(repository, event_id):
    stored = repository.save_prediction(
        prediction=make_prediction(
            event_id,
            None,
            status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            cells=(),
        ),
        weather_observation_id=None,
    )

    assert stored.prediction.status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
    assert stored.prediction.cells == ()
    assert stored.prediction.severity_assessment_id is None
    assert stored.weather_observation_id is None


def test_save_inactive_event_with_zero_cells(repository, fire_event_repository, satellite_repository):
    # FireEventRepository.create_event only accepts SUSPECTED/CONFIRMED; an
    # event transitions to RESOLVED via update_event, matching its real lifecycle.
    inactive_event_id = _persist_event(fire_event_repository, satellite_repository)
    resolved_event = fire_event_repository.get_by_id(inactive_event_id).event
    fire_event_repository.update_event(
        inactive_event_id,
        FireEvent(
            latitude=resolved_event.latitude,
            longitude=resolved_event.longitude,
            detected_at=resolved_event.detected_at,
            updated_at=PREDICTED_AT,
            status=FireEventStatus.RESOLVED,
            detection_confidence=resolved_event.detection_confidence,
            methodology=resolved_event.methodology,
            methodology_version=resolved_event.methodology_version,
        ),
    )

    stored = repository.save_prediction(
        prediction=make_prediction(
            inactive_event_id,
            None,
            status=FireSpreadPredictionStatus.INACTIVE_EVENT,
            cells=(),
        ),
        weather_observation_id=None,
    )

    assert stored.prediction.status is FireSpreadPredictionStatus.INACTIVE_EVENT
    assert stored.prediction.cells == ()


def test_save_insufficient_data_with_known_severity_id_preserved(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(
        severity_repository,
        weather_repository,
        satellite_repository,
        event_id,
        status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA,
        score=None,
        level=None,
    )

    stored = repository.save_prediction(
        prediction=make_prediction(
            event_id,
            assessment_id,
            status=FireSpreadPredictionStatus.INSUFFICIENT_DATA,
            cells=(),
        ),
        weather_observation_id=None,
    )

    assert stored.prediction.severity_assessment_id == assessment_id


# ---------------------------------------------------------------------------
# History (append-only)
# ---------------------------------------------------------------------------


def test_two_predictions_for_same_event_and_horizon_produce_two_rows(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    first = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT - timedelta(minutes=30)),
        weather_observation_id=observation_id,
    )
    second = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )

    assert first.id != second.id
    assert repository.get_by_id(first.id) is not None
    assert repository.get_by_id(second.id) is not None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_get_by_id_returns_stored_prediction(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id),
        weather_observation_id=observation_id,
    )

    found = repository.get_by_id(stored.id)

    assert found is not None
    assert found.id == stored.id
    assert found.prediction.fire_event_id == event_id


def test_get_by_id_returns_none_for_missing_id(repository):
    assert repository.get_by_id(999999) is None


def test_get_latest_for_event_and_horizon_returns_most_recent(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT - timedelta(minutes=30)),
        weather_observation_id=observation_id,
    )
    newest = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon(event_id, 30)

    assert latest is not None
    assert latest.id == newest.id
    assert latest.prediction.predicted_at == PREDICTED_AT


def test_get_latest_separates_30_and_60_minute_horizons(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)

    thirty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=30),
        weather_observation_id=observation_id,
    )
    sixty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=60),
        weather_observation_id=observation_id,
    )

    latest_30 = repository.get_latest_for_event_and_horizon(event_id, 30)
    latest_60 = repository.get_latest_for_event_and_horizon(event_id, 60)

    assert latest_30.id == thirty.id
    assert latest_60.id == sixty.id
    assert latest_30.id != latest_60.id


def test_get_latest_returns_none_when_no_prediction_exists(repository, event_id):
    assert repository.get_latest_for_event_and_horizon(event_id, 30) is None


def test_get_latest_for_event_and_horizon_as_of_filters_future_prediction(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    current = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT - timedelta(minutes=5)),
        weather_observation_id=observation_id,
    )
    repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT + timedelta(minutes=5)),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert latest.id == current.id


def test_get_latest_for_event_and_horizon_as_of_exact_timestamp_boundary_included(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    expected = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert latest.id == expected.id


def test_get_latest_for_event_and_horizon_as_of_tie_break_uses_newest_id(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    first = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )
    second = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert latest.id == second.id
    assert second.id > first.id


def test_get_latest_for_event_and_horizon_as_of_filters_horizon(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    thirty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=30),
        weather_observation_id=observation_id,
    )
    sixty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=60),
        weather_observation_id=observation_id,
    )

    latest_30 = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)
    latest_60 = repository.get_latest_for_event_and_horizon_as_of(event_id, 60, PREDICTED_AT)

    assert latest_30.id == thirty.id
    assert latest_60.id == sixty.id


def test_get_latest_for_event_and_horizon_as_of_filters_fire_event(
    repository,
    event_id,
    fire_event_repository,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    other_event_id = _persist_event(fire_event_repository, satellite_repository)
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    other_assessment_id = persist_assessment(
        severity_repository,
        weather_repository,
        satellite_repository,
        other_event_id,
    )
    observation_id = persist_weather(weather_repository)
    expected = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, predicted_at=PREDICTED_AT),
        weather_observation_id=observation_id,
    )
    repository.save_prediction(
        prediction=make_prediction(other_event_id, other_assessment_id, predicted_at=PREDICTED_AT + timedelta(minutes=1)),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT + timedelta(minutes=10))

    assert latest.id == expected.id


def test_get_latest_for_event_and_horizon_as_of_returns_cell_ids(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, cells=(make_cell(), make_cell(latitude=32.74, longitude=35.055))),
        weather_observation_id=observation_id,
    )

    latest = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert isinstance(latest, StoredFireSpreadPredictionWithCells)
    assert latest.id == stored.id
    assert len(latest.cells) == 2
    assert all(cell.cell_id > 0 for cell in latest.cells)
    assert [cell.cell.spread_risk_score for cell in latest.cells] == [60.0, 60.0]


def test_get_latest_for_event_and_horizons_as_of_batches_both_horizons_in_one_call(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    thirty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=30, cells=(make_cell(),)),
        weather_observation_id=observation_id,
    )
    sixty = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=60, cells=(make_cell(),)),
        weather_observation_id=observation_id,
    )

    by_horizon = repository.get_latest_for_event_and_horizons_as_of(event_id, (30, 60), PREDICTED_AT)

    assert set(by_horizon) == {30, 60}
    assert by_horizon[30].id == thirty.id
    assert by_horizon[60].id == sixty.id
    assert len(by_horizon[30].cells) == 1
    assert len(by_horizon[60].cells) == 1


def test_get_latest_for_event_and_horizons_as_of_matches_per_horizon_method(
    repository,
    event_id,
    severity_repository,
    weather_repository,
    satellite_repository,
):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, horizon_minutes=30),
        weather_observation_id=observation_id,
    )
    # No 60m prediction persisted - horizon should be absent from the result.

    by_horizon = repository.get_latest_for_event_and_horizons_as_of(event_id, (30, 60), PREDICTED_AT)
    individual_30 = repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT)

    assert set(by_horizon) == {30}
    assert by_horizon[30].id == individual_30.id


def test_get_latest_for_event_and_horizons_as_of_empty_horizons_returns_empty_dict(repository, event_id):
    assert repository.get_latest_for_event_and_horizons_as_of(event_id, (), PREDICTED_AT) == {}


def test_get_latest_for_event_and_horizon_as_of_returns_none_when_no_past_prediction(repository, event_id):
    assert repository.get_latest_for_event_and_horizon_as_of(event_id, 30, PREDICTED_AT) is None


def test_deterministic_cell_reconstruction_order(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    observation_id = persist_weather(weather_repository)
    cells = (
        make_cell(latitude=32.74, longitude=35.05, reached_step=2, reached_minutes=10),
        make_cell(latitude=32.73, longitude=35.04, reached_step=1, reached_minutes=5),
        make_cell(latitude=32.735, longitude=35.045, reached_step=1, reached_minutes=5),
    )

    stored = repository.save_prediction(
        prediction=make_prediction(event_id, assessment_id, cells=cells),
        weather_observation_id=observation_id,
    )
    found = repository.get_by_id(stored.id)

    ordering = [(c.reached_step, c.latitude, c.longitude) for c in found.prediction.cells]
    assert ordering == sorted(ordering)


# ---------------------------------------------------------------------------
# Transaction / validation
# ---------------------------------------------------------------------------


def test_invalid_fire_event_fk_rolls_back(repository, event_id, severity_repository, weather_repository, satellite_repository):
    assessment_id = persist_assessment(severity_repository, weather_repository, satellite_repository, event_id)
    bad_prediction = make_prediction(999999, assessment_id)

    with pytest.raises(FireSpreadPredictionRepositoryError):
        repository.save_prediction(prediction=bad_prediction, weather_observation_id=None)

    # No row should remain from the failed attempt.
    assert repository.get_latest_for_event_and_horizon(999999, 30) is None


def test_save_prediction_requires_fire_spread_prediction_instance(repository):
    with pytest.raises(FireSpreadPredictionRepositoryError):
        repository.save_prediction(prediction="not-a-prediction", weather_observation_id=None)


@pytest.mark.parametrize("invalid_id", [0, -1, True])
def test_get_by_id_rejects_invalid_id(repository, invalid_id):
    with pytest.raises(FireSpreadPredictionRepositoryError):
        repository.get_by_id(invalid_id)
