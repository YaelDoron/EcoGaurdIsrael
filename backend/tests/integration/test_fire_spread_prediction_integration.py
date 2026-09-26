"""Live end-to-end acceptance test for User Story 4.2 (wildfire-spread prediction)
against the real Neon/PostgreSQL database.

Exercises the full real production chain with no mocks below the agent
boundary: FireEventRepository -> FireSeverityAssessmentRepository ->
WeatherRepository -> FireSpreadInputService -> FireSpreadCalculator ->
FireSpreadPredictionAgent -> FireSpreadPredictionRepository. Never calls live
IMS, Copernicus, or NASA FIRMS -- all upstream data is persisted directly via
the same repositories production code uses.

Complements, and deliberately does not duplicate, the already-thorough unit
coverage in:
- tests/calculators/fire_spread/test_fire_spread_calculator.py (scientific
  properties: downwind/upwind/crosswind, wind strength, moisture monotonicity,
  fuel-class differences, threshold boundary, wind wrap-around, determinism)
- tests/services/fire_spread/test_fire_spread_input_service.py (every
  vegetation label, every INSUFFICIENT_DATA branch, weather selection policy)
- tests/repositories/test_fire_spread_prediction_repository.py and
  tests/agents/analysis/test_fire_spread_prediction_agent.py (persistence and
  orchestration behavior, exercised there via SQLite/fakes)
This file's job is to prove those already-verified units are wired together
correctly against the real database, which none of the above exercise.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from src.agents.analysis import FireSpreadPredictionAgent
from src.calculators.fire_detection.fire_detection_config import (
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.calculators.fire_spread import FireSpreadCalculator
from src.calculators.fire_spread.fire_spread_config import (
    METHODOLOGY_NAME,
    METHODOLOGY_VERSION,
    PROPAGATION_THRESHOLD,
)
from src.config.settings import settings
from src.database.connection import get_engine, init_db
from src.models import (
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadPredictionStatus,
    SatelliteHotspot,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import FireEventRepository
from src.repositories.fire_severity_assessment_repository import FireSeverityAssessmentRepository
from src.repositories.fire_spread_prediction_repository import FireSpreadPredictionRepository
from src.repositories.satellite_hotspot_repository import SatelliteHotspotRepository
from src.repositories.weather_repository import WeatherRepository
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 5, 17, 9, 30, 0, tzinfo=timezone.utc)
DETECTED_AT = AS_OF - timedelta(minutes=40)
ASSESSED_AT = AS_OF - timedelta(minutes=10)
LATITUDE = 32.91234
LONGITUDE = 35.31234
STATION_EXTERNAL_ID = 924924

# Deliberately favorable conditions (strong wind, very dry fuel, grassland's
# high self-transition p_n) so the real chain actually propagates cells --
# see the module-level derivation notes in the completion report for why
# these particular values were chosen (verified by hand against the exact
# verified formulas, not guessed).
FAVORABLE_WIND_SPEED_KMH = 100.0  # WeatherObservation.wind_speed is canonical km/h
FAVORABLE_WIND_DIRECTION_DEG = 0.0  # wind FROM north -> blows toward south
FAVORABLE_TEMPERATURE_C = 42.0
FAVORABLE_HUMIDITY_PCT = 5.0
FAVORABLE_LAND_COVER = "Grass cover"

# Deliberately unfavorable, but entirely valid, conditions expected to
# produce zero cells above the deterministic threshold (a valid "no
# predicted spread" result).
NO_SPREAD_WIND_SPEED_KMH = 15.0
NO_SPREAD_WIND_DIRECTION_DEG = 0.0
NO_SPREAD_TEMPERATURE_C = 20.0
NO_SPREAD_HUMIDITY_PCT = 45.0
NO_SPREAD_LAND_COVER = "Shrub cover"


@pytest.fixture(autouse=True)
def _require_database_url() -> None:
    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL is not configured; skipping live Neon integration test.")


@pytest.fixture(autouse=True)
def _clean_test_rows(_require_database_url) -> None:
    init_db()
    _delete_test_rows()
    yield
    _delete_test_rows()


def _delete_test_rows() -> None:
    """Delete only this test file's own rows, identified by DETECTED_AT/
    LATITUDE/LONGITUDE/STATION_EXTERNAL_ID, in FK-safe (child-first) order."""
    engine = get_engine()
    with engine.begin() as connection:
        event_subquery = (
            "SELECT id FROM fire_events WHERE methodology = :det_methodology AND detected_at = :detected_at"
        )
        params = {"det_methodology": FIRE_DETECTION_METHODOLOGY_NAME, "detected_at": DETECTED_AT}

        connection.execute(
            text(
                "DELETE FROM fire_spread_prediction_cells WHERE prediction_id IN ("
                f"SELECT id FROM fire_spread_predictions WHERE fire_event_id IN ({event_subquery}))"
            ),
            params,
        )
        connection.execute(
            text(
                "DELETE FROM fire_spread_prediction_weather_inputs WHERE prediction_id IN ("
                f"SELECT id FROM fire_spread_predictions WHERE fire_event_id IN ({event_subquery}))"
            ),
            params,
        )
        connection.execute(
            text(f"DELETE FROM fire_spread_predictions WHERE fire_event_id IN ({event_subquery})"),
            params,
        )
        connection.execute(
            text(
                "DELETE FROM fire_severity_assessment_weather_inputs WHERE assessment_id IN ("
                f"SELECT id FROM fire_severity_assessments WHERE fire_event_id IN ({event_subquery}))"
            ),
            params,
        )
        connection.execute(
            text(
                "DELETE FROM fire_severity_assessment_satellite_inputs WHERE assessment_id IN ("
                f"SELECT id FROM fire_severity_assessments WHERE fire_event_id IN ({event_subquery}))"
            ),
            params,
        )
        connection.execute(
            text(f"DELETE FROM fire_severity_assessments WHERE fire_event_id IN ({event_subquery})"),
            params,
        )
        connection.execute(
            text(f"DELETE FROM fire_event_satellite_evidence WHERE fire_event_id IN ({event_subquery})"),
            params,
        )
        connection.execute(
            text("DELETE FROM fire_events WHERE methodology = :det_methodology AND detected_at = :detected_at"),
            params,
        )
        connection.execute(
            text(
                "DELETE FROM weather_observations WHERE station_id IN ("
                "SELECT id FROM weather_stations WHERE external_station_id = :station_id)"
            ),
            {"station_id": STATION_EXTERNAL_ID},
        )
        connection.execute(
            text("DELETE FROM weather_stations WHERE external_station_id = :station_id"),
            {"station_id": STATION_EXTERNAL_ID},
        )
        connection.execute(
            text(
                "DELETE FROM satellite_hotspots "
                "WHERE detected_at = :detected_at AND latitude = :latitude AND longitude = :longitude"
            ),
            {"detected_at": DETECTED_AT, "latitude": LATITUDE, "longitude": LONGITUDE},
        )


def make_event(**overrides) -> FireEvent:
    defaults = dict(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        detected_at=DETECTED_AT,
        updated_at=AS_OF - timedelta(minutes=5),
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
        score=60.0,
        level=FireSeverityLevel.HIGH,
        methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
        methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
        vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API",
        vegetation_dataset_year=2019,
        vegetation_radius_km=1.0,
        vegetation_dominant_land_cover=FAVORABLE_LAND_COVER,
        vegetation_fuel_score=0.8,
    )
    defaults.update(overrides)
    return FireSeverityAssessment(**defaults)


def _persist_event_with_evidence(
    fire_event_repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
    **event_overrides,
) -> tuple[int, int]:
    """Persist a FireEvent with real satellite evidence; returns (event_id, hotspot_id)."""
    satellite_repository.save_hotspot(
        SatelliteHotspot(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            detected_at=DETECTED_AT,
            confidence="h",
            frp=65.0,
            satellite="N20",
        )
    )
    hotspot_id = satellite_repository.get_recent_hotspots(as_of=AS_OF, lookback_minutes=120)[0].id
    stored = fire_event_repository.create_event(
        make_event(**event_overrides),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, hotspot_id),),
    )
    return stored.id, hotspot_id


def _persist_weather(
    weather_repository: WeatherRepository,
    *,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    temperature_c: float,
    relative_humidity_pct: float,
    timestamp: datetime,
) -> int:
    weather_repository.save_station(
        WeatherStation(
            external_station_id=STATION_EXTERNAL_ID,
            name="Spread Integration Station",
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )
    )
    weather_repository.save_observation(
        WeatherObservation(
            station_external_id=STATION_EXTERNAL_ID,
            timestamp=timestamp,
            temperature=temperature_c,
            relative_humidity=relative_humidity_pct,
            wind_speed=wind_speed_kmh,
            wind_direction=wind_direction_deg,
        )
    )
    candidates = weather_repository.get_recent_observations_for_area_candidates(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        radius_km=5.0,
        start_time=timestamp - timedelta(minutes=1),
        end_time=timestamp,
    )
    # Match by station rather than exact timestamp equality: this query path
    # can return naive datetimes on read-back (a pre-existing, already
    # elsewhere-handled characteristic of this repository, not something to
    # "fix" here), so an aware-vs-naive `==` comparison would silently never
    # match. Only one station is used per test, so this is unambiguous.
    return next(
        record.observation_id
        for record in candidates
        if record.station.external_station_id == STATION_EXTERNAL_ID
    )


def _setup_ready_scenario(
    fire_event_repository: FireEventRepository,
    satellite_repository: SatelliteHotspotRepository,
    severity_repository: FireSeverityAssessmentRepository,
    weather_repository: WeatherRepository,
    *,
    wind_speed_kmh: float,
    wind_direction_deg: float,
    temperature_c: float,
    relative_humidity_pct: float,
    land_cover: str,
    event_status: FireEventStatus = FireEventStatus.CONFIRMED,
    weather_timestamp: datetime = AS_OF - timedelta(minutes=5),
) -> tuple[int, int, int]:
    """Persist a full READY-eligible scenario; returns (event_id, assessment_id, observation_id)."""
    event_id, hotspot_id = _persist_event_with_evidence(
        fire_event_repository, satellite_repository, status=event_status
    )
    observation_id = _persist_weather(
        weather_repository,
        wind_speed_kmh=wind_speed_kmh,
        wind_direction_deg=wind_direction_deg,
        temperature_c=temperature_c,
        relative_humidity_pct=relative_humidity_pct,
        timestamp=weather_timestamp,
    )
    assessment = severity_repository.save_assessment(
        make_assessment(event_id, vegetation_dominant_land_cover=land_cover),
        weather_observation_ids=(observation_id,),
        satellite_hotspot_ids=(hotspot_id,),
        selected_frp_hotspot_id=hotspot_id,
    )
    return event_id, assessment.assessment_id, observation_id


@pytest.fixture
def fire_event_repository() -> FireEventRepository:
    return FireEventRepository()


@pytest.fixture
def satellite_repository() -> SatelliteHotspotRepository:
    return SatelliteHotspotRepository()


@pytest.fixture
def severity_repository() -> FireSeverityAssessmentRepository:
    return FireSeverityAssessmentRepository()


@pytest.fixture
def weather_repository() -> WeatherRepository:
    return WeatherRepository()


@pytest.fixture
def prediction_repository() -> FireSpreadPredictionRepository:
    return FireSpreadPredictionRepository()


@pytest.fixture
def agent(prediction_repository) -> FireSpreadPredictionAgent:
    return FireSpreadPredictionAgent(
        input_service=FireSpreadInputService(),
        calculator=FireSpreadCalculator(),
        repository=prediction_repository,
    )


# ---------------------------------------------------------------------------
# A. Core happy path (§7, §18, §20)
# ---------------------------------------------------------------------------


def test_happy_path_valid_prediction_persisted_with_full_traceability(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository, prediction_repository
):
    event_id, assessment_id, observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=FAVORABLE_WIND_SPEED_KMH,
        wind_direction_deg=FAVORABLE_WIND_DIRECTION_DEG,
        temperature_c=FAVORABLE_TEMPERATURE_C,
        relative_humidity_pct=FAVORABLE_HUMIDITY_PCT,
        land_cover=FAVORABLE_LAND_COVER,
    )

    stored = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert stored.prediction.status is FireSpreadPredictionStatus.VALID
    assert stored.prediction.fire_event_id == event_id
    assert stored.prediction.severity_assessment_id == assessment_id
    assert stored.weather_observation_id == observation_id
    assert stored.prediction.predicted_at == AS_OF
    assert stored.prediction.horizon_minutes == 30
    assert stored.prediction.methodology == METHODOLOGY_NAME
    assert stored.prediction.methodology_version == METHODOLOGY_VERSION
    assert len(stored.prediction.cells) > 0

    # Independently re-fetch to confirm real round-trip persistence, not just
    # the in-memory return value.
    refetched = prediction_repository.get_by_id(stored.id)
    assert refetched is not None
    assert refetched.prediction == stored.prediction
    assert refetched.weather_observation_id == observation_id


# ---------------------------------------------------------------------------
# B. 30/60-minute integration (§4, §17 horizon separation)
# ---------------------------------------------------------------------------


def test_horizons_30_and_60_are_consistent_and_persisted_separately(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository, prediction_repository
):
    event_id, assessment_id, observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=FAVORABLE_WIND_SPEED_KMH,
        wind_direction_deg=FAVORABLE_WIND_DIRECTION_DEG,
        temperature_c=FAVORABLE_TEMPERATURE_C,
        relative_humidity_pct=FAVORABLE_HUMIDITY_PCT,
        land_cover=FAVORABLE_LAND_COVER,
    )

    result_30 = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)
    result_60 = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=60)

    assert result_30.id != result_60.id
    assert result_30.prediction.horizon_minutes == 30
    assert result_60.prediction.horizon_minutes == 60

    # Cells reached within the first 6 steps of the 60-minute run must match
    # the 30-minute run's cells exactly (same key, probability, minutes).
    cells_60_by_key = {
        (cell.reached_step, cell.latitude, cell.longitude): cell for cell in result_60.prediction.cells
    }
    for cell in result_30.prediction.cells:
        key = (cell.reached_step, cell.latitude, cell.longitude)
        assert key in cells_60_by_key
        assert cells_60_by_key[key].spread_probability == cell.spread_probability
        assert cells_60_by_key[key].reached_minutes == cell.reached_minutes
    assert any(cell.reached_step > 6 for cell in result_60.prediction.cells)

    latest_30 = prediction_repository.get_latest_for_event_and_horizon(event_id, 30)
    latest_60 = prediction_repository.get_latest_for_event_and_horizon(event_id, 60)
    assert latest_30.id == result_30.id
    assert latest_60.id == result_60.id
    # A 30-minute save did not overwrite/affect the 60-minute row or vice versa.
    assert prediction_repository.get_by_id(result_30.id) is not None
    assert prediction_repository.get_by_id(result_60.id) is not None


# ---------------------------------------------------------------------------
# C. Determinism across the full real chain (§13)
# ---------------------------------------------------------------------------


def test_determinism_full_chain_same_inputs_produce_equivalent_output(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository
):
    event_id, _assessment_id, _observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=FAVORABLE_WIND_SPEED_KMH,
        wind_direction_deg=FAVORABLE_WIND_DIRECTION_DEG,
        temperature_c=FAVORABLE_TEMPERATURE_C,
        relative_humidity_pct=FAVORABLE_HUMIDITY_PCT,
        land_cover=FAVORABLE_LAND_COVER,
    )

    first = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)
    second = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert first.id != second.id  # each run is its own historical row
    assert first.prediction.status == second.prediction.status
    assert first.prediction.cells == second.prediction.cells
    assert first.prediction.severity_assessment_id == second.prediction.severity_assessment_id
    assert first.weather_observation_id == second.weather_observation_id


# ---------------------------------------------------------------------------
# D. History is append-only (§17)
# ---------------------------------------------------------------------------


def test_history_is_append_only_across_multiple_runs(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository, prediction_repository
):
    event_id, _assessment_id, _observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=FAVORABLE_WIND_SPEED_KMH,
        wind_direction_deg=FAVORABLE_WIND_DIRECTION_DEG,
        temperature_c=FAVORABLE_TEMPERATURE_C,
        relative_humidity_pct=FAVORABLE_HUMIDITY_PCT,
        land_cover=FAVORABLE_LAND_COVER,
    )

    earliest = agent.predict(fire_event_id=event_id, as_of=AS_OF - timedelta(minutes=10), horizon_minutes=30)
    middle = agent.predict(fire_event_id=event_id, as_of=AS_OF - timedelta(minutes=5), horizon_minutes=30)
    newest = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert len({earliest.id, middle.id, newest.id}) == 3
    assert prediction_repository.get_by_id(earliest.id) is not None
    assert prediction_repository.get_by_id(middle.id) is not None
    assert prediction_repository.get_by_id(newest.id) is not None

    latest = prediction_repository.get_latest_for_event_and_horizon(event_id, 30)
    assert latest.id == newest.id
    assert latest.prediction.predicted_at == AS_OF


# ---------------------------------------------------------------------------
# E. Missing-data acceptance (§14)
# ---------------------------------------------------------------------------


def test_missing_severity_assessment_produces_insufficient_data(
    agent, fire_event_repository, satellite_repository, prediction_repository
):
    event_id, _hotspot_id = _persist_event_with_evidence(fire_event_repository, satellite_repository)

    stored = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert stored.prediction.status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
    assert stored.prediction.cells == ()
    assert stored.prediction.severity_assessment_id is None
    assert stored.weather_observation_id is None
    assert prediction_repository.get_by_id(stored.id) is not None


def test_ambiguous_vegetation_produces_insufficient_data(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository
):
    event_id, assessment_id, _observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=FAVORABLE_WIND_SPEED_KMH,
        wind_direction_deg=FAVORABLE_WIND_DIRECTION_DEG,
        temperature_c=FAVORABLE_TEMPERATURE_C,
        relative_humidity_pct=FAVORABLE_HUMIDITY_PCT,
        land_cover="Moss and lichen cover",  # still ambiguous; "Tree cover" maps to GENERIC_TREE since Task 14
    )

    stored = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert stored.prediction.status is FireSpreadPredictionStatus.INSUFFICIENT_DATA
    assert stored.prediction.cells == ()
    # A real severity assessment was found and was VALID -- its id is still
    # known even though vegetation could not be mapped to a fuel class.
    assert stored.prediction.severity_assessment_id == assessment_id


# ---------------------------------------------------------------------------
# F. Inactive event acceptance (§15)
# ---------------------------------------------------------------------------


def test_inactive_event_produces_no_active_cells(agent, fire_event_repository, satellite_repository):
    event_id, _hotspot_id = _persist_event_with_evidence(fire_event_repository, satellite_repository)
    active_event = fire_event_repository.get_by_id(event_id).event
    fire_event_repository.update_event(
        event_id,
        FireEvent(
            latitude=active_event.latitude,
            longitude=active_event.longitude,
            detected_at=active_event.detected_at,
            updated_at=AS_OF,
            status=FireEventStatus.RESOLVED,
            detection_confidence=active_event.detection_confidence,
            methodology=active_event.methodology,
            methodology_version=active_event.methodology_version,
        ),
    )

    stored = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert stored.prediction.status is FireSpreadPredictionStatus.INACTIVE_EVENT
    assert stored.prediction.cells == ()
    assert stored.prediction.severity_assessment_id is None


# ---------------------------------------------------------------------------
# G. Valid no-spread acceptance (§16)
# ---------------------------------------------------------------------------


def test_valid_no_spread_prediction_remains_valid(
    agent, fire_event_repository, satellite_repository, severity_repository, weather_repository, prediction_repository
):
    event_id, assessment_id, observation_id = _setup_ready_scenario(
        fire_event_repository,
        satellite_repository,
        severity_repository,
        weather_repository,
        wind_speed_kmh=NO_SPREAD_WIND_SPEED_KMH,
        wind_direction_deg=NO_SPREAD_WIND_DIRECTION_DEG,
        temperature_c=NO_SPREAD_TEMPERATURE_C,
        relative_humidity_pct=NO_SPREAD_HUMIDITY_PCT,
        land_cover=NO_SPREAD_LAND_COVER,
    )

    stored = agent.predict(fire_event_id=event_id, as_of=AS_OF, horizon_minutes=30)

    assert stored.prediction.status is FireSpreadPredictionStatus.VALID
    # Methodology 1.1: no cell propagates, so only the origin's first ring
    # of risk-only cells is persisted.
    assert len(stored.prediction.cells) == 8
    assert all(cell.spread_probability < PROPAGATION_THRESHOLD for cell in stored.prediction.cells)
    assert all(cell.reached_step == 1 for cell in stored.prediction.cells)
    assert stored.prediction.severity_assessment_id == assessment_id
    assert stored.weather_observation_id == observation_id
    # Confirm it was actually persisted as VALID, not silently downgraded.
    refetched = prediction_repository.get_by_id(stored.id)
    assert refetched.prediction.status is FireSpreadPredictionStatus.VALID
