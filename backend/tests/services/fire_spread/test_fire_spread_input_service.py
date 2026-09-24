"""Tests for FireSpreadInputService."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.calculators.fire_danger.ffwi_calculator import _celsius_to_fahrenheit, _equilibrium_moisture_content
from src.calculators.fire_severity.fire_severity_config import (
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models import (
    FireEvent,
    FireEventStatus,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    FireSpreadFuelClass,
    FireSpreadInputStatus,
)
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation
from src.repositories.fire_event_repository import StoredFireEvent
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment
from src.repositories.weather_repository import StoredWeatherObservation
from src.services.fire_spread.fire_spread_input_service import (
    FireSpreadInputService,
    _map_dominant_land_cover,
)
from src.services.fire_severity.fire_severity_input_config import MAX_SEVERITY_WEATHER_AGE_MINUTES

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
EVENT_LAT = 32.731
EVENT_LON = 35.046

_UNSET = object()


class FakeFireEventRepository:
    def __init__(self, stored_event: StoredFireEvent | None) -> None:
        self.stored_event = stored_event
        self.calls: list[int] = []

    def get_by_id(self, fire_event_id: int):
        self.calls.append(fire_event_id)
        return self.stored_event


class FakeFireSeverityAssessmentRepository:
    """Mirrors FireSeverityAssessmentRepository.get_latest_for_event_as_of's SQL
    semantics: same fire_event_id, assessed_at <= as_of, newest assessed_at
    first, ties broken by highest assessment_id.

    Accepts a single StoredFireSeverityAssessment (the common case), None, or
    an iterable of several (for as_of/historical-selection tests).
    """

    def __init__(self, stored_assessment) -> None:
        if stored_assessment is None:
            self.stored_assessments: tuple[StoredFireSeverityAssessment, ...] = ()
        elif isinstance(stored_assessment, StoredFireSeverityAssessment):
            self.stored_assessments = (stored_assessment,)
        else:
            self.stored_assessments = tuple(stored_assessment)
        self.calls: list[tuple[int, datetime]] = []

    def get_latest_for_event_as_of(self, fire_event_id: int, as_of: datetime):
        self.calls.append((fire_event_id, as_of))
        eligible = [
            stored
            for stored in self.stored_assessments
            if stored.assessment.fire_event_id == fire_event_id and stored.assessment.assessed_at <= as_of
        ]
        if not eligible:
            return None
        eligible.sort(key=lambda stored: (stored.assessment.assessed_at, stored.assessment_id), reverse=True)
        return eligible[0]


class FakeWeatherRepository:
    def __init__(self, records: list[StoredWeatherObservation]) -> None:
        self.records = records
        self.calls: list[tuple[int, ...]] = []

    def get_observations_by_ids(self, observation_ids):
        self.calls.append(tuple(observation_ids))
        requested = set(observation_ids)
        return tuple(record for record in self.records if record.observation_id in requested)


def make_event(status=FireEventStatus.SUSPECTED, fire_event_id=10) -> StoredFireEvent:
    return StoredFireEvent(
        id=fire_event_id,
        event=FireEvent(
            latitude=EVENT_LAT,
            longitude=EVENT_LON,
            detected_at=AS_OF - timedelta(minutes=40),
            updated_at=AS_OF - timedelta(minutes=5),
            status=status,
            detection_confidence=0.75,
            methodology="ECOGUARD_MULTI_SOURCE_DETECTION",
            methodology_version="1.0",
        ),
        supporting_evidence=(),
    )


def make_assessment(
    assessment_id=500,
    fire_event_id=10,
    status=FireSeverityAssessmentStatus.VALID,
    assessed_at=None,
    dominant_land_cover="Shrub cover",
    weather_observation_ids=(101,),
) -> StoredFireSeverityAssessment:
    is_valid = status is FireSeverityAssessmentStatus.VALID
    return StoredFireSeverityAssessment(
        assessment_id=assessment_id,
        assessment=FireSeverityAssessment(
            fire_event_id=fire_event_id,
            assessed_at=assessed_at or (AS_OF - timedelta(minutes=10)),
            status=status,
            score=55.0 if is_valid else None,
            level=FireSeverityLevel.HIGH if is_valid else None,
            methodology=FIRE_SEVERITY_METHODOLOGY_NAME,
            methodology_version=FIRE_SEVERITY_METHODOLOGY_VERSION,
            vegetation_source="COPERNICUS_GLOBAL_LAND_COVER_100M_API" if is_valid else None,
            vegetation_dataset_year=2019 if is_valid else None,
            vegetation_radius_km=1.0 if is_valid else None,
            vegetation_dominant_land_cover=dominant_land_cover if is_valid else None,
            vegetation_fuel_score=0.8 if is_valid and dominant_land_cover is not None else None,
        ),
        weather_observation_ids=weather_observation_ids,
        satellite_hotspot_ids=(1,),
        selected_frp_hotspot_id=1,
    )


def make_weather(
    observation_id=101,
    station_id=1,
    latitude=EVENT_LAT,
    longitude=EVENT_LON,
    minutes_old=5,
    temperature=28.0,
    relative_humidity=35.0,
    wind_speed=6.0,
    wind_direction=270.0,
) -> StoredWeatherObservation:
    external_station_id = 900000 + station_id
    return StoredWeatherObservation(
        observation_id=observation_id,
        station_id=station_id,
        station=WeatherStation(
            external_station_id=external_station_id,
            name=f"Station {station_id}",
            latitude=latitude,
            longitude=longitude,
        ),
        observation=WeatherObservation(
            station_external_id=external_station_id,
            timestamp=AS_OF - timedelta(minutes=minutes_old),
            temperature=temperature,
            relative_humidity=relative_humidity,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
        ),
    )


def build_service(
    *,
    stored_event=_UNSET,
    stored_assessment=_UNSET,
    weather_records=None,
):
    event_repo = FakeFireEventRepository(make_event() if stored_event is _UNSET else stored_event)
    assessment_repo = FakeFireSeverityAssessmentRepository(
        make_assessment() if stored_assessment is _UNSET else stored_assessment
    )
    weather_repo = FakeWeatherRepository(weather_records if weather_records is not None else [make_weather()])
    service = FireSpreadInputService(
        fire_event_repository=event_repo,
        fire_severity_assessment_repository=assessment_repo,
        weather_repository=weather_repo,
    )
    return service, event_repo, assessment_repo, weather_repo


# ---------------------------------------------------------------------------
# READY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [FireEventStatus.SUSPECTED, FireEventStatus.CONFIRMED])
def test_active_fire_event_may_be_ready(status):
    service, *_ = build_service(stored_event=make_event(status=status))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.input_data is not None


@pytest.mark.parametrize("horizon_minutes", [30, 60])
def test_supported_horizons_produce_ready_input(horizon_minutes):
    service, *_ = build_service()

    result = service.prepare_input(10, AS_OF, horizon_minutes=horizon_minutes)

    assert result.status is FireSpreadInputStatus.READY
    assert result.input_data.horizon_minutes == horizon_minutes


def test_ready_result_traceability_matches_stored_ids():
    service, *_ = build_service()

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.fire_event_id == 10
    assert result.severity_assessment_id == 500
    assert result.weather_observation_id == 101


def test_ready_input_uses_event_origin():
    service, *_ = build_service()

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.input_data.origin_latitude == EVENT_LAT
    assert result.input_data.origin_longitude == EVENT_LON


def test_stored_km_h_wind_speed_is_passed_through_without_conversion():
    # WeatherObservation.wind_speed is canonically km/h: 36 stays 36, never 129.6.
    service, *_ = build_service(weather_records=[make_weather(wind_speed=36.0)])

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.input_data.wind_speed_kmh == pytest.approx(36.0)


def test_wind_direction_preserved_unmodified():
    service, *_ = build_service(weather_records=[make_weather(wind_direction=123.0)])

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.input_data.wind_direction_deg == 123.0


def test_moisture_matches_existing_ffwi_helper_exactly():
    service, *_ = build_service(weather_records=[make_weather(temperature=25.0, relative_humidity=40.0)])

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    expected = _equilibrium_moisture_content(
        relative_humidity_pct=40.0, temperature_f=_celsius_to_fahrenheit(25.0)
    )
    assert result.input_data.fuel_moisture_percent == pytest.approx(expected)


def test_fuel_class_mapped_from_dominant_land_cover():
    service, *_ = build_service(stored_assessment=make_assessment(dominant_land_cover="Grass cover"))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.input_data.fuel_class is FireSpreadFuelClass.GRASSLAND


def test_nearest_of_multiple_eligible_stations_is_selected():
    near = make_weather(observation_id=101, station_id=1, latitude=EVENT_LAT, longitude=EVENT_LON, wind_speed=5.0)
    far = make_weather(observation_id=102, station_id=2, latitude=EVENT_LAT + 1.0, longitude=EVENT_LON, wind_speed=99.0)
    service, *_ = build_service(
        stored_assessment=make_assessment(weather_observation_ids=(101, 102)),
        weather_records=[far, near],
    )

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.weather_observation_id == 101
    assert result.input_data.wind_speed_kmh == pytest.approx(5.0)


def test_deterministic_tiebreak_distance_then_newest_then_id():
    # Same distance (both at event location); newer timestamp wins.
    older = make_weather(observation_id=101, station_id=1, minutes_old=20)
    newer = make_weather(observation_id=102, station_id=2, minutes_old=5)
    service, *_ = build_service(
        stored_assessment=make_assessment(weather_observation_ids=(101, 102)),
        weather_records=[older, newer],
    )

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.weather_observation_id == 102


# ---------------------------------------------------------------------------
# INACTIVE_EVENT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_inactive_fire_event_returns_inactive_without_loading_severity_or_weather(status):
    service, _, assessment_repo, weather_repo = build_service(stored_event=make_event(status=status))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INACTIVE_EVENT
    assert result.input_data is None
    assert assessment_repo.calls == []
    assert weather_repo.calls == []


# ---------------------------------------------------------------------------
# INSUFFICIENT_DATA
# ---------------------------------------------------------------------------


def test_missing_fire_event_is_insufficient():
    service, *_ = build_service(stored_event=None)

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.input_data is None


def test_missing_severity_assessment_is_insufficient():
    service, *_ = build_service(stored_assessment=None)

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.severity_assessment_id is None


@pytest.mark.parametrize(
    "status", [FireSeverityAssessmentStatus.INSUFFICIENT_DATA, FireSeverityAssessmentStatus.INACTIVE_EVENT]
)
def test_non_valid_severity_assessment_is_insufficient(status):
    service, *_ = build_service(stored_assessment=make_assessment(status=status))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.severity_assessment_id == 500


def test_future_severity_assessment_relative_to_as_of_is_insufficient():
    service, *_ = build_service(
        stored_assessment=make_assessment(assessed_at=AS_OF + timedelta(minutes=1))
    )

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_no_weather_traceability_on_assessment_is_insufficient():
    service, *_ = build_service(stored_assessment=make_assessment(weather_observation_ids=()))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_missing_referenced_observation_is_insufficient():
    service, *_ = build_service(weather_records=[])  # id 101 referenced but not returned

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_weather_older_than_freshness_policy_is_insufficient():
    service, *_ = build_service(
        weather_records=[make_weather(minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 0.1)]
    )

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_weather_exactly_at_freshness_boundary_is_ready():
    service, *_ = build_service(weather_records=[make_weather(minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES)])

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.READY


def test_future_weather_observation_is_insufficient():
    service, *_ = build_service(weather_records=[make_weather(minutes_old=-1)])

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.INSUFFICIENT_DATA


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": None},
        {"relative_humidity": None},
        {"wind_speed": None},
        {"wind_direction": None},
    ],
)
def test_missing_mandatory_weather_field_is_insufficient(overrides):
    service, *_ = build_service(weather_records=[make_weather(**overrides)])

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_missing_vegetation_snapshot_is_insufficient():
    service, *_ = build_service(stored_assessment=make_assessment(dominant_land_cover=None))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.weather_observation_id == 101


@pytest.mark.parametrize("label", ["Tree cover", "Moss and lichen cover"])
def test_ambiguous_vegetation_label_is_insufficient(label):
    service, *_ = build_service(stored_assessment=make_assessment(dominant_land_cover=label))

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.INSUFFICIENT_DATA


def test_unknown_vegetation_label_is_insufficient():
    service, *_ = build_service(stored_assessment=make_assessment(dominant_land_cover="Some New Category"))

    assert service.prepare_input(10, AS_OF, horizon_minutes=30).status is FireSpreadInputStatus.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# insufficient_data_reason
# ---------------------------------------------------------------------------

from src.models import FireSpreadInsufficientDataReason as Reason  # noqa: E402


@pytest.mark.parametrize(
    ("service_kwargs", "expected_reason"),
    [
        pytest.param(dict(stored_event=None), Reason.EVENT_UNAVAILABLE, id="missing-event"),
        pytest.param(dict(stored_assessment=None), Reason.MISSING_SEVERITY, id="missing-severity"),
        pytest.param(
            dict(stored_assessment=make_assessment(assessed_at=AS_OF + timedelta(minutes=1))),
            Reason.MISSING_SEVERITY,
            id="only-future-severity",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(fire_event_id=99)),
            Reason.MISSING_SEVERITY,
            id="severity-of-other-event",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(status=FireSeverityAssessmentStatus.INSUFFICIENT_DATA)),
            Reason.SEVERITY_NOT_VALID,
            id="severity-insufficient",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(status=FireSeverityAssessmentStatus.INACTIVE_EVENT)),
            Reason.SEVERITY_NOT_VALID,
            id="severity-inactive",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(weather_observation_ids=())),
            Reason.MISSING_WEATHER,
            id="no-linked-weather",
        ),
        pytest.param(dict(weather_records=[]), Reason.MISSING_WEATHER, id="linked-weather-not-retrievable"),
        pytest.param(
            dict(weather_records=[make_weather(wind_direction=None)]),
            Reason.INCOMPLETE_WEATHER,
            id="incomplete-weather",
        ),
        pytest.param(
            dict(weather_records=[make_weather(minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 0.1)]),
            Reason.STALE_WEATHER,
            id="stale-weather",
        ),
        pytest.param(dict(weather_records=[make_weather(minutes_old=-1)]), Reason.FUTURE_WEATHER, id="future-weather"),
        pytest.param(
            dict(stored_assessment=make_assessment(dominant_land_cover=None)),
            Reason.MISSING_VEGETATION,
            id="missing-vegetation",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(dominant_land_cover="Tree cover")),
            Reason.UNSUPPORTED_VEGETATION,
            id="tree-cover",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(dominant_land_cover="Moss and lichen cover")),
            Reason.UNSUPPORTED_VEGETATION,
            id="moss-and-lichen",
        ),
        pytest.param(
            dict(stored_assessment=make_assessment(dominant_land_cover="Some New Category")),
            Reason.UNSUPPORTED_VEGETATION,
            id="unknown-vegetation",
        ),
    ],
)
def test_insufficient_data_carries_the_exact_reason(service_kwargs, expected_reason):
    service, *_ = build_service(**service_kwargs)

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.insufficient_data_reason is expected_reason


def _weather_precedence_service(*records):
    return build_service(
        stored_assessment=make_assessment(weather_observation_ids=tuple(r.observation_id for r in records)),
        weather_records=list(records),
    )[0]


@pytest.mark.parametrize(
    ("records", "expected_reason"),
    [
        pytest.param(
            (
                make_weather(observation_id=101, temperature=None),
                make_weather(observation_id=102, station_id=2, minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 5),
            ),
            Reason.STALE_WEATHER,
            id="incomplete-plus-stale-is-stale",
        ),
        pytest.param(
            (
                make_weather(observation_id=101, temperature=None),
                make_weather(observation_id=102, station_id=2, minutes_old=-5),
            ),
            Reason.FUTURE_WEATHER,
            id="incomplete-plus-future-is-future",
        ),
        pytest.param(
            (
                make_weather(observation_id=101, minutes_old=-5),
                make_weather(observation_id=102, station_id=2, minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 5),
            ),
            Reason.STALE_WEATHER,
            id="future-plus-stale-is-stale",
        ),
        pytest.param(
            (
                make_weather(observation_id=101, temperature=None),
                make_weather(observation_id=102, station_id=2, wind_speed=None),
            ),
            Reason.INCOMPLETE_WEATHER,
            id="all-incomplete-is-incomplete",
        ),
    ],
)
def test_weather_reason_precedence(records, expected_reason):
    result = _weather_precedence_service(*records).prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.insufficient_data_reason is expected_reason


def test_one_usable_observation_still_selected_despite_other_invalid_ones():
    service = _weather_precedence_service(
        make_weather(observation_id=101, temperature=None),
        make_weather(observation_id=102, station_id=2, minutes_old=MAX_SEVERITY_WEATHER_AGE_MINUTES + 5),
        make_weather(observation_id=103, station_id=3, minutes_old=5),
    )

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.weather_observation_id == 103
    assert result.insufficient_data_reason is None


@pytest.mark.parametrize("status", [FireEventStatus.RESOLVED, FireEventStatus.DISMISSED])
def test_ready_and_inactive_results_have_no_reason(status):
    ready_service, *_ = build_service()
    inactive_service, *_ = build_service(stored_event=make_event(status=status))

    assert ready_service.prepare_input(10, AS_OF, horizon_minutes=30).insufficient_data_reason is None
    inactive = inactive_service.prepare_input(10, AS_OF, horizon_minutes=30)
    assert inactive.status is FireSpreadInputStatus.INACTIVE_EVENT
    assert inactive.insufficient_data_reason is None


# ---------------------------------------------------------------------------
# Historical as_of selection (FND-08 regression)
# ---------------------------------------------------------------------------


def test_historical_as_of_selects_older_valid_severity_not_global_latest():
    """Regression for FND-08 (Test A): S0 at T0, S1 at T2, request at T1 with
    T0 < T1 < T2. S0 must be selected and used for weather/fuel traceability;
    the service must not reject the request merely because a newer
    assessment (S1) exists after the requested as_of."""
    t0 = AS_OF - timedelta(minutes=30)
    t1 = AS_OF
    t2 = AS_OF + timedelta(minutes=30)
    s0 = make_assessment(
        assessment_id=500, assessed_at=t0, weather_observation_ids=(101,), dominant_land_cover="Shrub cover"
    )
    s1 = make_assessment(
        assessment_id=501, assessed_at=t2, weather_observation_ids=(102,), dominant_land_cover="Grass cover"
    )
    service, _, assessment_repo, _ = build_service(
        stored_assessment=(s0, s1),
        weather_records=[make_weather(observation_id=101, minutes_old=5)],
    )

    result = service.prepare_input(10, t1, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.severity_assessment_id == 500
    assert result.weather_observation_id == 101
    assert result.input_data.fuel_class is FireSpreadFuelClass.SHRUBS
    assert assessment_repo.calls == [(10, t1)]


def test_severity_assessed_exactly_at_as_of_is_eligible():
    """Regression for FND-08 (Test B): assessed_at <= as_of, not strict <."""
    assessment = make_assessment(assessment_id=500, assessed_at=AS_OF, weather_observation_ids=(101,))
    service, *_ = build_service(
        stored_assessment=assessment,
        weather_records=[make_weather(observation_id=101, minutes_old=5)],
    )

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.severity_assessment_id == 500


def test_only_future_severity_assessments_yields_insufficient_data():
    """Regression for FND-08 (Test C): every assessment postdates as_of, so
    none is eligible. The service must not fabricate an older severity."""
    s1 = make_assessment(assessment_id=501, assessed_at=AS_OF + timedelta(minutes=10))
    s2 = make_assessment(assessment_id=502, assessed_at=AS_OF + timedelta(minutes=40))
    service, *_ = build_service(stored_assessment=(s1, s2))

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.severity_assessment_id is None


def test_latest_of_several_historical_assessments_before_as_of_is_selected():
    """Regression for FND-08 (Test D): S0 at T0, S1 at T1, S2 at T2 (T2 is
    after the request). Requesting between T1 and T2 must select S1, the
    newest assessment that is still at or before as_of -- not S0 and not
    the globally-latest S2."""
    t0 = AS_OF - timedelta(minutes=60)
    t1 = AS_OF - timedelta(minutes=30)
    t2 = AS_OF + timedelta(minutes=30)
    request_time = AS_OF - timedelta(minutes=10)  # strictly between t1 and t2
    s0 = make_assessment(assessment_id=500, assessed_at=t0, weather_observation_ids=(101,))
    s1 = make_assessment(assessment_id=501, assessed_at=t1, weather_observation_ids=(102,))
    s2 = make_assessment(assessment_id=502, assessed_at=t2, weather_observation_ids=(103,))
    service, *_ = build_service(
        stored_assessment=(s0, s1, s2),
        weather_records=[make_weather(observation_id=102, minutes_old=15)],
    )

    result = service.prepare_input(10, request_time, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.READY
    assert result.severity_assessment_id == 501
    assert result.weather_observation_id == 102


def test_assessment_for_a_different_fire_event_is_never_selected():
    """A severity assessment that qualifies on assessed_at <= as_of but
    belongs to a different FireEvent must never be treated as eligible."""
    other_event_assessment = make_assessment(
        assessment_id=999, fire_event_id=99, assessed_at=AS_OF - timedelta(minutes=5)
    )
    service, *_ = build_service(stored_assessment=other_event_assessment)

    result = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert result.status is FireSpreadInputStatus.INSUFFICIENT_DATA
    assert result.severity_assessment_id is None


# ---------------------------------------------------------------------------
# Determinism and validation
# ---------------------------------------------------------------------------


def test_same_inputs_produce_identical_result():
    service, *_ = build_service()

    first = service.prepare_input(10, AS_OF, horizon_minutes=30)
    second = service.prepare_input(10, AS_OF, horizon_minutes=30)

    assert first == second


def test_prepare_input_requires_timezone_aware_as_of():
    service, *_ = build_service()

    with pytest.raises(ValueError):
        service.prepare_input(10, datetime(2026, 9, 14, 12, 0), horizon_minutes=30)


@pytest.mark.parametrize("horizon_minutes", [0, 15, 45, 90, True])
def test_prepare_input_rejects_unsupported_horizon(horizon_minutes):
    service, *_ = build_service()

    with pytest.raises(ValueError):
        service.prepare_input(10, AS_OF, horizon_minutes=horizon_minutes)


def test_prepare_input_rejects_invalid_fire_event_id():
    service, *_ = build_service()

    with pytest.raises(ValueError):
        service.prepare_input(0, AS_OF, horizon_minutes=30)


# ---------------------------------------------------------------------------
# Vegetation mapping (pure function, exhaustive label coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Shrub cover", FireSpreadFuelClass.SHRUBS),
        ("Grass cover", FireSpreadFuelClass.GRASSLAND),
        ("Crop cover", FireSpreadFuelClass.AGRO_FORESTRY),
        ("Bare cover", FireSpreadFuelClass.BARE_SOIL),
        ("Built-up cover", FireSpreadFuelClass.BARE_SOIL),
        ("Permanent water cover", FireSpreadFuelClass.BARE_SOIL),
        ("Seasonal water cover", FireSpreadFuelClass.BARE_SOIL),
        ("Snow cover", FireSpreadFuelClass.BARE_SOIL),
    ],
)
def test_unambiguous_labels_map_to_expected_fuel_class(label, expected):
    assert _map_dominant_land_cover(label) is expected


@pytest.mark.parametrize("label", ["Tree cover", "Moss and lichen cover", "Unknown Label", ""])
def test_ambiguous_or_unknown_labels_return_none(label):
    assert _map_dominant_land_cover(label) is None


def test_missing_label_returns_none():
    assert _map_dominant_land_cover(None) is None
