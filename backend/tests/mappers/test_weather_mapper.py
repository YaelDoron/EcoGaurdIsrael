"""Unit tests for WeatherMapper.

All inputs are deterministic dictionaries - no IMS/network/database access,
no API token, and no reliance on system time.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.mappers.exceptions import MissingRequiredWeatherFieldError, WeatherMappingError
from src.mappers.weather_mapper import WeatherMapper
from src.models.weather_observation import WeatherObservation
from src.models.weather_station import WeatherStation

RAW_STATION = {
    "stationId": 17,
    "name": "HAIFA",
    "active": True,
    "location": {
        "latitude": 32.79,
        "longitude": 34.99,
    },
    "regionId": 3,
    "monitors": [],
}

RAW_DATETIME = "2026-09-02T12:30:00+03:00"


def _wrapped(record: dict, station_id: int = 17) -> dict:
    """Build the real IMS `/data/latest` shape: `{"stationId", "data": [record]}`."""
    return {"stationId": station_id, "data": [record]}


RAW_OBSERVATION = _wrapped(
    {
        "datetime": RAW_DATETIME,
        "channels": [
            {"name": "TD", "value": 31.4, "valid": True},
            {"name": "RH", "value": 42, "valid": True},
            {"name": "WS", "value": 5.8, "valid": True},
            {"name": "WD", "value": 240, "valid": True},
            {"name": "WSmax", "value": 8.2, "valid": True},
            {"name": "Rain", "value": 0, "valid": True},
        ],
    }
)


def _channels(*entries):
    return _wrapped({"datetime": RAW_DATETIME, "channels": list(entries)})


# ---------------------------------------------------------------------------
# Station mapping
# ---------------------------------------------------------------------------


def test_map_station_valid_raw_station_produces_correct_weather_station():
    station = WeatherMapper.map_station(RAW_STATION)

    assert station == WeatherStation(
        external_station_id=17,
        name="HAIFA",
        latitude=32.79,
        longitude=34.99,
        region_id=3,
        active=True,
    )


def test_map_station_missing_region_id_becomes_none():
    raw = dict(RAW_STATION)
    del raw["regionId"]

    station = WeatherMapper.map_station(raw)

    assert station.region_id is None


def test_map_station_region_id_zero_becomes_none():
    # The real IMS API returns regionId 0 for some active stations (e.g. NAZARET_1m).
    station = WeatherMapper.map_station({**RAW_STATION, "regionId": 0})

    assert station.region_id is None


@pytest.mark.parametrize("region_id", [1, 3, 8])
def test_map_station_positive_region_id_is_unchanged(region_id):
    station = WeatherMapper.map_station({**RAW_STATION, "regionId": region_id})

    assert station.region_id == region_id


def test_map_station_negative_region_id_still_raises_mapping_error():
    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_station({**RAW_STATION, "regionId": -1})


def test_map_station_missing_active_becomes_none():
    raw = dict(RAW_STATION)
    del raw["active"]

    station = WeatherMapper.map_station(raw)

    assert station.active is None


def test_map_station_missing_station_id_raises_mapping_error():
    raw = dict(RAW_STATION)
    del raw["stationId"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_name_raises_mapping_error():
    raw = dict(RAW_STATION)
    del raw["name"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_latitude_raises_mapping_error():
    raw = {**RAW_STATION, "location": {"longitude": 34.99}}

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_missing_longitude_raises_mapping_error():
    raw = {**RAW_STATION, "location": {"latitude": 32.79}}

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_station(raw)


def test_map_station_invalid_coordinates_raise_mapping_error():
    raw = {**RAW_STATION, "location": {"latitude": 999, "longitude": 34.99}}

    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_station(raw)


def test_map_stations_maps_a_list_of_raw_stations():
    raw_list = [RAW_STATION, {**RAW_STATION, "stationId": 21, "name": "SECOND"}]

    stations = WeatherMapper.map_stations(raw_list)

    assert [s.external_station_id for s in stations] == [17, 21]


# ---------------------------------------------------------------------------
# Observation mapping
# ---------------------------------------------------------------------------


def test_map_observation_valid_raw_observation_maps_all_six_channels():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert observation.station_external_id == 17
    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42
    # IMS WS/WSmax are m/s; WeatherObservation stores canonical km/h.
    assert observation.wind_speed == pytest.approx(5.8 * 3.6)  # 20.88 km/h
    assert observation.wind_direction == 240
    assert observation.wind_gust == pytest.approx(8.2 * 3.6)  # 29.52 km/h
    assert observation.rainfall == 0


def test_map_observation_real_shape_fixture_maps_channels_and_converts_wind():
    fixtures_path = Path(__file__).resolve().parents[1] / "fixtures" / "ims_sample_response.json"
    raw = json.loads(fixtures_path.read_text(encoding="utf-8"))["observation"]

    observation = WeatherMapper.map_observation(raw)

    assert observation.station_external_id == 17
    assert observation.timestamp == datetime(2026, 9, 2, 12, 30, tzinfo=timezone(timedelta(hours=3)))
    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42.0
    assert observation.wind_speed == pytest.approx(5.8 * 3.6)
    assert observation.wind_direction == 240.0
    assert observation.wind_gust == pytest.approx(8.2 * 3.6)
    assert observation.rainfall == 0.0


@pytest.mark.parametrize(
    ("channel", "field", "value_ms", "expected_kmh"),
    [("WS", "wind_speed", 10, 36.0), ("WSmax", "wind_gust", 15, 54.0), ("WS", "wind_speed", "10", 36.0)],
)
def test_map_observation_converts_ims_wind_channels_from_ms_to_kmh(channel, field, value_ms, expected_kmh):
    observation = WeatherMapper.map_observation(_channels({"name": channel, "value": value_ms, "valid": True}))

    assert getattr(observation, field) == pytest.approx(expected_kmh)


def test_map_observation_non_wind_channels_are_not_unit_converted():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert (observation.temperature, observation.relative_humidity, observation.wind_direction, observation.rainfall) == (
        31.4,
        42,
        240,
        0,
    )


def test_map_observation_missing_or_invalid_ws_stays_none():
    missing = WeatherMapper.map_observation(_channels({"name": "TD", "value": 31.4, "valid": True}))
    invalid = WeatherMapper.map_observation(_channels({"name": "WS", "value": 5.8, "valid": False}))
    unparseable = WeatherMapper.map_observation(_channels({"name": "WS", "value": "calm", "valid": True}))

    assert missing.wind_speed is None
    assert invalid.wind_speed is None
    assert unparseable.wind_speed is None


def test_map_observation_missing_wsmax_channel_becomes_none():
    raw = _channels({"name": "TD", "value": 31.4, "valid": True})

    observation = WeatherMapper.map_observation(raw)

    assert observation.wind_gust is None


def test_map_observation_wsmax_invalid_becomes_none():
    raw = _channels({"name": "WSmax", "value": 8.2, "valid": False})

    observation = WeatherMapper.map_observation(raw)

    assert observation.wind_gust is None


def test_map_observation_td_invalid_becomes_none_even_with_value_present():
    raw = _channels({"name": "TD", "value": 35.2, "valid": False})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_unknown_channel_is_ignored():
    raw = _channels(
        {"name": "TD", "value": 31.4, "valid": True},
        {"name": "Pressure", "value": 1013, "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4
    # No unexpected attribute should exist for the unknown channel.
    assert not hasattr(observation, "pressure")


def test_map_observation_missing_station_id_raises_mapping_error():
    raw = dict(RAW_OBSERVATION)
    del raw["stationId"]

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation(raw)


def test_map_observation_missing_datetime_raises_mapping_error():
    raw = _wrapped({"channels": RAW_OBSERVATION["data"][0]["channels"]})

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation(raw)


def test_map_observation_invalid_datetime_raises_mapping_error():
    raw = _wrapped({**RAW_OBSERVATION["data"][0], "datetime": "not-a-timestamp"})

    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_observation(raw)


def test_map_observation_reads_station_id_top_level_and_record_from_data_0():
    raw = {
        "stationId": 2,
        "data": [
            {
                "datetime": "2026-09-29T14:10:00+03:00",
                "channels": [
                    {"id": 7, "name": "TD", "alias": None, "value": 26.2, "status": 1, "valid": True, "description": None},
                    {"id": 8, "name": "RH", "alias": None, "value": 54.0, "status": 1, "valid": True, "description": None},
                ],
            }
        ],
    }

    observation = WeatherMapper.map_observation(raw)

    assert observation.station_external_id == 2
    assert observation.timestamp == datetime(2026, 9, 29, 14, 10, tzinfo=timezone(timedelta(hours=3)))
    assert observation.temperature == 26.2
    assert observation.relative_humidity == 54.0


def test_map_observation_missing_data_raises_mapping_error():
    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation({"stationId": 17})


def test_map_observation_empty_data_raises_mapping_error():
    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation({"stationId": 17, "data": []})


@pytest.mark.parametrize("data", [{"datetime": RAW_DATETIME, "channels": []}, "oops", 5])
def test_map_observation_non_list_data_raises_mapping_error(data):
    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_observation({"stationId": 17, "data": data})


def test_map_observation_non_object_data_record_raises_mapping_error():
    with pytest.raises(WeatherMappingError):
        WeatherMapper.map_observation({"stationId": 17, "data": ["oops"]})


def test_map_observation_legacy_flat_shape_is_rejected():
    flat = {"stationId": 17, "datetime": RAW_DATETIME, "channels": [{"name": "TD", "value": 31.4, "valid": True}]}

    with pytest.raises(MissingRequiredWeatherFieldError):
        WeatherMapper.map_observation(flat)


def test_map_observation_real_invalid_sentinel_channels_are_ignored():
    # Real IMS marks bad readings `valid: false` with sentinel values like -999.
    raw = _channels(
        {"id": 7, "name": "TD", "value": -999.0, "status": 2, "valid": False},
        {"id": 8, "name": "RH", "value": 54.0, "status": 1, "valid": True},
        {"id": 25, "name": "TW", "value": -999.0, "status": 2, "valid": False},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None
    assert observation.relative_humidity == 54.0


def test_map_observation_numeric_string_values_are_converted():
    raw = _channels(
        {"name": "TD", "value": "31.4", "valid": True},
        {"name": "RH", "value": "42", "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4
    assert observation.relative_humidity == 42.0


def test_map_observation_invalid_numeric_value_becomes_none():
    raw = _channels({"name": "TD", "value": "abc", "valid": True})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_missing_channels_key_does_not_fail():
    raw = _wrapped({"datetime": RAW_DATETIME})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None
    assert observation.relative_humidity is None
    assert observation.wind_speed is None
    assert observation.wind_direction is None
    assert observation.wind_gust is None
    assert observation.rainfall is None


def test_map_observation_timestamp_type_is_datetime():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert isinstance(observation.timestamp, datetime)
    assert observation.timestamp == datetime(2026, 9, 2, 12, 30, 0, tzinfo=timezone(timedelta(hours=3)))


@pytest.mark.parametrize(
    ("raw_datetime", "offset_hours"),
    [("2026-09-29T14:10:00+03:00", 3), ("2026-02-10T14:10:00+02:00", 2)],
)
def test_map_observation_israel_offset_timestamp_stays_timezone_aware(raw_datetime, offset_hours):
    raw = _wrapped({"datetime": raw_datetime, "channels": []})

    observation = WeatherMapper.map_observation(raw)

    assert observation.timestamp.tzinfo is not None
    assert observation.timestamp.utcoffset() == timedelta(hours=offset_hours)
    # Same instant as IMS reported - no conversion applied.
    assert observation.timestamp == datetime.fromisoformat(raw_datetime)


def test_map_observation_duplicate_channel_uses_last_valid_occurrence():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": True},
        {"name": "TD", "value": 25.0, "valid": True},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 25.0


def test_map_observation_duplicate_channel_later_invalid_keeps_earlier_valid_value():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": True},
        {"name": "TD", "value": 25.0, "valid": False},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 20.0


def test_map_observation_no_valid_occurrence_of_duplicate_channel_is_none():
    raw = _channels(
        {"name": "TD", "value": 20.0, "valid": False},
        {"name": "TD", "value": 25.0, "valid": False},
    )

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature is None


def test_map_observation_missing_valid_key_with_value_is_treated_as_valid():
    raw = _channels({"name": "TD", "value": 31.4})

    observation = WeatherMapper.map_observation(raw)

    assert observation.temperature == 31.4


def test_map_observation_returns_weather_observation_instance():
    observation = WeatherMapper.map_observation(RAW_OBSERVATION)

    assert isinstance(observation, WeatherObservation)


# ---------------------------------------------------------------------------
# Cross-source wind-unit consistency: IMS (m/s) and simulation (km/h) must
# reach every downstream consumer as the same canonical km/h value.
# ---------------------------------------------------------------------------

from datetime import timedelta, timezone  # noqa: E402

from src.calculators.fire_detection.fire_detection_config import (  # noqa: E402
    FIRE_DETECTION_METHODOLOGY_NAME,
    FIRE_DETECTION_METHODOLOGY_VERSION,
)
from src.calculators.fire_severity.fire_severity_config import (  # noqa: E402
    FIRE_SEVERITY_METHODOLOGY_NAME,
    FIRE_SEVERITY_METHODOLOGY_VERSION,
)
from src.models import (  # noqa: E402
    FireEvent,
    FireEventStatus,
    FireEvidenceRef,
    FireEvidenceType,
    FireSeverityAssessment,
    FireSeverityAssessmentStatus,
    FireSeverityLevel,
    SatelliteHotspot,
)
from src.models.assessment_area import AssessmentArea  # noqa: E402
from src.repositories.fire_event_repository import StoredFireEvent  # noqa: E402
from src.repositories.fire_severity_assessment_repository import StoredFireSeverityAssessment  # noqa: E402
from src.repositories.satellite_hotspot_repository import StoredSatelliteHotspot  # noqa: E402
from src.repositories.weather_repository import StoredWeatherObservation  # noqa: E402
from src.services.fire_danger.fire_danger_input_service import FireDangerInputService  # noqa: E402
from src.services.fire_severity.fire_severity_input_service import FireSeverityInputService  # noqa: E402
from src.services.fire_spread.fire_spread_input_service import FireSpreadInputService  # noqa: E402
from src.simulation.generators.weather_data_generator import WeatherDataGenerator  # noqa: E402
from src.simulation.scenario_type import ScenarioType  # noqa: E402

CROSS_SOURCE_AS_OF = datetime(2026, 9, 2, 12, 30, tzinfo=timezone.utc)
CROSS_SOURCE_LAT = 32.79
CROSS_SOURCE_LON = 34.99


class _FakeWeatherRepository:
    def __init__(self, record: StoredWeatherObservation) -> None:
        self._record = record

    def get_recent_observations_for_area_candidates(self, **kwargs):
        return (self._record,)

    def get_observations_by_ids(self, observation_ids):
        return tuple(record for record in (self._record,) if record.observation_id in observation_ids)


class _FakeSatelliteRepository:
    def get_by_ids(self, hotspot_ids):
        return (
            StoredSatelliteHotspot(
                id=1,
                hotspot=SatelliteHotspot(
                    latitude=CROSS_SOURCE_LAT,
                    longitude=CROSS_SOURCE_LON,
                    detected_at=CROSS_SOURCE_AS_OF - timedelta(hours=1),
                    confidence="h",
                    frp=50.0,
                ),
            ),
        )


class _NoLandCover:
    def get_land_cover_statistics(self, latitude, longitude, radius_km):
        return None


def _stored(observation: WeatherObservation) -> StoredWeatherObservation:
    observation.timestamp = CROSS_SOURCE_AS_OF - timedelta(minutes=5)
    return StoredWeatherObservation(
        observation_id=101,
        station_id=1,
        station=WeatherStation(
            external_station_id=observation.station_external_id,
            name="Station",
            latitude=CROSS_SOURCE_LAT,
            longitude=CROSS_SOURCE_LON,
        ),
        observation=observation,
    )


def _downstream_wind_kmh(observation: WeatherObservation) -> dict[str, float]:
    """What Fire Danger, Fire Severity and Fire Spread each receive as wind_speed_kmh."""
    record = _stored(observation)
    weather_repository = _FakeWeatherRepository(record)
    event = StoredFireEvent(
        id=10,
        event=FireEvent(
            latitude=CROSS_SOURCE_LAT,
            longitude=CROSS_SOURCE_LON,
            detected_at=CROSS_SOURCE_AS_OF - timedelta(minutes=20),
            updated_at=CROSS_SOURCE_AS_OF - timedelta(minutes=5),
            status=FireEventStatus.CONFIRMED,
            detection_confidence=0.85,
            methodology=FIRE_DETECTION_METHODOLOGY_NAME,
            methodology_version=FIRE_DETECTION_METHODOLOGY_VERSION,
        ),
        supporting_evidence=(FireEvidenceRef(FireEvidenceType.SATELLITE, 1),),
    )
    severity = StoredFireSeverityAssessment(
        assessment_id=500,
        assessment=FireSeverityAssessment(
            fire_event_id=10,
            assessed_at=CROSS_SOURCE_AS_OF - timedelta(minutes=1),
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
        weather_observation_ids=(101,),
        satellite_hotspot_ids=(1,),
        selected_frp_hotspot_id=1,
    )

    danger = FireDangerInputService(weather_repository=weather_repository).build_input(
        AssessmentArea(id="area", name="Area", latitude=CROSS_SOURCE_LAT, longitude=CROSS_SOURCE_LON, radius_km=5),
        CROSS_SOURCE_AS_OF,
    )
    severity_input = FireSeverityInputService(
        fire_event_repository=object(),
        weather_repository=weather_repository,
        satellite_hotspot_repository=_FakeSatelliteRepository(),
        land_cover_client=_NoLandCover(),
    ).prepare_input_for_event(event, CROSS_SOURCE_AS_OF)
    spread_context = FireSpreadInputService(
        fire_event_repository=object(),
        fire_severity_assessment_repository=object(),
        weather_repository=weather_repository,
    ).prepare_shared_context_for_event(event, CROSS_SOURCE_AS_OF, resolved_severity=severity)

    return {
        "danger": danger.input_data.wind_speed_kmh,
        "severity": severity_input.input_data.wind_speed_kmh,
        "spread": spread_context.wind_speed_kmh,
    }


def _ims_observation(wind_speed_ms: float, wind_gust_ms: float) -> WeatherObservation:
    return WeatherMapper.map_observation(
        {
            "stationId": 17,
            "data": [
                {
                    "datetime": "2026-09-02T12:25:00+00:00",
                    "channels": [
                        {"name": "TD", "value": 31.4, "valid": True},
                        {"name": "RH", "value": 30, "valid": True},
                        {"name": "WS", "value": wind_speed_ms, "valid": True},
                        {"name": "WD", "value": 240, "valid": True},
                        {"name": "WSmax", "value": wind_gust_ms, "valid": True},
                    ],
                }
            ],
        }
    )


def test_ims_10_ms_and_simulation_36_kmh_reach_every_consumer_as_the_same_36_kmh():
    ims = _ims_observation(wind_speed_ms=10, wind_gust_ms=15)
    simulation = WeatherObservation(
        station_external_id=901508,
        timestamp=CROSS_SOURCE_AS_OF,
        temperature=31.4,
        relative_humidity=30,
        wind_speed=36.0,
        wind_direction=240,
        wind_gust=54.0,
    )

    assert ims.wind_speed == pytest.approx(simulation.wind_speed) == pytest.approx(36.0)
    assert ims.wind_gust == pytest.approx(simulation.wind_gust) == pytest.approx(54.0)

    ims_downstream = _downstream_wind_kmh(ims)
    simulation_downstream = _downstream_wind_kmh(simulation)
    for consumer in ("danger", "severity", "spread"):
        assert ims_downstream[consumer] == pytest.approx(36.0), consumer
        assert simulation_downstream[consumer] == pytest.approx(36.0), consumer


def test_simulation_generator_km_h_matches_equivalent_ims_m_s_after_mapping():
    generated = WeatherDataGenerator(seed=7).generate(ScenarioType.ACTIVE_FIRE, timestamp=CROSS_SOURCE_AS_OF)
    for measurement in generated.measurements:
        simulated = measurement.observation
        ims = _ims_observation(wind_speed_ms=simulated.wind_speed / 3.6, wind_gust_ms=simulated.wind_gust / 3.6)

        assert ims.wind_speed == pytest.approx(simulated.wind_speed)
        assert ims.wind_gust == pytest.approx(simulated.wind_gust)


def test_simulation_generator_stores_profile_km_h_unconverted():
    from src.simulation.generators.weather_data_generator import WEATHER_SCENARIO_PROFILES

    profile = WEATHER_SCENARIO_PROFILES[ScenarioType.ACTIVE_FIRE]
    generated = WeatherDataGenerator(seed=7).generate(ScenarioType.ACTIVE_FIRE, timestamp=CROSS_SOURCE_AS_OF)
    for measurement in generated.measurements:
        low, high = profile.wind_speed_kmh
        assert low <= measurement.observation.wind_speed <= high
